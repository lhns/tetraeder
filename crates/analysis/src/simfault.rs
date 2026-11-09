//! Synthetic observations from a hardware graph with injected faults, for testing the detector
//! (AGENTS.md rule 4).
//!
//! The plan is simple: per token every resource does a fixed amount of work chosen so the nominal
//! busy times are of similar size (the cost-model formulas turn bytes into seconds). Faults change
//! rates or volumes and set the signals a real sampler would see. The generator writes signals
//! only, never the injected label. Lane-group contention is not modelled.

use std::collections::BTreeMap;

use tetraeder_costmodel::{OpShape, Roofline, transfer_time};
use tetraeder_profiler::{DeviceKind, HardwareGraph, LinkKind};

use crate::detect::FaultClass;
use crate::observation::{
    EnvSignals, Expected, Observation, PcieSignal, Throttle, ThrottleReason, Window,
};

/// Nominal busy seconds per token of each GPU, the CPU and the disk.
const T_COMPUTE: f64 = 0.025;
/// Nominal busy seconds per token of a PCIe link (expert streaming).
const T_PCIE: f64 = 0.020;
/// Nominal host seconds per token (scheduling, sampling).
const T_HOST: f64 = 0.004;
/// Activation bytes per pipeline hop.
const ACT_BYTES: f64 = 65536.0;
const SLOTS: u32 = 64;
const HIT_RATE: f64 = 0.9;
/// Swapping slows the CPU and host work by this factor: the stall lands in whatever touches the
/// swapped pages (the CPU testbed run: compute 60x and CPU busy time 2.5x slower, host 2x).
const SWAP_SLOWDOWN: f64 = 3.0;

#[derive(Debug, Clone)]
pub struct SimSpec {
    pub seed: u64,
    pub batch: u32,
    pub faults: Vec<FaultClass>,
    pub windows: usize,
    pub tokens_per_window: u64,
    /// Sigma of the lognormal per-window noise on every resource's busy time.
    pub noise: f64,
    /// Attach the calibrated expectations (nominal rates, band 0.1).
    pub expected: bool,
}

impl SimSpec {
    pub fn new(seed: u64, faults: &[FaultClass]) -> Self {
        Self {
            seed,
            batch: 1,
            faults: faults.to_vec(),
            windows: 8,
            tokens_per_window: 64,
            noise: 0.03,
            expected: true,
        }
    }
}

/// Fault classes the generator can inject on this hardware (it needs the resource).
pub fn applicable_faults(hw: &HardwareGraph) -> Vec<FaultClass> {
    [
        FaultClass::GpuThrottle,
        FaultClass::CpuCotenant,
        FaultClass::DiskSlow,
        FaultClass::VramPressure,
        FaultClass::PcieDegraded,
        FaultClass::LinkSlow,
        FaultClass::Swapping,
    ]
    .into_iter()
    .filter(|c| applicable(hw, *c))
    .collect()
}

fn has_kind(hw: &HardwareGraph, kind: DeviceKind) -> bool {
    hw.devices.iter().any(|d| d.kind == kind)
}

/// The first GPU with a PCIe link to a CPU, with that link's id.
fn gpu_pcie(hw: &HardwareGraph) -> Option<(String, String)> {
    pcie_links(hw).into_iter().next()
}

/// `(gpu id, link id)` of every PCIe link between a GPU and a CPU.
fn pcie_links(hw: &HardwareGraph) -> Vec<(String, String)> {
    hw.links
        .iter()
        .filter(|l| l.kind == LinkKind::Pcie)
        .filter_map(|l| {
            let a = hw.device(&l.a)?;
            let b = hw.device(&l.b)?;
            match (a.kind, b.kind) {
                (DeviceKind::Gpu, DeviceKind::Cpu) => Some((a.id.clone(), l.id.clone())),
                (DeviceKind::Cpu, DeviceKind::Gpu) => Some((b.id.clone(), l.id.clone())),
                _ => None,
            }
        })
        .collect()
}

/// The first GPU whose PCIe maximum width is at least 8, so a drop to a quarter is real.
fn degradable(hw: &HardwareGraph) -> Option<(String, String)> {
    pcie_links(hw).into_iter().find(|(g, _)| {
        hw.device(g)
            .and_then(|d| d.pcie)
            .is_some_and(|p| p.width_max >= 8)
    })
}

pub fn applicable(hw: &HardwareGraph, class: FaultClass) -> bool {
    let gpu = has_kind(hw, DeviceKind::Gpu);
    match class {
        FaultClass::GpuThrottle => gpu,
        FaultClass::CpuCotenant | FaultClass::Swapping => has_kind(hw, DeviceKind::Cpu),
        FaultClass::DiskSlow => has_kind(hw, DeviceKind::Disk),
        FaultClass::VramPressure => {
            gpu && (has_kind(hw, DeviceKind::Cpu) || has_kind(hw, DeviceKind::Disk))
        }
        FaultClass::PcieDegraded => degradable(hw).is_some(),
        FaultClass::LinkSlow => hw.links.iter().any(|l| l.kind == LinkKind::Network),
        FaultClass::ModelMismatch => false,
    }
}

/// One resource of the plan: work per token and what the nominal hardware does with it.
#[derive(Clone)]
struct Res {
    id: String,
    /// Work units per token (bytes, or 1 for the host).
    units: f64,
    /// Fixed seconds per token (link latency times hops).
    alpha_s: f64,
    /// Units per second.
    rate: f64,
}

impl Res {
    fn busy(&self) -> f64 {
        transfer_time(self.alpha_s, self.rate, self.units)
    }
}

/// Memory-bound op on a device: the roofline with one FLOP per byte.
fn device_res(id: String, bytes: f64, flops_per_s: f64, bandwidth: f64) -> Res {
    let t = Roofline {
        shape: OpShape {
            flops_per_token: bytes,
            fixed_bytes: bytes,
            bytes_per_token: 0.0,
        },
        flops_per_s,
        bytes_per_s: bandwidth,
    }
    .time(1.0);
    Res {
        id,
        units: bytes,
        alpha_s: 0.0,
        rate: bytes / t,
    }
}

fn plan(hw: &HardwareGraph) -> Vec<Res> {
    let mut v = Vec::new();
    for d in &hw.devices {
        let (Some(c), Some(b)) = (d.compute, d.bandwidth_bytes_per_s) else {
            if d.kind == DeviceKind::Disk
                && let Some(b) = d.bandwidth_bytes_per_s
            {
                v.push(Res {
                    id: format!("disk:{}", d.id),
                    units: T_COMPUTE * b.value,
                    alpha_s: 0.0,
                    rate: b.value,
                });
            }
            continue;
        };
        let prefix = if d.kind == DeviceKind::Gpu {
            "gpu:"
        } else {
            ""
        };
        let id = if d.kind == DeviceKind::Cpu {
            "cpu".to_string()
        } else {
            format!("{prefix}{}", d.id)
        };
        v.push(device_res(id, T_COMPUTE * b.value, c.flops_per_s, b.value));
    }
    for l in &hw.links {
        match l.kind {
            // The storage link is part of the disk resource.
            LinkKind::Storage => {}
            LinkKind::Pcie => v.push(Res {
                id: format!("link:{}", l.id),
                units: T_PCIE * l.beta_bytes_per_s.value,
                alpha_s: 0.0,
                rate: l.beta_bytes_per_s.value,
            }),
            LinkKind::P2p | LinkKind::Network => v.push(Res {
                id: format!("link:{}", l.id),
                units: 2.0 * ACT_BYTES,
                alpha_s: 2.0 * l.alpha_s.value,
                rate: l.beta_bytes_per_s.value,
            }),
        }
    }
    v.push(Res {
        id: "host".into(),
        units: 1.0,
        alpha_s: 0.0,
        rate: 1.0 / T_HOST,
    });
    v
}

fn find<'a>(res: &'a mut [Res], id: &str) -> Option<&'a mut Res> {
    res.iter_mut().find(|r| r.id == id)
}

/// Splitmix64 with a Box-Muller normal; enough for seeded noise.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// Build one observation. Errors when a fault needs a resource the hardware lacks.
pub fn simulate(hw: &HardwareGraph, spec: &SimSpec) -> Result<Observation, String> {
    for c in &spec.faults {
        if !applicable(hw, *c) {
            return Err(format!("{c:?} is not applicable to this hardware"));
        }
    }
    let nominal = plan(hw);
    let mut res = nominal.clone();
    let mut rng = Rng(spec.seed ^ 0xD1B5_4A32_D192_ED03);
    let mut env = EnvSignals::default();
    let gpus: Vec<String> = hw
        .devices
        .iter()
        .filter(|d| d.kind == DeviceKind::Gpu)
        .map(|d| d.id.clone())
        .collect();
    let cpu = hw.devices.iter().find(|d| d.kind == DeviceKind::Cpu);
    let offload = cpu.is_some() || has_kind(hw, DeviceKind::Disk);

    // Healthy signals with a little noise, including a harmless software throttle reason.
    env.cpu_quota_cores = cpu.and_then(|c| c.threads).map_or(0.0, f64::from);
    env.cpu_other_cores = env.cpu_quota_cores * 0.015 * rng.unit();
    env.swap_in_bytes_per_s = 2e6 * rng.unit();
    env.disk_other_read_bytes_per_s = 5e6 * rng.unit();
    for g in hw.devices.iter().filter(|d| d.kind == DeviceKind::Gpu) {
        env.vram_total_bytes.insert(g.id.clone(), g.capacity_bytes);
        env.vram_other_bytes.insert(
            g.id.clone(),
            (g.capacity_bytes as f64 * 0.012 * rng.unit()) as u64,
        );
        if let Some(p) = g.pcie {
            let link = pcie_links(hw)
                .into_iter()
                .find(|(d, _)| *d == g.id)
                .map(|(_, l)| l)
                .unwrap_or_default();
            env.pcie.push(PcieSignal {
                device: g.id.clone(),
                link,
                gen_current: p.gen_max,
                width_current: p.width_max,
                gen_max: p.gen_max,
                width_max: p.width_max,
            });
        }
    }
    if offload {
        env.expert_hit_rate = Some(HIT_RATE + 0.02 * (rng.unit() - 0.5));
        env.slots_planned = Some(SLOTS);
        env.slots_actual = Some(SLOTS);
    }
    // Memory signals use their own stream so the other signals stay as they were. PSI is there on
    // even seeds only, so both detector paths (with and without PSI) are exercised.
    let mut mem_rng = Rng(spec.seed ^ 0x5DEE_CE66_D1CE_4E5B);
    env.major_faults_per_s = 200.0 * mem_rng.unit();
    let psi = spec.seed.is_multiple_of(2);
    if spec.seed.is_multiple_of(3)
        && let Some(g) = gpus.first()
    {
        env.throttle.push(Throttle {
            device: g.clone(),
            reason: ThrottleReason::Sw,
        });
    }

    let primary = gpu_pcie(hw).map(|(g, l)| (g, format!("link:{l}")));
    for fault in &spec.faults {
        match fault {
            FaultClass::GpuThrottle => {
                let g = &gpus[0];
                if let Some(r) = find(&mut res, &format!("gpu:{g}")) {
                    r.rate *= 0.6;
                }
                env.throttle.clear();
                env.throttle.push(Throttle {
                    device: g.clone(),
                    reason: if spec.seed.is_multiple_of(2) {
                        ThrottleReason::Power
                    } else {
                        ThrottleReason::Thermal
                    },
                });
            }
            FaultClass::CpuCotenant => {
                if let Some(r) = find(&mut res, "cpu") {
                    r.rate *= 0.5;
                }
                env.cpu_other_cores = env.cpu_quota_cores / 2.0;
            }
            FaultClass::DiskSlow => {
                let disk = nominal.iter().find(|r| r.id.starts_with("disk:")).cloned();
                if let Some(d) = disk
                    && let Some(r) = find(&mut res, &d.id)
                {
                    r.rate *= 0.3;
                    if spec.seed.is_multiple_of(2) {
                        env.io_max_bytes_per_s = Some(d.rate * 0.3);
                    }
                }
            }
            FaultClass::VramPressure => {
                let g = primary
                    .as_ref()
                    .map(|p| p.0.clone())
                    .unwrap_or_else(|| gpus[0].clone());
                let total = hw.device(&g).map_or(0, |d| d.capacity_bytes);
                env.vram_other_bytes
                    .insert(g.clone(), (total as f64 * 0.3) as u64);
                env.slots_actual = Some((f64::from(SLOTS) * 0.6) as u32);
                env.expert_hit_rate = Some(0.6 + 0.02 * (rng.unit() - 0.5));
                // More experts stream in or run on the CPU; fewer are read from VRAM.
                for r in &mut res {
                    if r.id.starts_with("disk:") || primary.as_ref().is_some_and(|p| p.1 == r.id) {
                        r.units *= 1.8;
                    } else if r.id == "cpu" {
                        r.units *= 1.5;
                    } else if r.id == format!("gpu:{g}") {
                        r.units *= 0.8;
                    }
                }
            }
            FaultClass::PcieDegraded => {
                let (g, l) = degradable(hw).expect("checked applicable");
                if let Some(r) = find(&mut res, &format!("link:{l}")) {
                    r.rate /= 4.0;
                }
                if let Some(p) = env.pcie.iter_mut().find(|p| p.device == g) {
                    p.width_current = p.width_max / 4;
                }
            }
            FaultClass::LinkSlow => {
                let id = hw
                    .links
                    .iter()
                    .find(|l| l.kind == LinkKind::Network)
                    .map(|l| format!("link:{}", l.id))
                    .expect("checked applicable");
                if let Some(r) = find(&mut res, &id) {
                    r.alpha_s *= 10.0;
                    r.rate /= 5.0;
                }
            }
            FaultClass::Swapping => {
                env.swap_in_bytes_per_s = 2e8 * (0.9 + 0.2 * mem_rng.unit());
                env.major_faults_per_s += env.swap_in_bytes_per_s / 4096.0;
                for r in &mut res {
                    if r.id == "cpu" || r.id == "host" {
                        r.rate /= SWAP_SLOWDOWN;
                    }
                }
            }
            FaultClass::ModelMismatch => {}
        }
    }

    let serialized = spec.batch <= 1;
    let swapping = spec.faults.contains(&FaultClass::Swapping);
    let tokens = spec.tokens_per_window as f64;
    let mut windows = Vec::new();
    let mut stall_shares = Vec::new();
    for _ in 0..spec.windows {
        let mut stalled = 0.0;
        let mut busy_s = BTreeMap::new();
        let mut bytes = BTreeMap::new();
        let mut split_s: BTreeMap<String, f64> = BTreeMap::new();
        for r in &res {
            let b = r.busy() * tokens * (spec.noise * rng.normal()).exp();
            busy_s.insert(r.id.clone(), b);
            if swapping && (r.id == "cpu" || r.id == "host") {
                stalled += b * (1.0 - 1.0 / SWAP_SLOWDOWN);
            }
            bytes.insert(r.id.clone(), r.units * tokens);
            let key = if r.id.starts_with("gpu:") {
                "compute"
            } else if r.id == "cpu" {
                "cpu_experts"
            } else if r.id.starts_with("disk:")
                || pcie_links(hw)
                    .iter()
                    .any(|p| r.id == format!("link:{}", p.1))
            {
                "expert_load_wait"
            } else if r.id.starts_with("link:") {
                "transfer_wait"
            } else {
                "host"
            };
            *split_s.entry(key.into()).or_default() += b;
        }
        let wall_s = if serialized {
            busy_s.values().sum()
        } else {
            busy_s.values().copied().fold(0.0, f64::max)
        };
        stall_shares.push((stalled / wall_s).min(1.0));
        windows.push(Window {
            tokens: spec.tokens_per_window,
            wall_s,
            busy_s,
            bytes,
            split_s,
        });
    }

    if psi {
        let mean = stall_shares.iter().sum::<f64>() / stall_shares.len().max(1) as f64;
        env.memory_stall_share = Some(mean + 0.01 * mem_rng.unit());
    }

    let expected = spec.expected.then(|| Expected {
        rates: nominal
            .iter()
            .map(|r| (r.id.clone(), r.units / r.busy()))
            .collect(),
        band: 0.1,
    });
    Ok(Observation {
        id: format!("sim-{}", spec.seed),
        batch: spec.batch,
        windows,
        env,
        expected,
    })
}
