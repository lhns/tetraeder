//! Simulated hardware shared by the serve tests (AGENTS.md rule 4).

#![allow(dead_code)]

use tetraeder_planner::{Applies, ServeEngine, ServeInput, Workload, budget_of};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

const TWO_GPU: &str = include_str!("../../../costmodel/fixtures/two_gpu_cpu_nvme.toml");
const RTX3090: &str = include_str!("../../fixtures/rtx3090_prior.toml");
const CPU_ONLY: &str = include_str!("../../fixtures/cpu_only.toml");

pub const WORK: Workload = Workload {
    prompt: 2048,
    r#gen: 128,
};
pub const CTX: u32 = 4096;
pub const QUOTA: u32 = 16;

pub fn without_gpus(hw: &HardwareGraph) -> HardwareGraph {
    let mut g = hw.clone();
    g.devices.retain(|d| d.kind != DeviceKind::Gpu);
    let ids: Vec<String> = g.devices.iter().map(|d| d.id.clone()).collect();
    g.links.retain(|l| ids.contains(&l.a) && ids.contains(&l.b));
    g
}

/// Every graph with a `cpu0`: the costmodel fixtures that have one (and a CPU-only copy), the
/// 3090 prior and the CPU-only fixture.
pub fn graphs() -> Vec<(&'static str, HardwareGraph)> {
    let two = HardwareGraph::from_toml(TWO_GPU).unwrap();
    vec![
        ("two_gpu_cpu_nvme_cpu_only", without_gpus(&two)),
        ("two_gpu_cpu_nvme", two),
        ("rtx3090_prior", HardwareGraph::from_toml(RTX3090).unwrap()),
        ("cpu_only", HardwareGraph::from_toml(CPU_ONLY).unwrap()),
    ]
}

pub fn has_gpu(hw: &HardwareGraph) -> bool {
    hw.devices.iter().any(|d| d.kind == DeviceKind::Gpu)
}

pub fn input(
    hw: &HardwareGraph,
    layout: &tetraeder_planner::ModelLayout,
    engine: ServeEngine,
) -> ServeInput {
    ServeInput {
        hw: hw.clone(),
        layout: layout.clone(),
        budget: budget_of(hw, QUOTA, None, None),
        ctx: CTX,
        workload: WORK,
        engine,
        applies: Applies::today(),
    }
}

/// Scales the rates of device `id` by `f` (a co-tenant or a throttle: `f < 1`).
pub fn scale(hw: &HardwareGraph, id: &str, f: f64) -> HardwareGraph {
    let mut g = hw.clone();
    if let Some(d) = g.devices.iter_mut().find(|d| d.id == id) {
        if let Some(c) = d.compute.as_mut() {
            c.flops_per_s *= f;
        }
        if let Some(b) = d.bandwidth_bytes_per_s.as_mut() {
            b.value *= f;
        }
        for p in &mut d.thread_curve {
            p.flops_per_s *= f;
            p.bytes_per_s *= f;
        }
    }
    g
}
