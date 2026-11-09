//! The planner as the replanner's `PlanSource` (ADR 0055, 3).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tetraeder_control::{PlanSource, Proposal, Trigger};
use tetraeder_plan::Plan;
use tetraeder_planner::diff::DRAIN_S;
use tetraeder_planner::{
    Priors, Proposed, ServeInput, budget_of, correct_for_drift, fill_from_priors, predict_plan,
    propose,
};
use tetraeder_profiler::{DiscoverOpts, HardwareGraph, discover};

/// Where the hardware graph comes from, now and when a trigger asks to look again.
#[derive(Debug, Clone)]
pub enum HwSource {
    /// A `HardwareGraph` as TOML or JSON (`.json`), typically calibrated by `tetraeder-plan score`.
    File(PathBuf),
    /// Discovery with GPU rates from the priors by name.
    Discover,
}

impl HwSource {
    /// The graph and notes about what could not be filled. Discovery probes RAM bandwidth only
    /// at startup (`probe`).
    pub fn read(&self, probe: bool) -> Result<(HardwareGraph, Vec<String>), String> {
        match self {
            Self::File(p) => {
                let text = std::fs::read_to_string(p)
                    .map_err(|e| format!("cannot read {}: {e}", p.display()))?;
                let json = p.extension().is_some_and(|e| e == "json");
                let hw = if json {
                    HardwareGraph::from_json(&text).map_err(|e| e.to_string())?
                } else {
                    HardwareGraph::from_toml(&text).map_err(|e| e.to_string())?
                };
                hw.validate().map_err(|e| format!("{}: {e}", p.display()))?;
                Ok((hw, vec![]))
            }
            Self::Discover => {
                let mut hw = discover(&DiscoverOpts {
                    probe_ram: probe,
                    ..DiscoverOpts::default()
                });
                let mut notes = hw.notes.clone();
                notes.extend(fill_from_priors(&mut hw, &Priors::builtin()));
                Ok((hw, notes))
            }
        }
    }
}

/// Largest spread of a measured bandwidth in the graph: how noisy its numbers are.
pub fn band_of(hw: &HardwareGraph) -> f64 {
    hw.devices
        .iter()
        .filter_map(|d| d.bandwidth_bytes_per_s.map(|b| b.spread))
        .fold(0.0, f64::max)
}

/// What the control loop and the source share: the planner's belief and the hand-over of
/// measurements and proposals.
pub struct Shared {
    pub inp: ServeInput,
    /// Median decode step of the last residual tick; the drift correction uses it.
    pub measured_step_s: Option<f64>,
    /// The latest proposal, to set the monitor's prediction after a switch.
    pub last: Option<Proposed>,
}

impl Shared {
    pub fn new(inp: ServeInput) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            inp,
            measured_step_s: None,
            last: None,
        }))
    }
}

pub struct PlannerSource {
    shared: Arc<Mutex<Shared>>,
    hw_source: HwSource,
    quota: u32,
    vram: Option<u64>,
    ram: Option<u64>,
    last_load_s: Arc<Mutex<f64>>,
}

impl PlannerSource {
    pub fn new(
        shared: Arc<Mutex<Shared>>,
        hw_source: HwSource,
        (quota, vram, ram): (u32, Option<u64>, Option<u64>),
        last_load_s: Arc<Mutex<f64>>,
    ) -> Self {
        Self {
            shared,
            hw_source,
            quota,
            vram,
            ram,
            last_load_s,
        }
    }
}

/// Scales the device the running plan is bound by so its predicted step equals the measured one.
fn correct(s: &mut Shared, current: &Plan) {
    let Some(measured) = s.measured_step_s else {
        return;
    };
    let Ok(pred) = predict_plan(&s.inp.hw, &s.inp.layout, current, &s.inp.workload) else {
        return;
    };
    if let Some(g) = correct_for_drift(&s.inp.hw, &pred, measured) {
        s.inp.hw = g;
    }
}

impl PlanSource for PlannerSource {
    fn propose(&mut self, current: &Plan, trigger: &Trigger) -> Option<Proposal> {
        let mut s = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        match trigger {
            Trigger::Drift { .. } => correct(&mut s, current),
            // A read that fails keeps the old belief.
            _ => match self.hw_source.read(false) {
                Ok((hw, _)) => {
                    s.inp.budget = budget_of(&hw, self.quota, self.vram, self.ram);
                    s.inp.hw = hw;
                }
                Err(e) => {
                    eprintln!("replan: cannot re-read the hardware, keeping the old graph: {e}");
                }
            },
        }
        let p = propose(&s.inp, current)?;
        let load = *self.last_load_s.lock().unwrap_or_else(|e| e.into_inner());
        let proposal = Proposal {
            plan: p.plan.clone(),
            predicted_gain_s_per_s: p.gain,
            switch_cost_s: DRAIN_S + load,
            knobs_only: p.plan.replicas == current.replicas,
        };
        s.last = Some(p);
        Some(proposal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetraeder_planner::layout::fixtures;
    use tetraeder_planner::{Applies, ServeEngine, Workload, candidates};

    const CPU_ONLY: &str = include_str!("../../../planner/fixtures/cpu_only.toml");

    fn setup(dir: &std::path::Path) -> (PlannerSource, Arc<Mutex<Shared>>, PathBuf, Plan) {
        let file = dir.join("hw.toml");
        std::fs::write(&file, CPU_ONLY).unwrap();
        let src = HwSource::File(file.clone());
        let (hw, _) = src.read(false).unwrap();
        let layout = fixtures::moe();
        let inp = ServeInput {
            budget: budget_of(&hw, 16, None, None),
            hw,
            layout,
            ctx: 4096,
            workload: Workload {
                prompt: 2048,
                r#gen: 128,
            },
            engine: ServeEngine::Llama,
            applies: Applies::today(),
        };
        let cur = tetraeder_planner::pick(&inp).unwrap().plan;
        let shared = Shared::new(inp);
        let load = Arc::new(Mutex::new(3.0));
        let ps = PlannerSource::new(Arc::clone(&shared), src, (16, None, None), load);
        (ps, shared, file, cur)
    }

    // hardware: none
    #[test]
    fn a_device_change_rereads_the_file_and_switches_when_the_plan_stops_fitting() {
        let dir = std::env::temp_dir().join(format!("tetraeder-source-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (mut ps, shared, file, cur) = setup(&dir);
        assert_eq!(cur.knobs.threads, Some(8));
        // nothing changed: nothing to propose
        assert!(ps.propose(&cur, &Trigger::DeviceChange).is_none());
        // the CPU shrinks to 4 threads
        std::fs::write(&file, CPU_ONLY.replace("threads = 8", "threads = 4")).unwrap();
        let p = ps.propose(&cur, &Trigger::DeviceChange).expect("proposal");
        assert_eq!(p.plan.knobs.threads, Some(4));
        assert_eq!(p.predicted_gain_s_per_s, 1.0);
        assert!((p.switch_cost_s - (DRAIN_S + 3.0)).abs() < 1e-12);
        assert!(!p.knobs_only || p.plan.replicas == cur.replicas);
        assert_eq!(shared.lock().unwrap().last.as_ref().unwrap().plan, p.plan);
        assert!(candidates(&shared.lock().unwrap().inp).contains(&p.plan));
        // a file that cannot be read keeps the belief
        std::fs::write(&file, "not toml [").unwrap();
        let again = ps.propose(&p.plan, &Trigger::DeviceChange);
        assert!(again.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    // hardware: none
    #[test]
    fn drift_scales_the_bound_device_before_the_pick() {
        let dir = std::env::temp_dir().join(format!("tetraeder-source-d-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (mut ps, shared, _, cur) = setup(&dir);
        let before = shared
            .lock()
            .unwrap()
            .inp
            .hw
            .device("cpu0")
            .unwrap()
            .clone();
        let pred = {
            let s = shared.lock().unwrap();
            predict_plan(&s.inp.hw, &s.inp.layout, &cur, &s.inp.workload).unwrap()
        };
        // the box is twice as slow as predicted
        shared.lock().unwrap().measured_step_s = Some(2.0 / pred.tg_tok_s);
        let _ = ps.propose(&cur, &Trigger::Drift { residual: 1.0 });
        let after = shared
            .lock()
            .unwrap()
            .inp
            .hw
            .device("cpu0")
            .unwrap()
            .clone();
        assert!(
            after.bandwidth_bytes_per_s.unwrap().value
                < before.bandwidth_bytes_per_s.unwrap().value
        );
        let s = shared.lock().unwrap();
        let again = predict_plan(&s.inp.hw, &s.inp.layout, &cur, &s.inp.workload).unwrap();
        // one factor on the rates leaves the per-op cost, so the step lands between the old
        // prediction and the measurement
        let (old, measured, new) = (
            1.0 / pred.tg_tok_s,
            2.0 / pred.tg_tok_s,
            1.0 / again.tg_tok_s,
        );
        assert!(
            new > old * 1.5 && new <= measured * 1.0001,
            "{old} {new} {measured}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
