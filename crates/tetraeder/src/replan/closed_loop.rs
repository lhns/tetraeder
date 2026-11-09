//! The drift loop of ADR 0056 on a fake clock: the builder runs a plan the disk binds, the disk
//! slows to a quarter at t = 300 s, the residual monitor ticks every 60 fake seconds and reports
//! a drift after five ticks over the band, the planner scales the disk curve by the measured
//! step and proposes a plan that reads fewer bytes, and the replanner switches. No sleeping, no
//! engine: the step times are the cost model's own on the slowed graph.

use super::monitor::ResidualMonitor;
use super::source::{HwSource, PlannerSource, Shared};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tetraeder_control::{Clock, Config, EngineControl, FakeClock, Outcome, Replanner};
use tetraeder_plan::Plan;
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::score::RTX3090_PRIOR;
use tetraeder_planner::{
    Applies, Placement, ServeEngine, ServeInput, Workload, budget_of, candidates, predict_plan_for,
    predict_plan_tier, recognize, uniform_tier,
};
use tetraeder_profiler::HardwareGraph;
use tetraeder_telemetry::{Phase, StepRing, StepSample};

/// Records what the replanner asks of the engine; everything succeeds at once.
#[derive(Default)]
struct FakeEngine {
    log: Vec<String>,
}

impl EngineControl for FakeEngine {
    fn stop_admitting(&mut self) {
        self.log.push("stop".into());
    }
    fn in_flight(&self) -> usize {
        0
    }
    fn wait_drained(&mut self, _: f64) -> bool {
        true
    }
    fn restart(&mut self, _: &Plan) -> Result<(), String> {
        self.log.push("restart".into());
        Ok(())
    }
    fn healthy(&mut self) -> bool {
        true
    }
    fn resume(&mut self) {
        self.log.push("resume".into());
    }
    fn apply_knobs(&mut self, _: &Plan) -> Result<(), String> {
        Err("this engine reloads".into())
    }
    fn supports_knobs(&self) -> bool {
        false
    }
}

fn sample(step_s: f64) -> StepSample {
    StepSample {
        phase: Phase::Decode,
        stage: 0,
        graphs: false,
        duration: Duration::from_secs_f64(step_s),
        tokens: 1,
        segments: 1,
    }
}

// hardware: none
#[test]
fn a_slowing_disk_re_picks_a_plan_with_fewer_disk_reads() {
    let hw = HardwareGraph::from_toml(RTX3090_PRIOR).unwrap();
    let layout = fixtures::moe();
    let mut inp = ServeInput {
        hw: hw.clone(),
        layout: layout.clone(),
        budget: budget_of(&hw, 16, None, None),
        ctx: 4096,
        workload: Workload {
            prompt: 2048,
            r#gen: 128,
        },
        engine: ServeEngine::Builder,
        applies: Applies::today(),
        tier: None,
    };
    let slots = |p: &Plan| match recognize(p, &layout).unwrap().0 {
        Placement::Builder {
            vram_slots,
            ram_slots,
        } => (vram_slots, ram_slots),
        other => panic!("{other:?}"),
    };
    let pairs: Vec<(u32, u32)> = candidates(&inp).iter().map(slots).collect();
    inp.tier = Some(uniform_tier(&layout, &pairs));
    let disk = |p: &Plan| {
        let (v, r) = slots(p);
        inp.tier
            .as_ref()
            .unwrap()
            .iter()
            .find(|o| o.vram_slots == v && o.ram_slots == r)
            .unwrap()
            .decode
            .disk_bytes_per_token
    };
    // the running plan: the fastest one the disk binds
    let cur = candidates(&inp)
        .into_iter()
        .filter_map(|p| {
            let pred = predict_plan_for(&inp, &p).ok()?;
            (pred.binding == "read:nvme0").then_some((pred.tg_tok_s, p))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .expect("a plan the disk binds")
        .1;
    let predicted = predict_plan_for(&inp, &cur).unwrap();

    // the box's disk delivers a quarter of the curve from t = 300 s
    let mut slow = hw.clone();
    for d in &mut slow.devices {
        for q in &mut d.queue_curve {
            q.bytes_per_s *= 0.25;
        }
    }
    let step_at = |t: f64, plan: &Plan| {
        let g = if t > 300.0 { &slow } else { &hw };
        let p = predict_plan_tier(g, &layout, plan, &inp.workload, inp.tier.as_deref()).unwrap();
        1.0 / p.tg_tok_s
    };

    let clock = FakeClock::at(0, 0);
    let t0 = clock.now_s();
    let shared = Shared::new(inp.clone());
    let source = PlannerSource::new(
        Arc::clone(&shared),
        HwSource::File(PathBuf::from("unused.toml")),
        (16, None, None),
        Arc::new(Mutex::new(3.0)),
    );
    let mut r = Replanner::new(
        Config::default(),
        cur.clone(),
        source,
        FakeEngine::default(),
        clock.clone(),
    );
    let ring = StepRing::new(4096);
    let mut monitor = ResidualMonitor::new(0.0);
    monitor.set_prediction(Some(1.0 / predicted.tg_tok_s), None);

    let (mut ticks_after_slowdown, mut result) = (0, None);
    for _ in 0..40 {
        clock.sleep(60.0);
        let t = clock.now_s() - t0;
        let step = step_at(t, r.current());
        for _ in 0..20 {
            ring.push(sample(step));
        }
        ticks_after_slowdown += u32::from(t > 300.0);
        if let Some(trigger) = monitor.tick(&ring, None) {
            shared.lock().unwrap().measured_step_s = monitor.last_median_s;
            result = Some((t, r.on_trigger(trigger)));
            break;
        }
    }
    let (t, outcome) = result.expect("the monitor reports the drift");
    assert_eq!(
        ticks_after_slowdown, 5,
        "five ticks over the band, at t = {t}"
    );
    assert_eq!(outcome, Outcome::Switched);
    assert_eq!(r.engine().log, ["stop", "restart", "resume"]);
    let (old, new) = (slots(&cur), slots(r.current()));
    assert!(new.0 > old.0 || new.1 > old.1, "{old:?} -> {new:?}");
    assert!(disk(r.current()) < disk(&cur));
}
