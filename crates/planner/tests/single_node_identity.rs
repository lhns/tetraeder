//! ADR 0091 C7, step 2: every existing hardware fixture keeps its plans and predictions when
//! `Device` gains `node` and the cluster code lands. The golden file was written by the code
//! before step 2 (`TETRAEDER_UPDATE_GOLDEN=1` rewrites it; do that only for a change that is
//! meant to move single-node numbers). Floats are compared by bit pattern.
//!
//! ADR 0092 section 7: the golden is two files. `single_node_predictions.txt` holds every
//! prediction that cost model 6 must leave bit-identical and is never rewritten for it;
//! `single_node_predictions_streamed.txt` holds the builder plans whose RAM tier does not cover
//! the experts outside VRAM, which cost model 6 may move once, in the commit that sets
//! `MODEL_VERSION` 6.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;
use tetraeder_plan::{Format, Hints, ModelDesc, Phase, Plan, UnitSize};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{
    Applies, Budget, ModelLayout, NullQuality, Placement, SearchInput, ServeEngine, ServeInput,
    Workload, budget_of, candidates, predict_plan, recognize, search,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

const GRAPHS: [(&str, &str); 9] = [
    (
        "costmodel/decode_clocks_host20210",
        include_str!("../../costmodel/fixtures/decode_clocks_host20210.toml"),
    ),
    (
        "costmodel/four_3090_hoststaged",
        include_str!("../../costmodel/fixtures/four_3090_hoststaged.toml"),
    ),
    (
        "costmodel/two_3090_hoststaged",
        include_str!("../../costmodel/fixtures/two_3090_hoststaged.toml"),
    ),
    (
        "costmodel/two_gpu_cpu_nvme",
        include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml"),
    ),
    (
        "costmodel/wan_two_node",
        include_str!("../../costmodel/fixtures/wan_two_node.toml"),
    ),
    (
        "analysis/rtx3090_box",
        include_str!("../../analysis/fixtures/rtx3090_box.toml"),
    ),
    (
        "planner/rtx3090_prior",
        include_str!("../fixtures/rtx3090_prior.toml"),
    ),
    (
        "planner/cpu_only",
        include_str!("../fixtures/cpu_only.toml"),
    ),
    (
        "profiler/two_gpu_host",
        include_str!("../../profiler/fixtures/two_gpu_host.toml"),
    ),
];

const GOLDEN: &str = "tests/golden/single_node_predictions.txt";
const GOLDEN_STREAMED: &str = "tests/golden/single_node_predictions_streamed.txt";

fn size(params: u64) -> UnitSize {
    UnitSize {
        params,
        flops_per_token: 2.0 * params as f64,
        source_format: Format::F16,
        roles: Vec::new(),
    }
}

fn model(name: &str, layers: u32, experts: u32) -> ModelDesc {
    ModelDesc {
        name: name.into(),
        layers,
        heads: 16,
        kv_heads: 8,
        experts,
        experts_used: if experts > 0 { 2 } else { 0 },
        embed: size(100_000_000),
        attn: size(30_000_000),
        mlp: size(60_000_000),
        expert: (experts > 0).then(|| size(50_000_000)),
        head: size(100_000_000),
        kv_bytes_per_token_per_layer: 4096,
        activation_bytes_per_token: 8192,
    }
}

fn reserve(hw: &HardwareGraph) -> BTreeMap<String, u64> {
    hw.devices
        .iter()
        .map(|d| {
            let r = match d.kind {
                DeviceKind::Gpu => 600 << 20,
                DeviceKind::Cpu => 2 << 30,
                DeviceKind::Disk => 0,
            };
            (d.id.clone(), r)
        })
        .collect()
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// ADR 0092 section 7: a builder plan is streamed when its RAM tier does not cover the experts
/// outside VRAM. (The golden prices with `tier: None`, so no plan here moves disk bytes.)
fn streamed(plan: &Plan, layout: &ModelLayout) -> bool {
    matches!(
        recognize(plan, layout),
        Ok((Placement::Builder { vram_slots, ram_slots }, _))
            if ram_slots < layout.n_expert.saturating_sub(vram_slots)
    )
}

/// One line per plan and prediction of every fixture: the non-streamed lines and the streamed
/// ones, each in the order of the whole.
fn snapshot() -> (String, String) {
    let mut streamed_out = String::new();
    let mut out = String::new();
    let models = [model("dense", 12, 0), model("moe", 12, 8)];
    for (name, toml) in GRAPHS {
        let hw = HardwareGraph::from_toml(toml).unwrap();
        for m in &models {
            for phase in [3u8, 5] {
                let hints = Hints::default();
                let res = search(&SearchInput {
                    hw: &hw,
                    model: m,
                    hints: &hints,
                    phase: Phase(phase),
                    workload: (512, 128, 1),
                    quality: &NullQuality::new(m),
                    reserve: reserve(&hw),
                    routing: None,
                    linux: true,
                });
                for (i, s) in res.scored.iter().enumerate() {
                    let plan = s.plan.to_toml().unwrap();
                    writeln!(
                        out,
                        "search {name} {} p{phase} #{i} plan={:016x} tok_s={:016x} ttft={:016x}",
                        m.name,
                        fnv(&plan),
                        s.tok_s.to_bits(),
                        s.ttft_s.to_bits()
                    )
                    .unwrap();
                }
            }
        }
        let has = |id: &str| hw.device(id).is_some();
        if !(has("gpu0") && has("cpu0")) {
            continue;
        }
        for (lname, layout) in [("dense", fixtures::dense()), ("moe", fixtures::moe())] {
            for engine in [ServeEngine::Llama, ServeEngine::Builder] {
                let inp = ServeInput {
                    budget: Budget {
                        gpu_bytes: hw.device("gpu0").unwrap().capacity_bytes,
                        ..budget_of(&hw, 8, None, None)
                    },
                    hw: hw.clone(),
                    layout: layout.clone(),
                    ctx: 4096,
                    workload: Workload {
                        prompt: 512,
                        r#gen: 128,
                    },
                    engine,
                    applies: Applies::all(),
                    tier: None,
                };
                for (i, plan) in candidates(&inp).iter().enumerate() {
                    let p = predict_plan(&hw, &layout, plan, &inp.workload);
                    let (pp, tg, req) = match p {
                        Ok(p) => (p.pp_tok_s, p.tg_tok_s, p.request_s),
                        Err(_) => (f64::NAN, f64::NAN, f64::NAN),
                    };
                    let dst = if streamed(plan, &layout) {
                        &mut streamed_out
                    } else {
                        &mut out
                    };
                    writeln!(
                        dst,
                        "serve {name} {lname} {engine:?} #{i} plan={:016x} pp={:016x} tg={:016x} req={:016x}",
                        fnv(&plan.to_toml().unwrap()),
                        pp.to_bits(),
                        tg.to_bits(),
                        req.to_bits()
                    )
                    .unwrap();
                }
            }
        }
    }
    (out, streamed_out)
}

fn check(now: &str, golden: &str, update: bool) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(golden);
    if update {
        std::fs::write(&path, now).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\r\n", "\n");
    for (i, (a, b)) in want.lines().zip(now.lines()).enumerate() {
        assert_eq!(a, b, "{golden}: line {i} differs");
    }
    assert_eq!(want.lines().count(), now.lines().count(), "{golden}");
}

#[test]
fn single_node_plans_and_predictions_are_unchanged() {
    let (now, streamed) = snapshot();
    let update = std::env::var_os("TETRAEDER_UPDATE_GOLDEN").is_some();
    assert!(now.lines().count() > 50, "golden file is nearly empty");
    assert!(
        streamed.lines().count() > 10,
        "no streamed plan in the golden"
    );
    check(&now, GOLDEN, update);
    check(&streamed, GOLDEN_STREAMED, update);
}
