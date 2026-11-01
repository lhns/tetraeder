//! Slot tiers from byte budgets and the memory report (ADR 0037) on the K-quant fixture of
//! `tests/multipass.rs`: Q4_K_M, where some layers have a wider `ffn_down` and so bigger experts.
//!
//! Hardware labels (AGENTS rule 9): all tests are `cpu`.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe};
use std::path::PathBuf;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{
    BackendChoice, LoadOptions, MemoryUse, MissPolicy, ModelError, ModelFile, Qwen3Moe,
};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_EXPERT: u64 = 32;

fn path() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km")
}

fn per_layer() -> Vec<u64> {
    ModelFile::open(path())
        .unwrap()
        .expert_bytes_per_layer()
        .unwrap()
}

/// Bytes of one expert over all layers: what one slot per layer costs.
fn unit() -> u64 {
    per_layer().iter().sum()
}

fn cpu_policy(ram_slots: Option<usize>, ram_slot_bytes: Option<u64>) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(4),
        ram_slots,
        ram_slot_bytes,
        miss: MissPolicy::Cpu,
        ..LoadOptions::new(CPU)
    }
}

fn find<'a>(m: &'a [MemoryUse], tier: &str, kind: &str) -> &'a MemoryUse {
    m.iter()
        .find(|u| u.tier == tier && u.kind == kind)
        .unwrap_or_else(|| panic!("no {tier} {kind} entry in {m:?}"))
}

fn ram_tier(m: &Qwen3Moe) -> MemoryUse {
    m.memory()
        .into_iter()
        .find(|u| u.device == "host" && u.kind == "slots")
        .expect("a RAM tier entry")
}

fn invalid(r: Result<Qwen3Moe, ModelError>) -> String {
    match r {
        Err(ModelError::Invalid(m)) => m,
        Err(e) => panic!("unexpected error {e}"),
        Ok(_) => panic!("expected an error"),
    }
}

/// Prefill and a few decode steps; the logits of every eval.
fn run(m: &mut Qwen3Moe) -> Vec<Vec<f32>> {
    let n_vocab = m.n_vocab();
    let prompt: Vec<i32> = (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect();
    let prefill = m.eval(&prompt).unwrap();
    let mut next = argmax(&prefill[(N_PROMPT - 1) * n_vocab..]) as i32;
    let mut logits = vec![prefill];
    for _ in 0..3 {
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
        logits.push(l);
    }
    logits
}

/// hardware: cpu
#[test]
fn cpu_expert_bytes_differ_across_layers() {
    let sizes = per_layer();
    eprintln!("expert bytes per layer: {sizes:?}");
    assert_eq!(sizes.len(), 4);
    assert!(sizes.iter().min() != sizes.iter().max());
    let f = ModelFile::open(path()).unwrap();
    assert_eq!(f.slots_for_bytes(0).unwrap(), 0);
    assert_eq!(f.slots_for_bytes(3 * unit()).unwrap(), 3);
    assert_eq!(f.slots_for_bytes(3 * unit() - 1).unwrap(), 2);
    assert_eq!(f.slots_for_bytes(u64::MAX).unwrap(), N_EXPERT);
}

/// hardware: cpu
#[test]
fn cpu_ram_slot_bytes_buy_whole_slots() {
    let k = 3;
    let m = Qwen3Moe::load(path(), &cpu_policy(None, Some(k * unit()))).unwrap();
    let ram = ram_tier(&m);
    assert_eq!((ram.tier, ram.kind), ("ram", "slots"));
    assert!(ram.capacity <= k * unit(), "{ram:?}");
    assert_eq!(ram.capacity, k * unit());
    assert_eq!(ram.used, 0, "nothing is resident before the first step");

    let m = Qwen3Moe::load(path(), &cpu_policy(None, Some(k * unit() - 1))).unwrap();
    assert_eq!(ram_tier(&m).capacity, (k - 1) * unit());
}

/// hardware: cpu
#[test]
fn cpu_budget_errors() {
    let u = unit();
    let msg = invalid(Qwen3Moe::load(path(), &cpu_policy(None, Some(u - 1))));
    assert!(msg.contains(&u.to_string()), "{msg}");
    let o = LoadOptions {
        expert_slot_bytes: Some(1),
        ..LoadOptions::new(CPU)
    };
    assert!(invalid(Qwen3Moe::load(path(), &o)).contains(&u.to_string()));
    let msg = invalid(Qwen3Moe::load(path(), &cpu_policy(Some(8), Some(8 * u))));
    assert!(msg.contains("not both"), "{msg}");
    let o = LoadOptions {
        expert_slots: Some(4),
        expert_slot_bytes: Some(4 * u),
        ..LoadOptions::new(CPU)
    };
    assert!(invalid(Qwen3Moe::load(path(), &o)).contains("not both"));
}

/// hardware: cpu. A compute budget that buys every expert means all resident, like no limit.
#[test]
fn cpu_expert_slot_bytes_for_all_experts_is_resident() {
    for bytes in [N_EXPERT * unit(), 100 * N_EXPERT * unit()] {
        let o = LoadOptions {
            expert_slot_bytes: Some(bytes),
            ..LoadOptions::new(CPU)
        };
        let m = Qwen3Moe::load(path(), &o).unwrap();
        assert!(m.expert_stats().is_none());
        assert!(m.memory().iter().all(|u| u.kind != "slots"));
    }
    // One slot less than all experts streams.
    let o = LoadOptions {
        expert_slot_bytes: Some((N_EXPERT - 1) * unit()),
        ..LoadOptions::new(CPU)
    };
    let m = Qwen3Moe::load(path(), &o).unwrap();
    assert!(m.expert_stats().is_some());
}

/// hardware: cpu. The report lists the buffers; after a run the RAM slots in use are some of the
/// tier, and the logits equal those of the same tier sized by count.
#[test]
fn cpu_memory_report_and_budget_equals_count() {
    let mut by_count = Qwen3Moe::load(path(), &cpu_policy(Some(8), None)).unwrap();
    let mut by_bytes = Qwen3Moe::load(path(), &cpu_policy(None, Some(8 * unit()))).unwrap();
    let (a, b) = (run(&mut by_count), run(&mut by_bytes));
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(
            x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
            "eval {i}: logits differ"
        );
    }
    let report = by_bytes.memory();
    eprintln!("{report:#?}");
    let ram = ram_tier(&by_bytes);
    assert!(ram.used > 0 && ram.used <= ram.capacity, "{ram:?}");
    let slots = report
        .iter()
        .find(|u| u.device != "host" && u.kind == "slots")
        .unwrap();
    assert!(slots.used > 0 && slots.used <= slots.capacity, "{slots:?}");
    // The compute slot tensors hold 4 slots per layer.
    assert_eq!(slots.capacity, 4 * unit());
    let weights = find(&report, "ram", "weights");
    let kv = find(&report, "ram", "kv");
    assert!(weights.capacity > 0 && kv.capacity > 0);
    assert!(
        report
            .iter()
            .any(|u| u.kind == "workspace" && u.capacity > 0)
    );
    // On the CPU backend every primary entry is host memory.
    assert!(report.iter().all(|u| u.tier == "ram"));
}
