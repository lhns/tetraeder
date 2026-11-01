//! Split GGUF files (ADR 0037): the header reader merges the shards, the builder gives bitwise
//! the same logits as on the single file, libllama reads the fixture's shards like the original.
//! The shards are written by `common::split`, which follows llama.cpp's `gguf-split`.
//!
//! Hardware labels (AGENTS rule 9): all tests are `cpu`.

mod common;

use common::split::{expert_cuts, split_gguf, split_gguf_at};
use common::synth::{MoeSpec, quantized_moe};
use common::{PROMPT, argmax, libllama_reference_at};
use std::path::{Path, PathBuf};
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::hparams::EXPERT_TENSORS;
use tetraeder_model::{BackendChoice, ExpertStats, Gguf, LoadOptions, MissPolicy, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_DECODE: usize = 4;

fn single() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km")
}

/// Shards of the fixture under a prefix of their own, written once per process and name.
fn shards(name: &str, n: usize) -> Vec<PathBuf> {
    let prefix =
        std::env::temp_dir().join(format!("tetraeder-split-{name}-{n}-{}", std::process::id()));
    split_gguf(&single(), &prefix, n)
}

/// hardware: cpu
#[test]
fn cpu_split_header_equals_single_file() {
    let one = Gguf::open(single()).unwrap();
    let paths = shards("header", 3);
    let many = Gguf::open(&paths[0]).unwrap();
    assert_eq!(many.n_shards(), 3);
    assert_eq!(many.paths, paths);
    assert_eq!(many.tensors.len(), one.tensors.len());
    let bytes: Vec<Vec<u8>> = paths.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let whole = std::fs::read(single()).unwrap();
    for t in &many.tensors {
        let s = one.tensor(&t.name).unwrap();
        assert_eq!((t.ty, t.ne, t.nbytes), (s.ty, s.ne, s.nbytes), "{}", t.name);
        let (a, b) = (t.offset as usize, s.offset as usize);
        assert_eq!(
            bytes[t.shard][a..a + t.nbytes as usize],
            whole[b..b + s.nbytes as usize],
            "{}",
            t.name
        );
    }
    // Every expert slab, read from its shard.
    let mut spread = 0;
    for il in 0..4u32 {
        let shards: Vec<usize> = EXPERT_TENSORS
            .iter()
            .map(|w| many.tensor(&format!("blk.{il}.{w}.weight")).unwrap().shard)
            .collect();
        eprintln!("layer {il}: gate, up, down in shards {shards:?}");
        spread += usize::from(shards.iter().any(|&s| s != shards[0]));
        for w in EXPERT_TENSORS {
            let name = format!("blk.{il}.{w}.weight");
            let (t, s) = (many.tensor(&name).unwrap(), one.tensor(&name).unwrap());
            for e in 0..t.ne[2] {
                let (x, y) = (t.slab(e).unwrap(), s.slab(e).unwrap());
                assert_eq!(x.len, y.len);
                assert_eq!(
                    bytes[x.shard][x.offset as usize..(x.offset + x.len) as usize],
                    whole[y.offset as usize..(y.offset + y.len) as usize],
                    "{name} expert {e}"
                );
            }
        }
    }
    assert!(
        spread > 0,
        "a layer has its expert tensors in different shards"
    );
    assert!(
        many.tensors.iter().map(|t| t.shard).max().unwrap() == 2,
        "all three shards hold tensors"
    );
}

/// hardware: cpu
#[test]
fn cpu_split_open_errors() {
    let paths = shards("errors", 3);
    let e = Gguf::open(&paths[1]).unwrap_err().to_string();
    assert!(e.contains("first shard"), "{e}");
    // A shard missing on disk.
    let prefix =
        std::env::temp_dir().join(format!("tetraeder-split-missing-{}", std::process::id()));
    let paths = split_gguf(&single(), &prefix, 3);
    std::fs::remove_file(&paths[2]).unwrap();
    assert!(Gguf::open(&paths[0]).is_err());
    // A first shard under a name that is not llama.cpp's.
    let renamed = paths[0].with_file_name("renamed.gguf");
    std::fs::copy(&paths[0], &renamed).unwrap();
    assert!(
        Gguf::open(&renamed)
            .unwrap_err()
            .to_string()
            .contains("named")
    );
}

/// hardware: cpu. `--no-tensor-first-split`: the first shard holds the metadata only.
#[test]
fn cpu_split_first_shard_without_tensors() {
    let one = Gguf::open(single()).unwrap();
    let mut cuts = vec![0];
    cuts.extend(expert_cuts(&one, 2));
    let prefix =
        std::env::temp_dir().join(format!("tetraeder-split-notensor-{}", std::process::id()));
    let paths = split_gguf_at(&single(), &prefix, &cuts);
    let many = Gguf::open(&paths[0]).unwrap();
    assert_eq!(many.n_shards(), 3);
    assert!(many.tensors.iter().all(|t| t.shard > 0));
    assert_eq!(many.tensors.len(), one.tensors.len());
}

/// hardware: cpu. The reference reader follows the shards of the fixture and reads the same
/// model: this validates the fixture against libllama.
#[test]
fn cpu_libllama_reads_the_split_fixture() {
    let paths = shards("llama", 3);
    let want = libllama_reference_at(&single(), &PROMPT, 3);
    let got = libllama_reference_at(&paths[0], &PROMPT, 3);
    assert_eq!(want.greedy, got.greedy);
    assert_eq!(want.prompt_logits.len(), got.prompt_logits.len());
    assert!(
        want.prompt_logits
            .iter()
            .zip(&got.prompt_logits)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "libllama logits differ between the single file and its shards"
    );
}

/// Prefill of a prompt, then decode steps; returns every eval's logits and the stats.
fn run(path: &Path, o: &LoadOptions) -> (Vec<Vec<f32>>, Option<ExpertStats>) {
    let mut m = Qwen3Moe::load(path, o).unwrap();
    let n_vocab = m.n_vocab();
    let prompt: Vec<i32> = (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect();
    let prefill = m.eval(&prompt).unwrap();
    let mut next = argmax(&prefill[(N_PROMPT - 1) * n_vocab..]) as i32;
    let mut logits = vec![prefill];
    for _ in 0..N_DECODE {
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
        logits.push(l);
    }
    (logits, m.expert_stats())
}

fn check_same(name: &str, o: &LoadOptions) {
    let (want, ws) = run(&single(), o);
    let paths = shards(name, 3);
    let (got, gs) = run(&paths[0], o);
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{name}: logits of eval {i} differ"
        );
    }
    if let (Some(ws), Some(gs)) = (ws, gs) {
        eprintln!("{name}: single {:?}, split {:?}", ws.io, gs.io);
        assert!(ws.io.experts_read > 0, "{name}: experts were read");
        assert_eq!(ws.io.experts_read, gs.io.experts_read);
        assert_eq!(ws.io.bytes_read, gs.io.bytes_read);
        assert!(gs.io.batches >= ws.io.batches, "{name}: a batch per shard");
    }
}

/// hardware: cpu
#[test]
fn cpu_split_builder_resident_is_exact() {
    check_same("resident", &LoadOptions::new(CPU));
}

/// hardware: cpu
#[test]
fn cpu_split_builder_upload_is_exact() {
    let o = LoadOptions {
        expert_slots: Some(4),
        ..LoadOptions::new(CPU)
    };
    check_same("upload", &o);
}

/// hardware: cpu. RAM misses and the hot set: prefill fills the RAM slots from the shards, the
/// decode steps mix compute-slot hits and CPU groups.
#[test]
fn cpu_split_builder_cpu_policy_is_exact() {
    let o = LoadOptions {
        expert_slots: Some(4),
        ram_slots: Some(8),
        miss: MissPolicy::Cpu,
        ..LoadOptions::new(CPU)
    };
    check_same("cpu-policy", &o);
}
