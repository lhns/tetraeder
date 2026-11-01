//! A synthetic random-weight `qwen3moe` GGUF of any shape, written in Rust so tests need no
//! Python. Closer to a real MoE than the committed tiny fixture: top-8 routing, more experts and
//! dims that are multiples of 256, so `quantize` can produce K-quant mixes such as Q4_K_M.
//! Weights are F32 (the quantizer's input); keys and tensor layout as `xtask/tiny_moe.py`.

use std::io::Write;
use std::path::{Path, PathBuf};
use tetraeder_ggml_sys::llama::{Ftype, QuantizeParams, quantize};

#[derive(Debug, Clone, Copy)]
pub struct MoeSpec {
    pub layers: usize,
    /// Multiple of 256 for K-quants.
    pub embd: usize,
    pub heads: usize,
    pub heads_kv: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub used: usize,
    /// Multiple of 256 for K-quants (the inner dimension of the down projection).
    pub ff: usize,
    pub seed: u64,
}

impl MoeSpec {
    /// Shaped like Qwen3-30B-A3B where it matters for the expert path (top-8, K-quant dims),
    /// small enough for a CPU test.
    pub const REALISH: MoeSpec = MoeSpec {
        layers: 4,
        embd: 256,
        heads: 4,
        heads_kv: 2,
        head_dim: 64,
        experts: 32,
        used: 8,
        ff: 256,
        seed: 20261101,
    };
}

const N_CTX: u32 = 1024;

/// Byte-level vocab of 256 tokens plus a merge and three control tokens, like the tiny fixture.
fn vocab() -> Vec<String> {
    let mut bs: Vec<u32> = (33..127).chain(161..173).chain(174..256).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut v: Vec<String> = cs
        .into_iter()
        .map(|c| char::from_u32(c).unwrap().to_string())
        .collect();
    v.extend(["th", "<|im_start|>", "<|im_end|>", "<|endoftext|>"].map(String::from));
    v
}

/// splitmix64 and Box-Muller: deterministic normals without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

enum Kv {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
    I32s(Vec<i32>),
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

impl Kv {
    fn put(&self, out: &mut Vec<u8>) {
        match self {
            Kv::U32(x) => {
                out.extend(4u32.to_le_bytes());
                out.extend(x.to_le_bytes());
            }
            Kv::F32(x) => {
                out.extend(6u32.to_le_bytes());
                out.extend(x.to_le_bytes());
            }
            Kv::Bool(x) => {
                out.extend(7u32.to_le_bytes());
                out.push(u8::from(*x));
            }
            Kv::Str(s) => {
                out.extend(8u32.to_le_bytes());
                put_str(out, s);
            }
            Kv::Strs(v) => {
                out.extend(9u32.to_le_bytes());
                out.extend(8u32.to_le_bytes());
                out.extend((v.len() as u64).to_le_bytes());
                for s in v {
                    put_str(out, s);
                }
            }
            Kv::I32s(v) => {
                out.extend(9u32.to_le_bytes());
                out.extend(5u32.to_le_bytes());
                out.extend((v.len() as u64).to_le_bytes());
                for x in v {
                    out.extend(x.to_le_bytes());
                }
            }
        }
    }
}

/// Writes the F32 model to `path`.
pub fn write_moe(path: &Path, s: &MoeSpec) {
    let mut rng = Rng(s.seed);
    let tokens = vocab();
    let n_vocab = tokens.len();
    let mut types = vec![1i32; 257];
    types.extend([3, 3, 3]); // control
    let eos = tokens.iter().position(|t| t == "<|im_end|>").unwrap() as u32;
    let pad = tokens.iter().position(|t| t == "<|endoftext|>").unwrap() as u32;
    let a = |k: &str| format!("qwen3moe.{k}");
    let kvs: Vec<(String, Kv)> = vec![
        ("general.architecture".into(), Kv::Str("qwen3moe".into())),
        ("general.name".into(), Kv::Str("synth-qwen3moe".into())),
        (a("block_count"), Kv::U32(s.layers as u32)),
        (a("context_length"), Kv::U32(N_CTX)),
        (a("embedding_length"), Kv::U32(s.embd as u32)),
        (a("feed_forward_length"), Kv::U32((s.ff * s.used) as u32)),
        (a("attention.head_count"), Kv::U32(s.heads as u32)),
        (a("attention.head_count_kv"), Kv::U32(s.heads_kv as u32)),
        (a("attention.key_length"), Kv::U32(s.head_dim as u32)),
        (a("attention.value_length"), Kv::U32(s.head_dim as u32)),
        (a("rope.freq_base"), Kv::F32(10000.0)),
        (a("attention.layer_norm_rms_epsilon"), Kv::F32(1e-6)),
        (a("expert_count"), Kv::U32(s.experts as u32)),
        (a("expert_used_count"), Kv::U32(s.used as u32)),
        (a("expert_feed_forward_length"), Kv::U32(s.ff as u32)),
        ("general.file_type".into(), Kv::U32(0)), // all F32
        ("tokenizer.ggml.model".into(), Kv::Str("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Kv::Str("qwen2".into())),
        ("tokenizer.ggml.tokens".into(), Kv::Strs(tokens)),
        ("tokenizer.ggml.token_type".into(), Kv::I32s(types)),
        ("tokenizer.ggml.merges".into(), Kv::Strs(vec!["t h".into()])),
        ("tokenizer.ggml.eos_token_id".into(), Kv::U32(eos)),
        ("tokenizer.ggml.padding_token_id".into(), Kv::U32(pad)),
        ("tokenizer.ggml.add_bos_token".into(), Kv::Bool(false)),
    ];

    // (name, ggml ne, std; std 0 means a norm around 1)
    let (e, q, kv) = (s.embd, s.heads * s.head_dim, s.heads_kv * s.head_dim);
    let mut tensors: Vec<(String, Vec<usize>, f32)> = vec![
        ("token_embd.weight".into(), vec![e, n_vocab], 0.5),
        ("output_norm.weight".into(), vec![e], 0.0),
        ("output.weight".into(), vec![e, n_vocab], 0.5),
    ];
    // Scaled so activations stay near unit size at any width.
    let w = |fan_in: usize| 2.0 / (fan_in as f32).sqrt();
    for il in 0..s.layers {
        let p = |n: &str| format!("blk.{il}.{n}.weight");
        tensors.extend([
            (p("attn_norm"), vec![e], 0.0),
            (p("attn_q"), vec![e, q], w(e)),
            (p("attn_k"), vec![e, kv], w(e)),
            (p("attn_v"), vec![e, kv], w(e)),
            (p("attn_output"), vec![q, e], w(q)),
            (p("attn_q_norm"), vec![s.head_dim], 0.0),
            (p("attn_k_norm"), vec![s.head_dim], 0.0),
            (p("ffn_norm"), vec![e], 0.0),
            (
                p("ffn_gate_inp"),
                vec![e, s.experts],
                4.0 / (e as f32).sqrt(),
            ),
            (p("ffn_gate_exps"), vec![e, s.ff, s.experts], w(e)),
            (p("ffn_up_exps"), vec![e, s.ff, s.experts], w(e)),
            (p("ffn_down_exps"), vec![s.ff, e, s.experts], w(s.ff)),
        ]);
    }

    const ALIGN: usize = 32;
    let mut head = Vec::new();
    head.extend(b"GGUF");
    head.extend(3u32.to_le_bytes());
    head.extend((tensors.len() as u64).to_le_bytes());
    head.extend((kvs.len() as u64).to_le_bytes());
    for (k, v) in &kvs {
        put_str(&mut head, k);
        v.put(&mut head);
    }
    let mut offset = 0usize;
    for (name, ne, _) in &tensors {
        put_str(&mut head, name);
        head.extend((ne.len() as u32).to_le_bytes());
        for &d in ne {
            head.extend((d as u64).to_le_bytes());
        }
        head.extend(0u32.to_le_bytes()); // F32
        head.extend((offset as u64).to_le_bytes());
        offset += (ne.iter().product::<usize>() * 4).next_multiple_of(ALIGN);
    }
    head.resize(head.len().next_multiple_of(ALIGN), 0);

    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    f.write_all(&head).unwrap();
    for (_, ne, std) in &tensors {
        let n: usize = ne.iter().product();
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            let x = if *std == 0.0 {
                1.0 + 0.1 * rng.normal()
            } else {
                std * rng.normal()
            };
            bytes.extend(x.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(ALIGN), 0);
        f.write_all(&bytes).unwrap();
    }
    f.flush().unwrap();
}

/// The model of `spec` quantized to `ftype`, in the temp dir, written once per process and name.
pub fn quantized_moe(spec: &MoeSpec, ftype: Ftype, name: &str) -> PathBuf {
    let id = std::process::id();
    quantized_moe_at(
        spec,
        ftype,
        &std::env::temp_dir().join(format!("tetraeder-synth-{name}-{id}.gguf")),
    )
}

/// The model of `spec` quantized to `ftype` at `out`, written only if `out` does not exist (the
/// F32 intermediate goes next to it, 4 bytes per weight, and is removed after).
pub fn quantized_moe_at(spec: &MoeSpec, ftype: Ftype, out: &Path) -> PathBuf {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !out.is_file() {
        let src = out.with_extension("f32.tmp");
        let part = out.with_extension("part");
        write_moe(&src, spec);
        let mut qp = QuantizeParams::new(ftype);
        qp.nthread = 2;
        quantize(&src, &part, &qp).unwrap();
        std::fs::remove_file(&src).ok();
        std::fs::rename(&part, out).unwrap();
    }
    out.to_path_buf()
}
