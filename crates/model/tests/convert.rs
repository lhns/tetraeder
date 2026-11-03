//! Load-time format conversion with its disk cache (ADR 0047): byte parity with libllama's
//! quantizer per format, the cache, the builder on converted weights against libllama, refused
//! conversions and the GGUF export.
//!
//! Hardware labels (AGENTS rule 9): all tests are `cpu`.

mod common;

use common::synth::{MoeSpec, write_moe};
use common::{TOL_CPU, check_against, libllama_reference_at};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use tetraeder_ggml_sys::ggml::Type;
use tetraeder_ggml_sys::llama::{Ftype, GgmlType, QuantizeParams, gguf_tensor_types, quantize};
use tetraeder_model::convert::{self, ConvertOptions, format_of};
use tetraeder_model::gguf_write::{shard_name, write_gguf, write_split};
use tetraeder_model::{BackendChoice, Gguf, LoadOptions, Qwen3Moe};
use tetraeder_plan::Format;

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };

/// Small enough for a CPU test, dims that are multiples of 256 for the K-quants.
const SPEC: MoeSpec = MoeSpec {
    layers: 2,
    experts: 16,
    used: 4,
    ..MoeSpec::REALISH
};

fn work_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = std::env::temp_dir().join(format!("tetraeder-convert-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    })
    .clone()
}

/// The F32 synthetic model, written once per process.
fn f32_model() -> PathBuf {
    static ONCE: OnceLock<PathBuf> = OnceLock::new();
    ONCE.get_or_init(|| {
        let p = work_dir().join("f32.gguf");
        write_moe(&p, &SPEC);
        p
    })
    .clone()
}

fn opts(cache: &str) -> ConvertOptions {
    ConvertOptions {
        cache: Some(work_dir().join(cache)),
        threads: 2,
        telemetry: None,
    }
}

fn tensor_bytes(g: &Gguf, name: &str) -> Vec<u8> {
    let t = g.tensor(name).unwrap();
    let mut f = std::fs::File::open(&g.paths[t.shard]).unwrap();
    f.seek(SeekFrom::Start(t.offset)).unwrap();
    let mut v = vec![0u8; t.nbytes as usize];
    f.read_exact(&mut v).unwrap();
    v
}

/// The format map that reproduces a llama-quantized file's tensor types.
fn types_of(path: &Path) -> BTreeMap<String, Format> {
    gguf_tensor_types(path)
        .unwrap()
        .into_iter()
        .map(|(n, ty)| (n, format_of(Type::from_name(&ty).unwrap()).unwrap()))
        .collect()
}

fn llama_quantize(name: &str, ftype: Ftype, overrides: &[(&str, GgmlType)]) -> PathBuf {
    let out = work_dir().join(format!("{name}.gguf"));
    let mut qp = QuantizeParams::new(ftype);
    qp.nthread = 2;
    qp.tensor_types = overrides.iter().map(|(p, t)| (p.to_string(), *t)).collect();
    quantize(f32_model(), &out, &qp).unwrap();
    out
}

/// Converts the F32 model with `map` and requires every tensor of `reference` to have the same
/// type and bytes. Returns the conversion time.
fn assert_same_as(reference: &Path, map: &BTreeMap<String, Format>, cache: &str) -> f64 {
    let mut g = Gguf::open(f32_model()).unwrap();
    let t0 = Instant::now();
    convert::convert(&mut g, map, &opts(cache)).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    let want = Gguf::open(reference).unwrap();
    assert_eq!(g.tensors.len(), want.tensors.len());
    for t in &want.tensors {
        assert_eq!(g.tensor(&t.name).unwrap().ty, t.ty, "type of {}", t.name);
        assert!(
            tensor_bytes(&g, &t.name) == tensor_bytes(&want, &t.name),
            "bytes of {} differ from libllama's",
            t.name
        );
    }
    secs
}

/// hardware: cpu. One format forced for every weight: our bytes equal libllama's quantizer's.
#[test]
fn cpu_bytes_match_libllama_per_format() {
    let f32_mb = std::fs::metadata(f32_model()).unwrap().len() as f64 / 1e6;
    let formats = [
        ("Q8_0", GgmlType::Q8_0),
        ("Q6_K", GgmlType::Q6K),
        ("Q5_K", GgmlType::Q5K),
        ("Q4_K", GgmlType::Q4K),
        ("Q5_0", GgmlType::Q5_0),
        ("Q4_0", GgmlType::Q4_0),
        ("IQ4_XS", GgmlType::Iq4Xs),
        ("IQ4_NL", GgmlType::Iq4Nl),
        ("Q3_K", GgmlType::Q3K),
        ("Q2_K", GgmlType::Q2K),
        ("F16", GgmlType::F16),
        ("BF16", GgmlType::Bf16),
    ];
    for (name, ty) in formats {
        let reference = llama_quantize(&format!("ref-{name}"), Ftype::Q8_0, &[(r"\.weight", ty)]);
        let map = types_of(&reference);
        let secs = assert_same_as(&reference, &map, &format!("cache-{name}"));
        eprintln!("{name}: {:.0} MB/s of f32 input", f32_mb / secs);
        std::fs::remove_file(&reference).ok();
    }
}

/// hardware: cpu. A real preset mix (Q4_K_M picks Q6_K for some tensors, Q4_K for others).
#[test]
fn cpu_bytes_match_libllama_q4_k_m() {
    let reference = llama_quantize("ref-q4km", Ftype::Q4KM, &[]);
    let map = types_of(&reference);
    let kinds: std::collections::BTreeSet<_> = map.values().map(|f| format!("{f:?}")).collect();
    assert!(kinds.len() >= 3, "a mix of formats: {kinds:?}");
    assert_same_as(&reference, &map, "cache-q4km");
    // The thread count does not change the bytes.
    let mut g = Gguf::open(f32_model()).unwrap();
    let one = ConvertOptions {
        threads: 1,
        ..opts("cache-q4km-1t")
    };
    convert::convert(&mut g, &map, &one).unwrap();
    let want = Gguf::open(&reference).unwrap();
    for t in &want.tensors {
        assert!(
            tensor_bytes(&g, &t.name) == tensor_bytes(&want, &t.name),
            "{}",
            t.name
        );
    }
}

fn experts(f: Format) -> BTreeMap<String, Format> {
    (0..SPEC.layers)
        .flat_map(|il| {
            ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                .map(|w| (format!("blk.{il}.{w}.weight"), f))
        })
        .collect()
}

/// hardware: cpu. The second load converts nothing; one other format converts one tensor; a
/// deleted cache file is converted again.
#[test]
fn cpu_cache_hits_and_misses() {
    let bins = |d: &Path| -> std::collections::BTreeSet<PathBuf> {
        std::fs::read_dir(d)
            .map(|r| r.map(|e| e.unwrap().path()).collect())
            .unwrap_or_default()
    };
    let cache = work_dir().join("cache-hits");
    let mut map = experts(Format::Q8_0);
    let n = map.len() as u64;
    let run = |map: &BTreeMap<String, Format>| {
        let mut g = Gguf::open(f32_model()).unwrap();
        convert::convert(&mut g, map, &opts("cache-hits")).unwrap()
    };
    let s = run(&map);
    assert_eq!((s.converted, s.cache_hits), (n, 0));
    assert!(s.bytes_written > 0);
    let s = run(&map);
    assert_eq!((s.converted, s.cache_hits, s.bytes_written), (0, n, 0));
    map.insert("blk.0.ffn_up_exps.weight".into(), Format::Q5_K);
    let before = bins(&cache);
    let s = run(&map);
    assert_eq!((s.converted, s.cache_hits), (1, n - 1));
    let mut new_files: Vec<_> = bins(&cache).difference(&before).cloned().collect();
    assert_eq!(new_files.len(), 1);
    let victim = new_files.pop().unwrap();
    std::fs::remove_file(&victim).unwrap();
    let s = run(&map);
    assert_eq!((s.converted, s.cache_hits), (1, n - 1));
    // No temporary files stay behind.
    assert!(
        std::fs::read_dir(&cache).unwrap().all(|e| e
            .unwrap()
            .path()
            .extension()
            .is_some_and(|e| e == "bin"))
    );
}

/// hardware: cpu. The builder on weights converted at load gives libllama's logits on the same
/// types, resident and with expert slots (the cache files are read as shards).
#[test]
fn cpu_builder_on_converted_weights_matches_libllama() {
    let mut map = experts(Format::Q8_0);
    for il in 0..SPEC.layers {
        for w in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            map.insert(format!("blk.{il}.{w}.weight"), Format::Q8_0);
        }
    }
    map.insert("token_embd.weight".into(), Format::F16);
    map.insert("output.weight".into(), Format::F16);
    let reference = llama_quantize(
        "ref-builder",
        Ftype::Q8_0,
        &[
            (r"^token_embd\.weight", GgmlType::F16),
            (r"^output\.weight", GgmlType::F16),
        ],
    );
    let mut want = types_of(&reference);
    want.retain(|_, f| *f != Format::F32);
    assert_eq!(want, map, "libllama picked other types than the map");
    let r = libllama_reference_at(&reference, &common::PROMPT, common::N_GREEDY);
    for (slots, ram, first) in [
        (None, None, true),
        (Some(8), None, false),
        (Some(2), Some(3), false),
    ] {
        let o = LoadOptions {
            expert_slots: slots,
            ram_slots: ram,
            formats: map.clone(),
            convert_cache: Some(work_dir().join("cache-builder")),
            ..LoadOptions::new(CPU)
        };
        let mut m = Qwen3Moe::load(f32_model(), &o).unwrap();
        let s = m.convert_stats();
        if first {
            assert_eq!(s.converted, map.len() as u64);
        } else {
            assert_eq!((s.converted, s.cache_hits), (0, map.len() as u64));
        }
        let worst = check_against(&mut m, &r, TOL_CPU);
        eprintln!("converted slots {slots:?} ram {ram:?}: max |delta| {worst:e}");
    }
}

fn refusal(name: &str, format: Format, model: &Path, cache: &str) -> String {
    let mut g = Gguf::open(model).unwrap();
    // A valid entry beside it: nothing may be converted when any entry is refused.
    let mut map = BTreeMap::from([("blk.0.ffn_up_exps.weight".to_string(), Format::Q8_0)]);
    map.insert(name.to_string(), format);
    let e = convert::convert(&mut g, &map, &opts(cache))
        .unwrap_err()
        .to_string();
    assert!(
        !work_dir().join(cache).exists(),
        "`{name}`: the cache was touched before the refusal"
    );
    e
}

/// hardware: cpu. Upward conversions, importance-matrix formats, unknown tensors, the router,
/// 1-D tensors and rows that do not fill a block are refused with the tensor named.
#[test]
fn cpu_refused_conversions() {
    let q4km = llama_quantize("ref-refuse", Ftype::Q4KM, &[]);
    let e = refusal("blk.0.attn_q.weight", Format::Q8_0, &q4km, "c-up");
    assert!(
        e.contains("blk.0.attn_q.weight") && e.contains("more precise"),
        "{e}"
    );

    let e = refusal(
        "blk.0.attn_q.weight",
        Format::IQ2_XXS,
        &f32_model(),
        "c-imat",
    );
    assert!(
        e.contains("blk.0.attn_q.weight") && e.contains("importance matrix"),
        "{e}"
    );

    let e = refusal("blk.9.nope.weight", Format::Q8_0, &f32_model(), "c-unknown");
    assert!(e.contains("blk.9.nope.weight"), "{e}");

    let e = refusal(
        "blk.0.ffn_gate_inp.weight",
        Format::Q8_0,
        &f32_model(),
        "c-router",
    );
    assert!(e.contains("router"), "{e}");

    let e = refusal(
        "blk.0.attn_norm.weight",
        Format::Q8_0,
        &f32_model(),
        "c-norm",
    );
    assert!(e.contains("1-D"), "{e}");

    // attn_output rows are 2 heads * 64 = 128 wide: a Q8_0 block fits, a K-quant block does not.
    let narrow = MoeSpec {
        layers: 1,
        heads: 2,
        heads_kv: 1,
        experts: 8,
        used: 4,
        ..SPEC
    };
    let path = work_dir().join("narrow.gguf");
    write_moe(&path, &narrow);
    let e = refusal("blk.0.attn_output.weight", Format::Q4_K, &path, "c-block");
    assert!(
        e.contains("blk.0.attn_output.weight") && e.contains("block size"),
        "{e}"
    );

    // The load path reports the same error.
    let o = LoadOptions {
        formats: BTreeMap::from([("blk.0.ffn_gate_inp.weight".to_string(), Format::Q8_0)]),
        convert_cache: Some(work_dir().join("c-load")),
        ..LoadOptions::new(CPU)
    };
    let e = Qwen3Moe::load(f32_model(), &o).unwrap_err().to_string();
    assert!(e.contains("router"), "{e}");
}

/// hardware: cpu. Export as a single file and split by two groups: the tensors keep their bytes
/// and types, `Gguf::open` and libllama read the shards.
#[test]
fn cpu_export_single_and_split() {
    let map = experts(Format::Q8_0);
    let mut g = Gguf::open(f32_model()).unwrap();
    convert::convert(&mut g, &map, &opts("cache-export")).unwrap();
    assert!(g.n_shards() > 1, "cache files are shards of the view");

    let single = write_gguf(&g, &work_dir().join("export-single.gguf")).unwrap();
    let one = Gguf::open(&single[0]).unwrap();
    assert_eq!(one.n_shards(), 1);

    let groups: Vec<Vec<String>> = (0..2)
        .map(|il| {
            ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                .map(|w| format!("blk.{il}.{w}.weight"))
                .to_vec()
        })
        .collect();
    let prefix = work_dir().join("export-split");
    let paths = write_split(&g, &prefix, &groups).unwrap();
    assert_eq!(
        paths,
        (0..3)
            .map(|i| shard_name(&prefix, i, 3))
            .collect::<Vec<_>>()
    );
    let many = Gguf::open(&paths[0]).unwrap();
    assert_eq!(many.n_shards(), 3);
    assert_eq!(many.tensors.len(), g.tensors.len());
    for t in &g.tensors {
        for other in [&one, &many] {
            let o = other.tensor(&t.name).unwrap();
            assert_eq!((o.ty, o.ne), (t.ty, t.ne), "{}", t.name);
            assert!(
                tensor_bytes(other, &t.name) == tensor_bytes(&g, &t.name),
                "{}",
                t.name
            );
        }
    }
    // Group i is shard i + 1; everything else is in shard 1.
    for (i, names) in groups.iter().enumerate() {
        for n in names {
            assert_eq!(many.tensor(n).unwrap().shard, i + 1);
        }
    }
    assert_eq!(many.tensor("token_embd.weight").unwrap().shard, 0);
    assert_eq!(many.str("general.architecture").unwrap(), "qwen3moe");

    // libllama reads the shards and the single file alike, and so does the builder.
    let r = libllama_reference_at(&paths[0], &common::PROMPT, common::N_GREEDY);
    let r1 = libllama_reference_at(&single[0], &common::PROMPT, common::N_GREEDY);
    assert_eq!(r.greedy, r1.greedy);
    let mut m = Qwen3Moe::load(&paths[0], &LoadOptions::new(CPU)).unwrap();
    check_against(&mut m, &r, TOL_CPU);

    // A tensor in two groups or missing from the model is an error.
    let dup = vec![groups[0].clone(), groups[0].clone()];
    assert!(write_split(&g, &prefix, &dup).is_err());
    assert!(write_split(&g, &prefix, &[vec!["nope".to_string()]]).is_err());
}
