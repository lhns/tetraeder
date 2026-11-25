//! `tetraeder cluster model-split` (the Case S dry run, ADR 0091): the eight-layer tiny MoE cut
//! into shards the way llama.cpp's `gguf-split` does, which `cluster model-id --install` then
//! gives one id on every node and `resolve` finds again. The dry run needs sharded model files
//! on a CI runner and on a box that has no llama.cpp tools.
//!
//! Hardware label: none.

use std::path::{Path, PathBuf};
use std::process::Command;
use tetraeder_model::Gguf;
use tetraeder_model::model_id::{is_id, resolve};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-l8-f16.gguf")
}

fn tmp(tag: &str) -> PathBuf {
    let d =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tetraeder() -> Command {
    Command::new(env!("CARGO_BIN_EXE_tetraeder"))
}

// hardware: none
#[test]
fn a_model_is_cut_into_shards_that_model_id_installs_under_one_id() {
    let dir = tmp("model-split");
    let prefix = dir.join("tiny");
    let o = tetraeder()
        .args(["cluster", "model-split"])
        .arg(fixture())
        .arg(&prefix)
        .args(["--shards", "3"])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    // The shard names are llama.cpp's: the driver finds the first one by name.
    let first = dir.join("tiny-00001-of-00003.gguf");
    for k in 1..=3 {
        assert!(dir.join(format!("tiny-{k:05}-of-00003.gguf")).is_file());
    }
    let g = Gguf::open(&first).unwrap();
    assert_eq!(g.n_shards(), 3);
    // Every tensor of the source is in some shard, and a layer's three expert tensors are not all in one
    // (the cut is before the second expert tensor of a layer, so a layer spans two shards).
    assert_eq!(
        g.tensors.len(),
        Gguf::open(fixture()).unwrap().tensors.len()
    );

    // Two nodes install the same shards and get the same id.
    let mut ids = Vec::new();
    for n in ["a", "b"] {
        let models = dir.join(format!("models-{n}"));
        let o = tetraeder()
            .args(["cluster", "model-id"])
            .arg(&first)
            .arg("--install")
            .arg(&models)
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let id = String::from_utf8_lossy(&o.stdout).trim().to_string();
        assert!(is_id(&id), "{id}");
        assert_eq!(resolve(&models, &id).unwrap().shards, 3);
        ids.push(id);
    }
    assert_eq!(ids[0], ids[1]);

    // A request for no shards, or more than the model can be cut into, is refused.
    let o = tetraeder()
        .args(["cluster", "model-split"])
        .arg(fixture())
        .arg(dir.join("none"))
        .args(["--shards", "0"])
        .output()
        .unwrap();
    assert!(!o.status.success());
}
