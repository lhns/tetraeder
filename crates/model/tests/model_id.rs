//! Model identity of a split GGUF (ADR 0091, Case S blockers): the 235B is five shards while the
//! daemon names a model `<sha256>.gguf`. A split fixture (the tiny 4-layer model cut in three by
//! the library's writer, as llama.cpp's `gguf-split` does) is identified, installed under its id
//! and found again by `resolve`, and the model loads from the path `resolve` gives.
//!
//! Hardware label: none.

mod common;

use common::split::split_gguf_at;
use std::path::{Path, PathBuf};
use tetraeder_model::Gguf;
use tetraeder_model::model_id::{self, identify, install, resolve};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3-l4-f16.gguf")
}

fn tmp(tag: &str) -> PathBuf {
    let d =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The fixture cut into three shards under `dir`; the first shard's path.
fn split(dir: &Path) -> PathBuf {
    let n = Gguf::open(fixture()).unwrap().tensors.len();
    let shards = split_gguf_at(&fixture(), &dir.join("tiny"), &[n / 3, 2 * n / 3]);
    assert_eq!(shards.len(), 3);
    shards[0].clone()
}

// hardware: none
#[test]
fn a_split_model_has_one_id_that_both_nodes_resolve_and_load() {
    let src = tmp("mid-src");
    let first = split(&src);
    let id = identify(&first).unwrap();
    assert!(model_id::is_id(&id.id));
    assert_eq!(id.files.len(), 3);
    // The single file has the sha256 of its bytes, which is not the split model's id.
    let single = identify(&fixture()).unwrap();
    assert_eq!(single.id, model_id::file_sha256(&fixture()).unwrap().0);
    assert_ne!(single.id, id.id);
    // Identifying again, from the same shards, gives the same id.
    assert_eq!(identify(&first).unwrap().id, id.id);

    // Two nodes: each installs the shards under the id, and resolves the same one.
    let (a, b) = (tmp("mid-a"), tmp("mid-b"));
    let (pa, pb) = (install(&id, &a).unwrap(), install(&id, &b).unwrap());
    for (models, p) in [(&a, &pa), (&b, &pb)] {
        let found = resolve(models, &id.id).unwrap();
        assert_eq!(&found.path, p);
        assert_eq!(found.shards, 3);
        assert_eq!(
            found.bytes,
            id.manifest
                .as_ref()
                .unwrap()
                .shards
                .iter()
                .map(|s| s.bytes)
                .sum::<u64>()
        );
        // The engine opens every shard from the first.
        assert_eq!(Gguf::open(&found.path).unwrap().n_shards(), 3);
        assert_eq!(model_id::id_of_path(&found.path), Some(id.id.clone()));
    }
    // A single file installs and resolves as before.
    let one = tmp("mid-one");
    let p = install(&single, &one).unwrap();
    let found = resolve(&one, &single.id).unwrap();
    assert_eq!((found.path, found.shards), (p, 1));
}

// hardware: none
#[test]
fn a_node_with_a_damaged_or_incomplete_split_model_refuses_it() {
    let src = tmp("mid-bad-src");
    let id = identify(&split(&src)).unwrap();
    let count = 3;

    // A missing shard.
    let m = tmp("mid-missing");
    install(&id, &m).unwrap();
    std::fs::remove_file(
        m.join(&id.id)
            .join(model_id::shard_file_name(&id.id, 2, count)),
    )
    .unwrap();
    let e = resolve(&m, &id.id).unwrap_err().to_string();
    assert!(e.contains("shard 2 of 3"), "{e}");

    // A shard of the wrong size (a partial copy).
    let m = tmp("mid-short");
    install(&id, &m).unwrap();
    let s3 = m
        .join(&id.id)
        .join(model_id::shard_file_name(&id.id, 3, count));
    let bytes = std::fs::read(&s3).unwrap();
    // The install links the shard: replace it with a short copy, not write through the link.
    std::fs::remove_file(&s3).unwrap();
    std::fs::write(&s3, &bytes[..bytes.len() - 1]).unwrap();
    let e = resolve(&m, &id.id).unwrap_err().to_string();
    assert!(e.contains("shard 3 of 3") && e.contains("bytes"), "{e}");

    // A manifest that is not the model's (another model's shards under this id).
    let m = tmp("mid-swapped");
    install(&id, &m).unwrap();
    let other = {
        let mut o = id.manifest.clone().unwrap();
        o.shards[0].sha256 = "0".repeat(64);
        o
    };
    std::fs::write(
        m.join(&id.id).join(model_id::MANIFEST_FILE),
        other.to_text(),
    )
    .unwrap();
    let e = resolve(&m, &id.id).unwrap_err().to_string();
    assert!(e.contains("manifest hashes to"), "{e}");

    // No such model, and an id that is not hex.
    let e = resolve(&tmp("mid-none"), &id.id).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
    assert!(resolve(&m, "../x").is_err());
}
