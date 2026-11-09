//! `tetraeder info` (ADR 0052). Hardware: cpu. The DL assertions need the default build;
//! `TETRAEDER_GGML_NATIVE=ON` builds are static and skip them.

use std::process::{Command, Output};

fn info(variant: Option<&str>) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_tetraeder"));
    c.args(["info", "--json"]);
    c.env_remove("TETRAEDER_GGML_CPU_VARIANT");
    if let Some(v) = variant {
        c.env("TETRAEDER_GGML_CPU_VARIANT", v);
    }
    c.output().expect("run tetraeder")
}

fn dl_x86() -> bool {
    if !cfg!(target_arch = "x86_64") {
        eprintln!("skipped: x86-64 only");
        return false;
    }
    if std::env::var("TETRAEDER_GGML_NATIVE").is_ok_and(|v| v == "ON") {
        eprintln!("skipped: static native build");
        return false;
    }
    true
}

#[test]
fn info_reports_dl_dispatch() {
    let out = info(None);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    if dl_x86() {
        assert_eq!(v["cpu"]["dispatch"], "dl", "{v}");
        assert_eq!(v["cpu"]["extra"], serde_json::json!([]), "{v}");
        assert_eq!(v["cpu"]["missing"], serde_json::json!([]), "{v}");
    }
    assert!(v["backends"].as_array().is_some_and(|b| !b.is_empty()));
}

#[test]
fn forced_x64_variant_is_the_sse_tier() {
    if !dl_x86() {
        return;
    }
    let out = info(Some("x64"));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert_eq!(v["cpu"]["tier"], "sse", "{v}");
    assert_eq!(v["cpu"]["forced"], "x64", "{v}");
}

#[test]
fn missing_variant_is_an_error_naming_the_file() {
    if !dl_x86() {
        return;
    }
    let out = info(Some("nonexistent"));
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("ggml-cpu-nonexistent"), "{err}");
}
