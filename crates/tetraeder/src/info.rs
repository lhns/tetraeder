//! `tetraeder info`: the CPU backend this binary loaded (ADR 0052), as text or JSON.

use anyhow::{Result, anyhow};
use clap::Args;
use tetraeder_ggml_sys::{cpu, llama};
use tetraeder_telemetry::Metrics;

#[derive(Args)]
pub struct InfoArgs {
    /// Print JSON.
    #[arg(long)]
    json: bool,
}

pub fn run(a: &InfoArgs) -> Result<()> {
    let info = cpu::backend_info().map_err(|e| anyhow!("{e}"))?;
    let backends: Vec<String> = llama::devices().into_iter().map(|d| d.name).collect();
    let system = llama::system_info();
    if a.json {
        let v = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "cpu": info,
            "backends": backends,
            "system_info": system,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        println!("dispatch: {}", info.dispatch.as_str());
        if let Some(d) = &info.module_dir {
            println!("module dir: {}", d.display());
        }
        if let Some(f) = &info.forced {
            println!("forced variant: {f}");
        }
        println!("tier: {}", info.tier);
        println!("isa: {}", info.isa.join(","));
        println!("host isa: {}", info.host_isa.join(","));
        println!("missing: {}", info.missing.join(","));
        println!("extra: {}", info.extra.join(","));
        println!("backends: {}", backends.join(", "));
        println!("{system}");
    }
    Ok(())
}

/// Records the CPU backend gauges once at engine start. A load error is not recorded here: the
/// engine constructors report it.
pub fn record_cpu_backend(m: &Metrics) {
    if let Ok(i) = cpu::backend_info() {
        m.cpu_backend(i.tier, &i.isa, i.dispatch.as_str(), &i.missing);
    }
}
