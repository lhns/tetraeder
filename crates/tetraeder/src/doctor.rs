//! `tetraeder doctor`: the detector report of one observation file (ADR 0041), as text or JSON.
//! Report only; it changes nothing.

use anyhow::{Context, Result};
use clap::Args;
use serde::Serialize;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use tetraeder_analysis::detect::{Report, Verdict, detect};
use tetraeder_analysis::observation::Observation;
use tetraeder_telemetry::{Metrics, Telemetry, TelemetryConfig};

#[derive(Args)]
pub struct DoctorArgs {
    /// Observation JSON (see crates/analysis/fixtures/observation_example.json).
    #[arg(long)]
    observation: PathBuf,
    /// Print the report as JSON.
    #[arg(long)]
    json: bool,
}

/// Enum names as in the JSON output (`gpu_throttle`).
pub fn snake<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

pub fn read_observation(path: &Path) -> Result<Observation> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn run(a: &DoctorArgs) -> Result<()> {
    let obs = read_observation(&a.observation)?;
    // Metrics only when an OTLP endpoint is configured; otherwise there is nowhere to send them.
    let cfg = TelemetryConfig::from_env();
    let telemetry = if cfg.otlp_endpoint.is_some() {
        Some(Telemetry::new(&cfg)?)
    } else {
        None
    };
    let out = output(&obs, a.json, telemetry.as_ref().map(Telemetry::metrics))?;
    print!("{out}");
    if let Some(t) = &telemetry {
        t.shutdown();
    }
    Ok(())
}

/// The report text, and the metrics of the report when a handle is given.
pub fn output(obs: &Observation, json: bool, metrics: Option<&Metrics>) -> Result<String> {
    let report = detect(obs);
    if let Some(m) = metrics {
        record(m, &report);
    }
    if json {
        Ok(serde_json::to_string_pretty(&report)? + "\n")
    } else {
        Ok(text(obs, &report))
    }
}

/// `bottleneck_share` per limit, `bottleneck_fault` per fault.
pub fn record(m: &Metrics, report: &Report) {
    for l in &report.limits {
        m.bottleneck_share(&l.resource, l.share);
    }
    for f in &report.faults {
        m.bottleneck_fault(&snake(&f.class), &f.resource);
    }
}

fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

fn text(obs: &Observation, r: &Report) -> String {
    let mut o = String::new();
    let _ = writeln!(
        o,
        "observation {}: batch {}, {} windows",
        obs.id,
        obs.batch,
        obs.windows.len()
    );
    let _ = writeln!(o, "regime: {}", snake(&r.regime));
    match &r.verdict {
        Verdict::Balanced => {
            let _ = writeln!(o, "verdict: balanced (no resource clearly limits)");
        }
        Verdict::Limited(res) => {
            let _ = writeln!(o, "verdict: limited by {res}");
        }
    }
    if r.faults.is_empty() {
        o.push_str("faults: none\n");
    } else {
        o.push_str("faults (ranked by cost share of the step):\n");
        for (i, f) in r.faults.iter().enumerate() {
            let _ = writeln!(
                o,
                "  {}. {} on {}: {}",
                i + 1,
                snake(&f.class),
                f.resource,
                pct(f.cost_share)
            );
            for e in &f.evidence {
                let _ = writeln!(o, "       {e}");
            }
        }
    }
    o.push_str("limits (share of the step, 95% interval):\n");
    for l in &r.limits {
        let _ = writeln!(
            o,
            "  {:<16} {:>6}  [{}, {}]",
            l.resource,
            pct(l.share),
            pct(l.ci_low),
            pct(l.ci_high)
        );
    }
    for n in &r.notes {
        let _ = writeln!(o, "note: {n}");
    }
    o
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use tetraeder_analysis::detect::FaultClass;
    use tetraeder_analysis::simfault::{SimSpec, simulate};
    use tetraeder_profiler::HardwareGraph;

    const BOX: &str = include_str!("../../analysis/fixtures/rtx3090_box.toml");
    const EXAMPLE: &str = include_str!("../../analysis/fixtures/observation_example.json");

    /// The fixture box with a GPU throttle injected.
    pub fn faulty() -> Observation {
        let hw = HardwareGraph::from_toml(BOX).unwrap();
        simulate(&hw, &SimSpec::new(1, &[FaultClass::GpuThrottle])).unwrap()
    }

    pub fn example() -> Observation {
        serde_json::from_str(EXAMPLE).unwrap()
    }

    // hardware: none
    #[test]
    fn example_text_has_regime_verdict_and_limits() {
        let out = output(&example(), false, None).unwrap();
        assert!(out.contains("regime: "), "{out}");
        assert!(out.contains("verdict: "), "{out}");
        assert!(out.contains("limits (share of the step"), "{out}");
    }

    // hardware: none
    #[test]
    fn example_json_is_the_report() {
        let out = output(&example(), true, None).unwrap();
        let back: Report = serde_json::from_str(&out).unwrap();
        let want = detect(&example());
        // floats may differ in the last digit after the text round trip
        assert_eq!((back.regime, &back.verdict), (want.regime, &want.verdict));
        assert_eq!(back.faults.len(), want.faults.len());
        let names = |r: &Report| {
            r.limits
                .iter()
                .map(|l| l.resource.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&back), names(&want));
    }

    // hardware: none
    #[test]
    fn faulty_observation_names_the_fault() {
        let obs = faulty();
        let text = output(&obs, false, None).unwrap();
        assert!(text.contains("gpu_throttle"), "{text}");
        assert!(text.contains("faults (ranked"), "{text}");
        let json = output(&obs, true, None).unwrap();
        let r: Report = serde_json::from_str(&json).unwrap();
        assert!(r.faults.iter().any(|f| f.class == FaultClass::GpuThrottle));
    }

    // hardware: none
    #[test]
    fn metrics_are_recorded_without_an_exporter() {
        let t = Telemetry::new(&TelemetryConfig::default()).unwrap();
        let out = output(&faulty(), false, Some(t.metrics())).unwrap();
        assert!(out.contains("gpu_throttle"));
    }

    // hardware: none
    #[test]
    fn read_observation_reports_the_path() {
        let e = read_observation(Path::new("/no/such/observation.json")).unwrap_err();
        assert!(format!("{e:#}").contains("observation.json"));
    }
}
