//! Plain-text view for `tetraeder top`: host and GPU metrics plus step statistics.

use crate::host::{DeviceKind, DeviceSample};
use crate::ring::StepStats;
use std::fmt::Write;

fn bar(frac: f64, width: usize) -> String {
    let filled = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!("[{}{}]", "#".repeat(filled), ".".repeat(width - filled))
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Render one frame.
pub fn render(devices: &[DeviceSample], steps: &StepStats) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "tetraeder top");
    let _ = writeln!(s, "\nDevices");
    for d in devices {
        let kind = match d.kind {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Gpu => "gpu",
        };
        let mem_frac = if d.mem_total > 0 {
            d.mem_used as f64 / d.mem_total as f64
        } else {
            0.0
        };
        let _ = write!(
            s,
            "  {:<5} {kind} busy {} {:>5.1}%  mem {} {:>6.1}/{:.1} GiB",
            d.id,
            bar(d.busy, 20),
            d.busy * 100.0,
            bar(mem_frac, 20),
            gib(d.mem_used),
            gib(d.mem_total),
        );
        if let Some(w) = d.power_w {
            let _ = write!(s, "  {w:.0} W");
        }
        if let Some(t) = d.temp_c {
            let _ = write!(s, "  {t:.0} C");
        }
        let _ = writeln!(s, "  {}", d.name);
    }
    let _ = writeln!(s, "\nSteps (recent window)");
    if steps.count == 0 {
        let _ = writeln!(s, "  no steps recorded in this process");
    } else {
        let _ = writeln!(
            s,
            "  n {}  mean {:.2} ms  p50 {:.2} ms  p95 {:.2} ms  max {:.2} ms  tokens {}",
            steps.count, steps.mean_ms, steps.p50_ms, steps.p95_ms, steps.max_ms, steps.tokens
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_devices_and_empty_steps() {
        let d = DeviceSample {
            id: "cpu0".into(),
            name: "Test CPU".into(),
            kind: DeviceKind::Cpu,
            busy: 0.25,
            mem_used: 1 << 30,
            mem_total: 4 << 30,
            power_w: None,
            temp_c: None,
        };
        let out = render(&[d], &StepStats::default());
        assert!(out.contains("25.0%") && out.contains("1.0/4.0 GiB"));
        assert!(out.contains("no steps recorded"));
    }
}
