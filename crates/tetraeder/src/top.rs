//! `tetraeder top`: live view without any backend.

use clap::Args;
use std::time::Duration;
use tetraeder_telemetry::{Telemetry, TelemetryConfig, default_sources, top};

#[derive(Args)]
pub struct TopArgs {
    /// Print one frame and exit.
    #[arg(long)]
    once: bool,
    /// Refresh interval in milliseconds.
    #[arg(long, default_value_t = 1000)]
    interval_ms: u64,
}

pub fn run(args: &TopArgs) {
    // No exporter: top reads the in-process state only.
    let telemetry =
        Telemetry::new(&TelemetryConfig::default()).expect("telemetry without exporter");
    let ring = telemetry.ring();
    let mut sources = default_sources();
    // CPU usage needs two readings a short time apart.
    std::thread::sleep(Duration::from_millis(250));
    loop {
        let devices: Vec<_> = sources
            .iter_mut()
            .flat_map(|s| telemetry.sample_host(s.as_mut()))
            .collect();
        let frame = top::render(&devices, &ring.stats());
        if args.once {
            print!("{frame}");
            return;
        }
        // Clear screen and home the cursor.
        print!("\x1b[2J\x1b[H{frame}");
        std::thread::sleep(Duration::from_millis(args.interval_ms));
    }
}
