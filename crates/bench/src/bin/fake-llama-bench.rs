//! Test double for `llama-bench`: prints the same JSON shape with throughput taken from its flags.
//! Flags: `--fake-pp N`, `--fake-tg N`, `--fake-jitter F` (relative, random from the clock),
//! `--fake-fail`, `--fake-sleep-ms N`. `FAKE_LOG` names a file that gets one line (the variant tag) per call, to check interleaving.
//! All other llama-bench flags are accepted and ignored except `-p`/`-n`, which switch the rows on.

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    if args.iter().any(|a| a == "--fake-fail") {
        eprintln!("fake-llama-bench: simulated failure");
        std::process::exit(3);
    }
    if let Some(ms) = flag("--fake-sleep-ms").and_then(|v| v.parse::<u64>().ok()) {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    let num = |name: &str, d: f64| flag(name).and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let tag = flag("--fake-tag").unwrap_or_default();
    if let Ok(path) = std::env::var("FAKE_LOG") {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("open FAKE_LOG");
        writeln!(f, "{tag}").expect("write FAKE_LOG");
    }
    // Uniform in +-jitter around the nominal rate, from the clock.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let jitter = num("--fake-jitter", 0.0) * ((f64::from(nanos / 1000 % 2001) - 1000.0) / 1000.0);
    let scale = 1.0 + jitter;
    let p = num("-p", 0.0) as u64;
    let n = num("-n", 0.0) as u64;
    let mut rows = Vec::new();
    if p > 0 {
        rows.push(format!(
            r#"{{"build_commit":"fakecommit","n_prompt":{p},"n_gen":0,"avg_ts":{}}}"#,
            num("--fake-pp", 100.0) * scale
        ));
    }
    if n > 0 {
        rows.push(format!(
            r#"{{"build_commit":"fakecommit","n_prompt":0,"n_gen":{n},"avg_ts":{}}}"#,
            num("--fake-tg", 10.0) * scale
        ));
    }
    println!("[{}]", rows.join(","));
}
