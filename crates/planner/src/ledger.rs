//! The calibration ledger (ADR 0056, section 3): CI re-predicts every committed planner fixture
//! with the calibration of its own session and fails when the model drifts.
//!
//! A fixture directory `bench/results/<date>/<dir>/` with `candidates.json` and `layout.json` is
//! one session (anchors and candidates from one engine build on one box). Its key is
//! `(engine_commit, host_id, cpu_variant)`, taken from a `provenance.json` beside the candidates
//! (legacy override), else from the harness file's `env` (`git_commit`, `host_id`,
//! `cpu_backend.tier`), else from the `provenance` block of the builder's `exit_regime` files.
//! A result with no engine commit is `uncalibrated`: listed, never failed.
//!
//! [`evaluate`] fits each session on its anchors (the scorer already does) and records the
//! relative request-time error of every candidate. [`check`] compares that with the committed
//! snapshot `bench/ledger.json`; [`write`] is the ratchet that produces the next snapshot.

use crate::campaign::{CampaignVariant, CandidatesFile};
use crate::layout::ModelLayout;
use crate::score::{self, request_s};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tetraeder_costmodel::MODEL_VERSION;
use tetraeder_profiler::HardwareGraph;

/// Relative tolerance of "the same prediction" (ADR 0056 section 3). Linux CI, Windows CI and
/// the machine that wrote the snapshot agree on every committed prediction within it.
pub const PRED_TOL: f64 = 1e-9;
/// Relative tolerance of "the same fit" for `calib.json`. The fit is a numerical minimizer
/// whose weakly constrained parameters differ more between Linux and Windows than the
/// predictions made from them, so it has a tolerance of its own. Measured on CI against a
/// snapshot written on a third machine (2026-11-09): the largest relative difference of a fitted
/// rate was 5.5e-8 (Linux, dense) and 2.6e-8 (Windows, dense), while every prediction agreed
/// within `PRED_TOL`; ten times the largest.
pub const CALIB_TOL: f64 = 5.5e-7;
/// Median candidate error allowed per session, MoE: ADR 0040's 9.2% (the anchor band).
pub const MOE_BAND: f64 = 0.092;
/// Median candidate error allowed per session, dense: ADR 0040's 5.8%.
pub const DENSE_BAND: f64 = 0.058;
/// Smallest error increase the ratchet refuses.
const RATCHET_FLOOR: f64 = 0.02;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SessionKey {
    pub engine_commit: String,
    pub host_id: String,
    pub cpu_variant: Option<String>,
}

/// The fit of one session, exported as `calib.json` beside its results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calib {
    pub cost_model_version: u32,
    pub session: String,
    pub key: SessionKey,
    pub anchors: Vec<String>,
    pub band: f64,
    pub hw: HardwareGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub name: String,
    pub key: SessionKey,
    pub class: String,
    pub anchors: Vec<String>,
    /// The fit's anchor band (ADR 0040), for the record.
    pub fit_band: f64,
    /// Builder fixtures: where the RAM tier came from (`replay`, `hitmodel`, or `hitmodel
    /// (fidelity gate failed: ...)`), so a failed fidelity gate shows in the ledger (ADR 0056).
    #[serde(default)]
    pub tier_source: String,
    /// The fixture directory; not part of the snapshot.
    #[serde(skip)]
    pub dir: PathBuf,
    /// The fit recomputed now; not part of the snapshot.
    #[serde(skip)]
    pub recomputed: Option<Calib>,
    /// The `calib.json` on disk, if any; not part of the snapshot.
    #[serde(skip)]
    pub stored: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Fixture directory relative to the results root, `/` separated: `20261103/planner-moe`.
    pub path: String,
    pub id: String,
    pub session: String,
    /// `moe` or `dense`.
    pub class: String,
    /// `checked`, `uncalibrated`, or `excluded: <reason>`.
    pub status: String,
    pub predicted_s: Option<f64>,
    pub measured_s: Option<f64>,
    /// `|predicted - measured| / measured`.
    pub err: Option<f64>,
    /// `(max - min) / median` of the variant's repeats (0 with fewer than two); the ratchet's slack.
    #[serde(default)]
    pub spread: f64,
    pub cost_model_version: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Accepted {
    pub id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ledger {
    pub cost_model_version: u32,
    pub sessions: Vec<SessionRecord>,
    pub entries: Vec<LedgerEntry>,
    /// Errors the ratchet let rise, with the reason given on the command line.
    #[serde(default)]
    pub accepted: Vec<Accepted>,
}

impl PartialEq for SessionRecord {
    fn eq(&self, o: &Self) -> bool {
        self.name == o.name
            && self.key == o.key
            && self.class == o.class
            && self.anchors == o.anchors
    }
}

impl Ledger {
    pub fn empty() -> Self {
        Self {
            cost_model_version: MODEL_VERSION,
            sessions: Vec::new(),
            entries: Vec::new(),
            accepted: Vec::new(),
        }
    }
}

fn read_json(p: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
}

fn text(v: &Value) -> Option<String> {
    v.as_str()
        .filter(|s| !s.is_empty() && *s != "unknown")
        .map(str::to_string)
}

/// The `tier` of a harness `cpu_backend` block (ADR 0052), or the string itself.
fn variant_of(v: &Value) -> Option<String> {
    text(v).or_else(|| text(&v["tier"]))
}

/// Session name and key of a fixture directory; `None` without an engine commit.
fn session_of(
    dir: &Path,
    harness: &[Value],
    builder: &BTreeMap<String, Vec<Value>>,
) -> Option<(String, SessionKey)> {
    let key = |c: &Value, h: &Value, v: &Value| {
        Some(SessionKey {
            engine_commit: text(c)?,
            host_id: text(h)?,
            cpu_variant: variant_of(v),
        })
    };
    if let Some(p) = read_json(&dir.join("provenance.json")) {
        let k = key(&p["engine_commit"], &p["host_id"], &p["cpu_variant"])?;
        let name = text(&p["session"]).unwrap_or_else(|| {
            format!(
                "unnamed-{}",
                dir.file_name().unwrap_or_default().to_string_lossy()
            )
        });
        return Some((name, k));
    }
    if let Some(env) = harness.iter().map(|f| &f["env"]).find(|e| e.is_object()) {
        let k = key(&env["git_commit"], &env["host_id"], &env["cpu_backend"])?;
        let name = text(&env["session"]).unwrap_or_else(|| {
            let s = env["started_utc"].as_str().unwrap_or("");
            let digits: String = s.chars().filter(char::is_ascii_digit).collect();
            let (date, time) = digits.split_at(digits.len().min(8));
            format!("{date}-{}-{}", k.host_id, &time[..time.len().min(6)])
        });
        return Some((name, k));
    }
    let p = builder
        .values()
        .flatten()
        .map(|f| &f["provenance"])
        .find(|p| p.is_object())?;
    let k = key(&p["engine_commit"], &p["host_id"], &p["cpu_variant"])?;
    let name = text(&p["session"])?;
    Some((name, k))
}

fn spread(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let med = if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0
    };
    (s[s.len() - 1] - s[0]) / med
}

/// Request seconds of every repeat of every variant, from the harness samples or the builder files.
fn repeat_times(
    cands: &CandidatesFile,
    harness: &[Value],
    builder: &BTreeMap<String, Vec<Value>>,
) -> BTreeMap<String, Vec<f64>> {
    let by_id: BTreeMap<&str, &CampaignVariant> = cands
        .candidates
        .iter()
        .map(|e| &e.variant)
        .chain(&cands.anchors)
        .map(|v| (v.id.as_str(), v))
        .collect();
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for pair in harness
        .iter()
        .flat_map(|r| r["pairs"].as_array().into_iter().flatten())
    {
        for s in pair["samples"].as_array().into_iter().flatten() {
            let Some(id) = s["variant"].as_str() else {
                continue;
            };
            let Some(v) = by_id.get(id) else { continue };
            if let Some(t) = request_s(v, s["pp_ts"].as_f64(), s["tg_ts"].as_f64()) {
                out.entry(id.to_string()).or_default().push(t);
            }
        }
    }
    for (id, files) in builder {
        let Some(v) = by_id.get(id.as_str()) else {
            continue;
        };
        for f in files {
            let last = f["runs"].as_array().and_then(|r| r.last());
            let pp = last.and_then(|r| r["prefill_tok_s"].as_f64());
            let tg = last.and_then(|r| r["decode_tok_s"].as_f64());
            if let Some(t) = request_s(v, pp, tg) {
                out.entry(id.clone()).or_default().push(t);
            }
        }
    }
    out
}

/// Fixture directories under `root/<date>/`, sorted.
fn fixture_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for day in std::fs::read_dir(root).into_iter().flatten().flatten() {
        for d in std::fs::read_dir(day.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let p = d.path();
            if p.join("candidates.json").is_file() && p.join("layout.json").is_file() {
                dirs.push(p);
            }
        }
    }
    dirs.sort();
    dirs
}

fn rel_path(root: &Path, dir: &Path) -> String {
    let rel = dir.strip_prefix(root).unwrap_or(dir);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Re-predicts every committed fixture with its own session's fit.
pub fn evaluate(results_root: &Path) -> Ledger {
    let mut ledger = Ledger::empty();
    for dir in fixture_dirs(results_root) {
        evaluate_dir(results_root, &dir, &mut ledger);
    }
    ledger
        .entries
        .sort_by(|a, b| (&a.path, &a.id).cmp(&(&b.path, &b.id)));
    ledger.sessions.sort_by(|a, b| a.name.cmp(&b.name));
    ledger
}

fn evaluate_dir(root: &Path, dir: &Path, ledger: &mut Ledger) {
    let path = rel_path(root, dir);
    let entry = |id: &str, session: &str, class: &str, status: String| LedgerEntry {
        path: path.clone(),
        id: id.to_string(),
        session: session.to_string(),
        class: class.to_string(),
        status,
        predicted_s: None,
        measured_s: None,
        err: None,
        spread: 0.0,
        cost_model_version: MODEL_VERSION,
    };
    let loaded = (|| -> Result<(ModelLayout, CandidatesFile), String> {
        let read = |n: &str| std::fs::read_to_string(dir.join(n)).map_err(|e| e.to_string());
        Ok((
            serde_json::from_str(&read("layout.json")?).map_err(|e| e.to_string())?,
            serde_json::from_str(&read("candidates.json")?).map_err(|e| e.to_string())?,
        ))
    })();
    let (layout, cands) = match loaded {
        Ok(x) => x,
        Err(e) => {
            ledger
                .entries
                .push(entry("*", "-", "-", format!("excluded: unreadable: {e}")));
            return;
        }
    };
    let class = if layout.n_expert > 0 { "moe" } else { "dense" };
    let harness = score::read_harness_files(dir).unwrap_or_default();
    let builder = score::read_builder_files(dir).unwrap_or_default();
    let session = session_of(dir, &harness, &builder);
    let scored = score::score_dir(dir);
    let (Some((name, key)), Ok(s)) = (session, &scored) else {
        // no engine commit, or no anchor measured: listed, not failed
        let status = match &scored {
            Err(e) if !e.contains("no anchor") && !e.contains("only") => {
                format!("excluded: {e}")
            }
            _ => "uncalibrated".to_string(),
        };
        for e in &cands.candidates {
            ledger
                .entries
                .push(entry(&e.candidate.id, "-", class, status.clone()));
        }
        return;
    };
    let times = repeat_times(&cands, &harness, &builder);
    for (e, c) in cands.candidates.iter().zip(&s.candidates) {
        let mut row = entry(&c.id, &name, class, "checked".into());
        match c.measured_s {
            None => row.status = "excluded: failed on the box".into(),
            Some(m) => {
                row.predicted_s = Some(c.predicted_s);
                row.measured_s = Some(m);
                row.err = Some((c.predicted_s - m).abs() / m);
                row.spread = times.get(&e.variant.id).map_or(0.0, |t| spread(t));
            }
        }
        ledger.entries.push(row);
    }
    let mut anchors: Vec<String> = s
        .anchor_residuals
        .iter()
        .filter_map(|(id, _)| id.rsplit_once(':').map(|x| x.0.to_string()))
        .collect();
    anchors.dedup();
    ledger.sessions.push(SessionRecord {
        name: name.clone(),
        key: key.clone(),
        class: class.into(),
        anchors: anchors.clone(),
        fit_band: s.band,
        tier_source: s.tier_source.clone(),
        dir: dir.to_path_buf(),
        recomputed: Some(Calib {
            cost_model_version: MODEL_VERSION,
            session: name,
            key,
            anchors,
            band: s.band,
            hw: s.calibrated.clone(),
        }),
        stored: read_json(&dir.join("calib.json")),
    });
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= PRED_TOL * a.abs().max(b.abs())
}

/// The largest relative difference between the numbers of two JSON values and where it is;
/// infinite for a difference in shape or in a non-number.
fn json_diff(a: &Value, b: &Value, path: &str) -> (f64, String) {
    let worse = |x: (f64, String), y: (f64, String)| if y.0 > x.0 { y } else { x };
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(x), Some(y)) => {
                let m = x.abs().max(y.abs());
                (if m == 0.0 { 0.0 } else { (x - y).abs() / m }, path.into())
            }
            _ => (if x == y { 0.0 } else { f64::INFINITY }, path.into()),
        },
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .enumerate()
            .map(|(i, (a, b))| json_diff(a, b, &format!("{path}[{i}]")))
            .fold((0.0, String::new()), worse),
        (Value::Object(x), Value::Object(y)) if x.len() == y.len() => {
            let mut acc = (0.0, String::new());
            for (k, v) in x {
                let d = match y.get(k) {
                    Some(w) => json_diff(v, w, &format!("{path}.{k}")),
                    None => (f64::INFINITY, format!("{path}.{k}")),
                };
                acc = worse(acc, d);
            }
            acc
        }
        _ if a == b => (0.0, String::new()),
        _ => (f64::INFINITY, path.into()),
    }
}

fn band_of(class: &str) -> f64 {
    if class == "moe" { MOE_BAND } else { DENSE_BAND }
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Median error of the `checked` entries of one session.
fn session_median(l: &Ledger, session: &str) -> Option<f64> {
    median(
        l.entries
            .iter()
            .filter(|e| e.session == session && e.status == "checked")
            .filter_map(|e| e.err)
            .collect(),
    )
}

/// Gates 1 to 4 of ADR 0056 section 3; an empty list passes.
pub fn check(current: &Ledger, snapshot: &Ledger) -> Vec<String> {
    let mut out = Vec::new();
    if snapshot.cost_model_version != current.cost_model_version {
        out.push(format!(
            "ledger.json is at cost model version {}, the code at {}: run `tetraeder-plan ledger --write`",
            snapshot.cost_model_version, current.cost_model_version
        ));
        return out;
    }
    let now: BTreeMap<(&str, &str), &LedgerEntry> = current
        .entries
        .iter()
        .map(|e| ((e.path.as_str(), e.id.as_str()), e))
        .collect();
    for old in &snapshot.entries {
        let Some(new) = now.get(&(old.path.as_str(), old.id.as_str())) else {
            out.push(format!(
                "{}/{}: in ledger.json, missing from the current results",
                old.path, old.id
            ));
            continue;
        };
        if new.status != old.status {
            out.push(format!(
                "{}/{}: status {:?} in ledger.json, {:?} now",
                old.path, old.id, old.status, new.status
            ));
        }
        let same = match (old.predicted_s, new.predicted_s) {
            (Some(a), Some(b)) => close(a, b),
            (a, b) => a == b,
        };
        if !same {
            out.push(format!(
                "{}/{}: prediction {:?} in ledger.json, {:?} now (same model version {})",
                old.path, old.id, old.predicted_s, new.predicted_s, current.cost_model_version
            ));
        }
    }
    for s in &current.sessions {
        if let Some(m) = session_median(current, &s.name) {
            let band = band_of(&s.class);
            if m > band {
                out.push(format!(
                    "session {}: median error {m:.3} exceeds the {} band {band}",
                    s.name, s.class
                ));
            }
        }
        if let Some(c) = &s.recomputed {
            let want = serde_json::to_value(c).unwrap_or(Value::Null);
            match &s.stored {
                None => out.push(format!(
                    "session {}: no calib.json in {}: run `tetraeder-plan ledger --write`",
                    s.name,
                    s.dir.display()
                )),
                Some(have) => {
                    let (d, at) = json_diff(have, &want, "");
                    if d > CALIB_TOL {
                        out.push(format!(
                            "session {}: calib.json in {} is not the current fit (differs by {d:.3e} at {at}, tolerance {CALIB_TOL:e}): run `tetraeder-plan ledger --write`",
                            s.name,
                            s.dir.display()
                        ));
                    }
                }
            }
        }
    }
    out
}

/// The ratchet: the next snapshot, or an error naming every entry whose error rose by more than
/// `max(0.02, spread)` over `previous` and is not in `accept_worse` (`id` or `path/id`, reason).
pub fn write(
    current: &Ledger,
    previous: Option<&Ledger>,
    accept_worse: &[(String, String)],
) -> Result<Ledger, String> {
    let mut out = current.clone();
    out.accepted = previous.map(|p| p.accepted.clone()).unwrap_or_default();
    let Some(prev) = previous else { return Ok(out) };
    let old: BTreeMap<(&str, &str), &LedgerEntry> = prev
        .entries
        .iter()
        .map(|e| ((e.path.as_str(), e.id.as_str()), e))
        .collect();
    let mut refused = Vec::new();
    for e in &current.entries {
        let (Some(err), Some(was)) = (
            e.err,
            old.get(&(e.path.as_str(), e.id.as_str()))
                .and_then(|o| o.err),
        ) else {
            continue;
        };
        let slack = RATCHET_FLOOR.max(e.spread);
        if err <= was + slack {
            continue;
        }
        let full = format!("{}/{}", e.path, e.id);
        match accept_worse.iter().find(|(id, _)| *id == e.id || *id == full) {
            Some((_, reason)) => {
                out.accepted.retain(|a| a.id != full);
                out.accepted.push(Accepted { id: full, reason: reason.clone() });
            }
            None => refused.push(format!(
                "{full}: error {was:.3} -> {err:.3} (more than {slack:.3}); pass --accept-worse {}=<reason>",
                e.id
            )),
        }
    }
    if refused.is_empty() {
        Ok(out)
    } else {
        Err(refused.join("\n"))
    }
}

/// The table `tetraeder-plan ledger` prints: one row per session.
pub fn table(l: &Ledger) -> String {
    let mut s = format!(
        "cost model version {}\n{:<44} {:<6} {:>7} {:>8}  worst\n",
        l.cost_model_version, "session", "class", "checked", "median"
    );
    for r in &l.sessions {
        let rows: Vec<&LedgerEntry> = l
            .entries
            .iter()
            .filter(|e| e.session == r.name && e.status == "checked")
            .collect();
        let worst = rows
            .iter()
            .max_by(|a, b| a.err.unwrap_or(0.0).total_cmp(&b.err.unwrap_or(0.0)));
        s += &format!(
            "{:<44} {:<6} {:>7} {:>8.3}  {}\n",
            r.name,
            r.class,
            rows.len(),
            session_median(l, &r.name).unwrap_or(f64::NAN),
            worst.map_or(String::new(), |w| format!(
                "{}/{} {:.3}",
                w.path,
                w.id,
                w.err.unwrap_or(0.0)
            ))
        );
    }
    let other = l.entries.iter().filter(|e| e.status != "checked").count();
    if other > 0 {
        s += &format!("{other} entries uncalibrated or excluded\n");
    }
    s
}

/// `bench/ledger.json` for a results root `bench/results`.
pub fn ledger_path(root: &Path) -> PathBuf {
    root.parent().unwrap_or(Path::new(".")).join("ledger.json")
}

/// `tetraeder-plan ledger`: evaluates, prints the table and, with `write`, runs the ratchet and
/// rewrites `ledger.json` and every session's `calib.json`.
pub fn ledger_cmd(
    root: &Path,
    write_files: bool,
    accept_worse: &[String],
) -> Result<String, String> {
    let current = evaluate(root);
    let mut out = table(&current);
    if !write_files {
        return Ok(out);
    }
    let accept: Vec<(String, String)> = accept_worse
        .iter()
        .map(|a| {
            a.split_once('=')
                .map(|(i, r)| (i.to_string(), r.to_string()))
                .ok_or_else(|| format!("--accept-worse {a}: expected id=reason"))
        })
        .collect::<Result<_, _>>()?;
    let path = ledger_path(root);
    let previous: Option<Ledger> = read_json(&path).and_then(|v| serde_json::from_value(v).ok());
    let next = write(&current, previous.as_ref(), &accept)?;
    for s in &next.sessions {
        if let Some(c) = &s.recomputed {
            let json = serde_json::to_string_pretty(c).map_err(|e| e.to_string())?;
            std::fs::write(s.dir.join("calib.json"), json + "\n").map_err(|e| e.to_string())?;
        }
    }
    let json = serde_json::to_string_pretty(&next).map_err(|e| e.to_string())?;
    std::fs::write(&path, json + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
    out += &format!("wrote {}\n", path.display());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, pred: f64, meas: f64) -> LedgerEntry {
        LedgerEntry {
            path: "d/f".into(),
            id: id.into(),
            session: "s".into(),
            class: "moe".into(),
            status: "checked".into(),
            predicted_s: Some(pred),
            measured_s: Some(meas),
            err: Some((pred - meas).abs() / meas),
            spread: 0.0,
            cost_model_version: 1,
        }
    }

    fn ledger(entries: Vec<LedgerEntry>) -> Ledger {
        Ledger {
            cost_model_version: 1,
            sessions: Vec::new(),
            entries,
            accepted: Vec::new(),
        }
    }

    fn session(calib: Option<(Value, Value)>) -> SessionRecord {
        let key = SessionKey {
            engine_commit: "c".into(),
            host_id: "h".into(),
            cpu_variant: None,
        };
        let mut r = SessionRecord {
            name: "s".into(),
            key: key.clone(),
            class: "moe".into(),
            anchors: vec!["a".into()],
            fit_band: 0.05,
            tier_source: String::new(),
            dir: PathBuf::from("d/f"),
            recomputed: None,
            stored: None,
        };
        if let Some((stored, _)) = calib {
            r.recomputed = Some(Calib {
                cost_model_version: 1,
                session: "s".into(),
                key,
                anchors: vec!["a".into()],
                band: 0.05,
                hw: HardwareGraph::default(),
            });
            r.stored = Some(stored);
        }
        r
    }

    // hardware: none
    #[test]
    fn equal_ledgers_pass() {
        let a = ledger(vec![entry("c01", 10.0, 10.2), entry("c02", 20.0, 19.0)]);
        assert!(check(&a, &a.clone()).is_empty());
    }

    // hardware: none
    #[test]
    fn a_version_mismatch_is_one_message() {
        let a = ledger(vec![entry("c01", 10.0, 10.2)]);
        let mut b = a.clone();
        b.cost_model_version = 2;
        b.entries[0].predicted_s = Some(11.0);
        assert_eq!(check(&b, &a).len(), 1);
    }

    // hardware: none
    #[test]
    fn a_small_prediction_change_is_caught() {
        let a = ledger(vec![entry("c01", 10.0, 10.2)]);
        let mut b = a.clone();
        b.entries[0].predicted_s = Some(10.0 * (1.0 + 1e-6));
        let m = check(&b, &a);
        assert_eq!(m.len(), 1, "{m:?}");
        // below the tolerance is not a change
        b.entries[0].predicted_s = Some(10.0 * (1.0 + 1e-12));
        assert!(check(&b, &a).is_empty());
    }

    // hardware: none
    #[test]
    fn a_session_median_over_its_band_is_one_message() {
        // errors .2 .2 .01: median .2 over the MoE band
        let mut l = ledger(vec![
            entry("a", 12.0, 10.0),
            entry("b", 12.0, 10.0),
            entry("c", 10.1, 10.0),
        ]);
        l.sessions.push(session(None));
        let m = check(&l, &l.clone());
        assert_eq!(m.len(), 1, "{m:?}");
        assert!(m[0].contains("median"));
        // errors .0 .0 .2: median under the band
        l.entries[0].predicted_s = Some(10.0);
        l.entries[0].err = Some(0.0);
        l.entries[1].predicted_s = Some(10.0);
        l.entries[1].err = Some(0.0);
        assert!(check(&l, &l.clone()).is_empty());
    }

    // hardware: none
    #[test]
    fn a_stale_calib_json_is_one_message() {
        let fresh = serde_json::to_value(
            session(Some((Value::Null, Value::Null)))
                .recomputed
                .unwrap(),
        )
        .unwrap();
        let mut stale = fresh.clone();
        stale["band"] = serde_json::json!(0.06);
        let mut l = ledger(vec![entry("a", 10.0, 10.0)]);
        l.sessions.push(session(Some((fresh, Value::Null))));
        assert!(check(&l, &l.clone()).is_empty());
        l.sessions[0].stored = Some(stale);
        let m = check(&l, &l.clone());
        assert_eq!(m.len(), 1, "{m:?}");
        assert!(m[0].contains("calib.json"));
        l.sessions[0].stored = None;
        assert_eq!(check(&l, &l.clone()).len(), 1);
    }

    // hardware: none
    #[test]
    fn a_missing_entry_is_one_message() {
        let a = ledger(vec![entry("c01", 10.0, 10.2), entry("c02", 20.0, 19.0)]);
        let b = ledger(vec![entry("c01", 10.0, 10.2)]);
        let m = check(&b, &a);
        assert_eq!(m.len(), 1, "{m:?}");
        assert!(m[0].contains("c02"));
        // a new entry is listed by the table but is not a failure
        assert!(check(&a, &b).is_empty());
    }

    // hardware: none
    #[test]
    fn the_ratchet_refuses_a_rise_over_the_floor() {
        let prev = ledger(vec![entry("c01", 10.0, 10.0)]);
        let worse = |d: f64| ledger(vec![entry("c01", 10.0 * (1.0 + d), 10.0)]);
        assert!(write(&worse(0.03), Some(&prev), &[]).is_err());
        let ok = write(&worse(0.01), Some(&prev), &[]).unwrap();
        assert!(ok.accepted.is_empty());
        let acc = [("c01".to_string(), "new term moved it".to_string())];
        let ok = write(&worse(0.03), Some(&prev), &acc).unwrap();
        assert_eq!(ok.accepted.len(), 1);
        assert_eq!(ok.accepted[0].reason, "new term moved it");
        // the repeat spread widens the slack
        let mut w = worse(0.03);
        w.entries[0].spread = 0.05;
        assert!(write(&w, Some(&prev), &[]).is_ok());
        // an improvement and a first snapshot always pass
        assert!(write(&worse(0.0), Some(&worse(0.1)), &[]).is_ok());
        assert!(write(&worse(0.5), None, &[]).is_ok());
    }

    // hardware: none
    #[test]
    fn spread_is_range_over_median() {
        assert_eq!(spread(&[5.0]), 0.0);
        assert!((spread(&[9.0, 10.0, 11.0]) - 0.2).abs() < 1e-12);
    }

    fn results_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/results")
    }

    // hardware: none
    #[test]
    fn committed_fixtures_are_checked_with_their_known_medians() {
        let root = results_root();
        if !root.join("20261103/planner-moe/candidates.json").is_file() {
            eprintln!("skip: planner fixtures not committed");
            return;
        }
        let l = evaluate(&root);
        eprintln!("{}", table(&l));
        let med = |class: &str| {
            let names: Vec<&str> = l
                .sessions
                .iter()
                .filter(|s| s.class == class)
                .map(|s| s.name.as_str())
                .collect();
            assert_eq!(names.len(), 1, "{class}: {names:?}");
            session_median(&l, names[0]).unwrap()
        };
        // ADR 0056 context: MoE about 4.3%, dense about 2.1%
        assert!((med("moe") - 0.043).abs() < 0.005, "moe {}", med("moe"));
        assert!(
            (med("dense") - 0.021).abs() < 0.005,
            "dense {}",
            med("dense")
        );
        assert!(l.entries.iter().any(|e| e.status == "checked"));
    }

    // hardware: none
    #[test]
    fn a_fixture_without_an_engine_commit_is_uncalibrated() {
        let dir = std::env::temp_dir().join(format!("tetraeder-ledger-{}", std::process::id()));
        let fx = dir.join("20260101/planner-x");
        let src = results_root().join("20261103/planner-moe");
        if !src.join("candidates.json").is_file() {
            eprintln!("skip: planner fixtures not committed");
            return;
        }
        std::fs::create_dir_all(&fx).unwrap();
        for n in ["layout.json", "candidates.json", "planner-moe-112053.json"] {
            std::fs::copy(src.join(n), fx.join(n)).unwrap();
        }
        // its env says git_commit "unknown" and there is no provenance.json
        let l = evaluate(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert!(l.sessions.is_empty());
        assert!(!l.entries.is_empty());
        assert!(l.entries.iter().all(|e| e.status == "uncalibrated"));
    }
}
