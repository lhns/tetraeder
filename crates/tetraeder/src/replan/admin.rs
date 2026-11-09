//! The admin listener: how triggers that come from outside the process reach the replanner, and
//! what is running (ADR 0055).
//!
//! - `POST /api/v2/alerts`: the Alertmanager v2 payload vmalert's notifier sends. Firing
//!   `TetraederSloBurnFast` and `TetraederSloBurnSlow` become `SloBurn { sli }`, each
//!   `(alertname, sli)` at most once per dwell; everything else is ignored. Always 200.
//! - `POST /admin/replan`: `{"trigger": "drift"|"slo_burn"|"load_shift"|"device_change",
//!   "sli"?: str, "residual"?: f64}`. 202 when queued, with `?wait=1` 200 and the outcome.
//! - `GET /admin/plan`: the running plan and the last decision.
//!
//! The listener is for the operator's own machine (default `127.0.0.1`); it has no
//! authentication.

use super::control::ReplanHandle;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tetraeder_control::{Outcome, Trigger};

const BURN_ALERTS: [&str; 2] = ["TetraederSloBurnFast", "TetraederSloBurnSlow"];

struct AdminState {
    handle: ReplanHandle,
    dwell_s: f64,
    seen: Mutex<HashMap<(String, String), f64>>,
    now: fn() -> f64,
}

pub fn router(handle: ReplanHandle, dwell_s: f64) -> Router {
    router_with_clock(handle, dwell_s, unix_now)
}

fn router_with_clock(handle: ReplanHandle, dwell_s: f64, now: fn() -> f64) -> Router {
    let state = Arc::new(AdminState {
        handle,
        dwell_s,
        seen: Mutex::default(),
        now,
    });
    Router::new()
        .route("/admin/replan", post(replan))
        .route("/admin/plan", get(plan))
        .route("/api/v2/alerts", post(alerts))
        .with_state(state)
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

fn bad(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": msg.into() })),
    )
        .into_response()
}

#[derive(Deserialize)]
struct WaitQuery {
    wait: Option<String>,
}

fn parse_trigger(v: &Value) -> Result<Trigger, String> {
    let kind = v
        .get("trigger")
        .and_then(Value::as_str)
        .ok_or("missing `trigger`")?;
    match kind {
        "drift" => Ok(Trigger::Drift {
            residual: v.get("residual").and_then(Value::as_f64).unwrap_or(0.0),
        }),
        "slo_burn" => Ok(Trigger::SloBurn {
            sli: v
                .get("sli")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
        }),
        "load_shift" => Ok(Trigger::LoadShift),
        "device_change" => Ok(Trigger::DeviceChange),
        other => Err(format!(
            "unknown trigger `{other}`: expected drift, slo_burn, load_shift or device_change"
        )),
    }
}

async fn replan(
    State(s): State<Arc<AdminState>>,
    Query(q): Query<WaitQuery>,
    body: Bytes,
) -> Response {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return bad(format!("bad JSON: {e}")),
    };
    let t = match parse_trigger(&v) {
        Ok(t) => t,
        Err(e) => return bad(e),
    };
    let wait = q.wait.as_deref().is_some_and(|w| w != "0" && w != "false");
    match s.handle.trigger(t, wait) {
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "error": e }))).into_response(),
        Ok(None) => (StatusCode::ACCEPTED, Json(json!({ "queued": true }))).into_response(),
        Ok(Some(rx)) => match rx.await {
            Ok(o) => {
                let mut out = json!({ "outcome": o.label() });
                match o {
                    Outcome::Rejected { reason } => out["reason"] = reason.into(),
                    Outcome::RolledBack { error } => out["reason"] = error.into(),
                    _ => {}
                }
                (StatusCode::OK, Json(out)).into_response()
            }
            Err(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": "the replan thread stopped" })),
            )
                .into_response(),
        },
    }
}

async fn plan(State(s): State<Arc<AdminState>>) -> Response {
    let st = s
        .handle
        .status
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    Json(st).into_response()
}

async fn alerts(State(s): State<Arc<AdminState>>, body: Bytes) -> Response {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return bad(format!("bad JSON: {e}")),
    };
    let Some(list) = v.as_array() else {
        return bad("expected an array of alerts");
    };
    let now = (s.now)();
    for a in list {
        let label = |k: &str| a.pointer(&format!("/labels/{k}")).and_then(Value::as_str);
        let Some(name) = label("alertname").filter(|n| BURN_ALERTS.contains(n)) else {
            continue;
        };
        let firing = match a.get("endsAt").and_then(Value::as_str) {
            None | Some("") => true,
            Some(t) => parse_rfc3339(t).is_none_or(|end| end > now),
        };
        if !firing {
            continue;
        }
        let sli = label("sli").unwrap_or("unknown").to_string();
        {
            let mut seen = s.seen.lock().unwrap_or_else(|e| e.into_inner());
            let key = (name.to_string(), sli.clone());
            if seen.get(&key).is_some_and(|&t| now - t < s.dwell_s) {
                continue;
            }
            seen.insert(key, now);
        }
        // A stopped replan thread is not the alert sender's problem.
        let _ = s.handle.trigger(Trigger::SloBurn { sli }, false);
    }
    (StatusCode::OK, Json(json!({}))).into_response()
}

/// Seconds since the Unix epoch of an RFC 3339 time (`2026-11-09T12:00:00.5+01:00`).
fn parse_rfc3339(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[7] != b'-' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    if !matches!(b[10], b'T' | b't' | b' ') {
        return None;
    }
    let num = |a: usize, n: usize| -> Option<i64> { s.get(a..a + n)?.parse().ok() };
    let (y, mo, d) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
    let (h, mi, sec) = (num(11, 2)?, num(14, 2)?, num(17, 2)?);
    let mut i = 19;
    let mut frac = 0.0;
    if b[i] == b'.' {
        let start = i;
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        frac = s[start..i].parse().ok()?;
    }
    let offset = match b.get(i)? {
        b'Z' | b'z' => 0,
        sign @ (b'+' | b'-') => {
            let (oh, om) = (num(i + 1, 2)?, num(i + 4, 2)?);
            let o = oh * 3600 + om * 60;
            if *sign == b'+' { o } else { -o }
        }
        _ => return None,
    };
    // Days from civil date (Howard Hinnant's algorithm).
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((mo + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some((days * 86_400 + h * 3600 + mi * 60 + sec - offset) as f64 + frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replan::control::{Cmd, Status};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use std::sync::mpsc;
    use tetraeder_plan::{Knobs, Plan};
    use tower::ServiceExt;

    fn handle() -> (ReplanHandle, mpsc::Receiver<Cmd>) {
        let (tx, rx) = mpsc::channel();
        let plan = Plan {
            model: "m".into(),
            knobs: Knobs {
                threads: Some(2),
                ..Knobs::default()
            },
            replicas: vec![],
        };
        let status = Arc::new(Mutex::new(Status::new(&plan, None, None)));
        (ReplanHandle::new(tx, status), rx)
    }

    // 2026-11-09T12:00:00Z
    fn fixed_now() -> f64 {
        parse_rfc3339("2026-11-09T12:00:00Z").unwrap()
    }

    async fn post_json(app: &Router, uri: &str, body: &str) -> (StatusCode, Value) {
        let req = Request::post(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    fn sent(rx: &mpsc::Receiver<Cmd>) -> Vec<Trigger> {
        rx.try_iter()
            .map(|Cmd::Trigger(t, _)| t)
            .collect::<Vec<_>>()
    }

    fn alert(name: &str, sli: &str, ends: Option<&str>) -> String {
        let ends = ends.map_or(String::new(), |e| format!(r#","endsAt":"{e}""#));
        format!(
            r#"{{"labels":{{"alertname":"{name}","sli":"{sli}","severity":"page"}},"annotations":{{}},"startsAt":"2026-11-09T11:00:00Z"{ends}}}"#
        )
    }

    // hardware: none
    #[tokio::test]
    async fn the_alert_payload_maps_to_slo_burn() {
        let (h, rx) = handle();
        let app = router_with_clock(h, 600.0, fixed_now);
        let body = format!(
            "[{}]",
            alert("TetraederSloBurnFast", "ttft", Some("2026-11-09T12:05:00Z"))
        );
        let (st, _) = post_json(&app, "/api/v2/alerts", &body).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(sent(&rx), vec![Trigger::SloBurn { sli: "ttft".into() }]);
        // no endsAt at all is firing too
        let body = format!("[{}]", alert("TetraederSloBurnSlow", "tpot", None));
        post_json(&app, "/api/v2/alerts", &body).await;
        assert_eq!(sent(&rx), vec![Trigger::SloBurn { sli: "tpot".into() }]);
    }

    // hardware: none
    #[tokio::test]
    async fn resolved_and_other_alerts_are_ignored() {
        let (h, rx) = handle();
        let app = router_with_clock(h, 600.0, fixed_now);
        let body = format!(
            "[{},{}]",
            alert("TetraederSloBurnFast", "ttft", Some("2026-11-09T11:59:00Z")),
            alert(
                "TetraederReplanFailed",
                "ttft",
                Some("2026-11-09T12:05:00Z")
            ),
        );
        let (st, _) = post_json(&app, "/api/v2/alerts", &body).await;
        assert_eq!(st, StatusCode::OK);
        assert!(sent(&rx).is_empty());
    }

    // hardware: none
    #[tokio::test]
    async fn a_repeat_within_the_dwell_is_dropped() {
        let (h, rx) = handle();
        let app = router_with_clock(h, 600.0, fixed_now);
        let one = format!("[{}]", alert("TetraederSloBurnFast", "ttft", None));
        let other_sli = format!("[{}]", alert("TetraederSloBurnFast", "tpot", None));
        post_json(&app, "/api/v2/alerts", &one).await;
        post_json(&app, "/api/v2/alerts", &one).await;
        post_json(&app, "/api/v2/alerts", &other_sli).await;
        assert_eq!(
            sent(&rx),
            vec![
                Trigger::SloBurn { sli: "ttft".into() },
                Trigger::SloBurn { sli: "tpot".into() }
            ]
        );
    }

    // hardware: none
    #[tokio::test]
    async fn bad_bodies_are_400() {
        let (h, rx) = handle();
        let app = router_with_clock(h, 600.0, fixed_now);
        for (uri, body) in [
            ("/api/v2/alerts", "{not json"),
            ("/api/v2/alerts", r#"{"labels":{}}"#),
            ("/admin/replan", "{not json"),
            ("/admin/replan", r#"{"trigger":"reboot"}"#),
            ("/admin/replan", r#"{}"#),
        ] {
            let (st, _) = post_json(&app, uri, body).await;
            assert_eq!(st, StatusCode::BAD_REQUEST, "{uri} {body}");
        }
        assert!(sent(&rx).is_empty());
    }

    // hardware: none
    #[tokio::test]
    async fn the_replan_route_queues_and_can_wait() {
        let (h, rx) = handle();
        let app = router_with_clock(h, 600.0, fixed_now);
        let (st, v) = post_json(&app, "/admin/replan", r#"{"trigger":"load_shift"}"#).await;
        assert_eq!(
            (st, v["queued"].as_bool()),
            (StatusCode::ACCEPTED, Some(true))
        );
        assert_eq!(sent(&rx), vec![Trigger::LoadShift]);
        // with ?wait=1 the answer is the outcome the replan thread sends back
        let answer = std::thread::spawn(move || {
            let Cmd::Trigger(t, reply) = rx.recv().unwrap();
            assert_eq!(t, Trigger::Drift { residual: 0.7 });
            reply
                .unwrap()
                .send(Outcome::Rejected {
                    reason: "dwell".into(),
                })
                .unwrap();
        });
        let (st, v) = post_json(
            &app,
            "/admin/replan?wait=1",
            r#"{"trigger":"drift","residual":0.7}"#,
        )
        .await;
        answer.join().unwrap();
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["outcome"], "rejected");
        assert_eq!(v["reason"], "dwell");
    }

    // hardware: none
    #[tokio::test]
    async fn the_plan_route_reports_the_status() {
        let (h, _rx) = handle();
        let app = router_with_clock(h, 600.0, fixed_now);
        let res = app
            .clone()
            .oneshot(Request::get("/admin/plan").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            (v["threads"].as_u64(), v["switches"].as_u64()),
            (Some(2), Some(0))
        );
    }

    // hardware: none
    #[test]
    fn rfc3339_parses_zones_and_fractions() {
        let z = parse_rfc3339("2026-11-09T12:00:00Z").unwrap();
        assert_eq!(parse_rfc3339("2026-11-09T13:00:00+01:00"), Some(z));
        assert_eq!(parse_rfc3339("2026-11-09T11:00:00-01:00"), Some(z));
        assert_eq!(parse_rfc3339("2026-11-09T12:00:00.5Z"), Some(z + 0.5));
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0.0));
        assert_eq!(parse_rfc3339("2000-03-01T00:00:00Z"), Some(951_868_800.0));
        assert_eq!(parse_rfc3339("yesterday"), None);
        assert_eq!(parse_rfc3339("2026-11-09 12:00:00"), None);
    }
}
