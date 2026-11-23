//! `POST /slots/{id_slot}?action=save|restore|erase` (ADR 0090), with llama-server's checks,
//! error texts and bodies (`post_slots` and its handlers at the pin).

use crate::engine::{SlotAction, SlotReply};
use crate::error::{LlamaError, bad};
use crate::gateway::Gateway;
use crate::native::json_body;
use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

type Result<T> = std::result::Result<T, LlamaError>;

/// `fs_validate_filename` of llama.cpp without subdirectories: no path separators, control or
/// reserved characters, no `..`, no leading or trailing space, no trailing dot, 255 bytes at most.
pub fn valid_filename(name: &str) -> bool {
    if name.is_empty() || name.len() > 255 {
        return false;
    }
    let forbidden = |c: char| {
        let u = u32::from(c);
        u <= 0x1F
            || (0x7F..=0x9F).contains(&u)
            || matches!(u, 0xFF0E | 0x2215 | 0x2216 | 0xFFFD | 0xFEFF)
            || matches!(c, ':' | '*' | '?' | '"' | '<' | '>' | '|' | '/' | '\\')
    };
    !name.chars().any(forbidden)
        && !name.starts_with(' ')
        && !name.ends_with([' ', '.'])
        && !name.contains("..")
        && !windows_reserved(name)
}

/// A Windows device name (`CON`, `NUL`, `COM1`, `LPT3`, ...), with or without an extension and
/// in any case. Refused on every platform so a save directory moves between systems.
fn windows_reserved(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    let stem = stem.trim_end_matches(' ');
    if matches!(stem, "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$") {
        return true;
    }
    ["COM", "LPT"].iter().any(|p| {
        stem.strip_prefix(p).is_some_and(|d| {
            let mut c = d.chars();
            matches!(c.next(), Some('0'..='9' | '\u{b9}' | '\u{b2}' | '\u{b3}'))
                && c.next().is_none()
        })
    })
}

/// The filename of a save or restore body (`filename`, a string).
fn filename(body: &Bytes) -> Result<String> {
    let body = json_body(body)?;
    let name = body
        .get("filename")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("\"filename\" must be provided as a string"))?;
    if !valid_filename(name) {
        return Err(bad("Invalid filename"));
    }
    Ok(name.to_owned())
}

pub async fn slot_action(
    State(gw): State<Arc<Gateway>>,
    Path(id): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    body: Bytes,
) -> Result<Json<Value>> {
    if !gw.slot_actions() {
        return Err(LlamaError::not_supported(
            "This server does not support slots action. Start it with `--slot-save-path`",
        ));
    }
    let id: i32 = id.parse().map_err(|_| bad("Invalid slot ID"))?;
    let action = match query.get("action").map(String::as_str) {
        Some("save") => SlotAction::Save {
            filename: filename(&body)?,
        },
        Some("restore") => SlotAction::Restore {
            filename: filename(&body)?,
        },
        Some("erase") => SlotAction::Erase,
        _ => return Err(bad("Invalid action")),
    };
    let saved = matches!(action, SlotAction::Save { .. });
    let name = match &action {
        SlotAction::Save { filename } | SlotAction::Restore { filename } => filename.clone(),
        SlotAction::Erase => String::new(),
    };
    // The answer is awaited, not blocked on: a client that goes away drops the receiver, which
    // cancels the action if it has not started.
    let wait = gw.slot_action(id, action)?;
    let reply = wait
        .await
        .map_err(|_| {
            LlamaError::from(crate::error::GatewayError::Internal(
                "the engine worker has stopped".into(),
            ))
        })?
        .map_err(|e| LlamaError::from(gw.slot_err(e)))?;
    Ok(Json(match reply {
        SlotReply::Saved { tokens, bytes, ms } if saved => json!({
            "id_slot": id,
            "filename": name,
            "n_saved": tokens,
            "n_written": bytes,
            "timings": {"save_ms": ms},
        }),
        SlotReply::Saved { .. } => unreachable!("only a save is answered with a save"),
        SlotReply::Restored { tokens, bytes, ms } => json!({
            "id_slot": id,
            "filename": name,
            "n_restored": tokens,
            "n_read": bytes,
            "timings": {"restore_ms": ms},
        }),
        SlotReply::Erased { tokens } => json!({"id_slot": id, "n_erased": tokens}),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // hardware: none
    #[test]
    fn filenames_follow_llama_cpps_rules() {
        for ok in ["s.bin", "slot 1.bin", "a-b_c.d", "ü.bin"] {
            assert!(valid_filename(ok), "{ok}");
        }
        for bad in [
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            "../x",
            "a..b",
            "x.",
            " x",
            "x ",
            "a:b",
            "a*b",
            "a?b",
            "a\"b",
            "a<b",
            "a>b",
            "a|b",
            "a\nb",
            "a\u{7f}b",
            "a\u{fffd}b",
            &"x".repeat(256),
        ] {
            assert!(!valid_filename(bad), "{bad:?}");
        }
        assert!(valid_filename(&"x".repeat(255)));
        for reserved in [
            "CON",
            "con",
            "Prn.txt",
            "aux.tar.gz",
            "NUL.bin",
            "COM1",
            "com0.bin",
            "LPT9",
            "lpt3.x",
            "COM\u{b9}",
            "NUL .txt",
        ] {
            assert!(!valid_filename(reserved), "{reserved}");
        }
        for fine in [
            "COM10",
            "COMX",
            "CONSOLE",
            "xcon",
            "LPT",
            "con-1.bin",
            "COM1x",
        ] {
            assert!(valid_filename(fine), "{fine}");
        }
    }
}
