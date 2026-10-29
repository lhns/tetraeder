//! Chat templates: Hugging Face Jinja templates rendered with minijinja.
//!
//! The environment mirrors transformers: `trim_blocks` and `lstrip_blocks` on, lenient undefined,
//! Python string and dict methods, `raise_exception`, `strftime_now` and a `tojson` that matches
//! `json.dumps(x, ensure_ascii=False)` byte for byte.

use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Serde};
use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior, Value};
use serde::Serialize;
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};

use crate::chat::{ChatMessage, ToolDef};

const NAME: &str = "chat";

#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("template syntax error: {0}")]
    Syntax(String),
    #[error("template render error: {0}")]
    Render(String),
}

pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn new(
        source: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .keep_trailing_newline(false)
            .build()
            .map_err(|e| TemplateError::Syntax(e.to_string()))?;
        env.set_syntax(syntax);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", |fmt: String| strftime(&fmt, now_secs()));
        // jinja2 turns every newline sequence in the source into "\n", string literals included.
        let normalized = source.replace("\r\n", "\n").replace('\r', "\n");
        env.add_template_owned(NAME, normalized)
            .map_err(|e| TemplateError::Syntax(format!("{e:#}")))?;
        Ok(Self {
            env,
            source: source.to_owned(),
            bos_token: bos_token.unwrap_or_default().to_owned(),
            eos_token: eos_token.unwrap_or_default().to_owned(),
        })
    }

    /// Fixes `strftime_now` to this Unix time; for tests.
    pub fn with_fixed_time(mut self, unix_secs: i64) -> Self {
        self.env
            .add_function("strftime_now", move |fmt: String| strftime(&fmt, unix_secs));
        self
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        add_generation_prompt: bool,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, TemplateError> {
        let mut ctx = kwargs.clone();
        ctx.insert("messages".into(), to_json(messages)?);
        // HF passes no `tools` at all when there are none, so `tools is defined` is false.
        ctx.remove("tools");
        if !tools.is_empty() {
            ctx.insert("tools".into(), to_json(tools)?);
        }
        ctx.insert("add_generation_prompt".into(), add_generation_prompt.into());
        ctx.insert("bos_token".into(), self.bos_token.clone().into());
        ctx.insert("eos_token".into(), self.eos_token.clone().into());
        let ctx = Value::from(Serde(serde_json::Value::Object(ctx)));
        let tmpl = self
            .env
            .get_template(NAME)
            .map_err(|e| TemplateError::Render(format!("{e:#}")))?;
        tmpl.render(ctx)
            .map_err(|e| TemplateError::Render(error_text(&e)))
    }

    pub fn source(&self) -> &str {
        &self.source
    }
}

fn to_json<T: Serialize + ?Sized>(v: &T) -> Result<serde_json::Value, TemplateError> {
    serde_json::to_value(v).map_err(|e| TemplateError::Render(e.to_string()))
}

/// Marks the error built by `raise_exception`.
#[derive(Debug)]
struct Raised;

impl std::fmt::Display for Raised {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("raised by the template")
    }
}

impl std::error::Error for Raised {}

/// Just the message for `raise_exception`, else the full error with its template position.
fn error_text(e: &Error) -> String {
    let raised = std::error::Error::source(e).is_some_and(|s| s.is::<Raised>());
    match e.detail() {
        Some(d) if raised => d.to_owned(),
        _ => format!("{e:#}"),
    }
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg).with_source(Raised))
}

// ---- tojson ----

/// Python's default `json.dumps` separators without indent: ", " and ": ".
struct SpacedFormatter;

impl Formatter for SpacedFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// `json.dumps(x, ensure_ascii=False, indent=indent)` as transformers' `tojson` calls it.
/// Also accepts `ensure_ascii` and `sort_keys`.
fn tojson(value: Value, indent: Option<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let indent = match indent {
        Some(v) => Some(v),
        None => kwargs.get::<Option<Value>>("indent")?,
    };
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    kwargs.assert_all_used()?;
    let indent = match indent {
        Some(v) if !v.is_none() && !v.is_undefined() => {
            Some(usize::try_from(v).map_err(|_| {
                Error::new(ErrorKind::InvalidOperation, "indent must be an integer")
            })?)
        }
        _ => None,
    };
    let fail = |e: serde_json::Error| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(e)
    };
    let mut buf = Vec::new();
    let value = if sort_keys {
        let j = serde_json::to_value(&value).map_err(fail)?;
        Value::from(Serde(sort_json(j)))
    } else {
        value
    };
    match indent {
        Some(n) => {
            let pad = vec![b' '; n];
            let mut ser = Serializer::with_formatter(&mut buf, PrettyFormatter::with_indent(&pad));
            value.serialize(&mut ser).map_err(fail)?;
        }
        None => {
            let mut ser = Serializer::with_formatter(&mut buf, SpacedFormatter);
            value.serialize(&mut ser).map_err(fail)?;
        }
    }
    let mut out = String::from_utf8(buf)
        .map_err(|_| Error::new(ErrorKind::InvalidOperation, "invalid UTF-8 in JSON"))?;
    if ensure_ascii {
        out = escape_non_ascii(&out);
    }
    Ok(Value::from_safe_string(out))
}

fn sort_json(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut entries: Vec<_> = m.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k, sort_json(v)))
                    .collect(),
            )
        }
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.into_iter().map(sort_json).collect())
        }
        other => other,
    }
}

/// Non-ASCII only occurs inside JSON strings, so a plain pass is safe.
fn escape_non_ascii(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for u in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{u:04x}"));
            }
        }
    }
    out
}

// ---- strftime ----

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Days since 1970-01-01 to (year, month 1-12, day 1-31).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const DAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// C `strftime` for the UTC time `unix_secs`. Supports the common conversions; an unknown
/// conversion is copied through unchanged.
fn strftime(fmt: &str, unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let sod = unix_secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (sod / 3600, sod % 3600 / 60, sod % 60);
    let wday = (days + 4).rem_euclid(7) as usize; // 1970-01-01 was a Thursday
    let yday = {
        let cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        cum[m as usize - 1] + d + u32::from(m > 2 && is_leap(y))
    };
    let mut out = String::new();
    let mut it = fmt.chars().peekable();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        // `-` (GNU) and `#` (Windows) drop zero padding.
        let nopad = matches!(it.peek(), Some('-' | '#'));
        if nopad {
            it.next();
        }
        let Some(spec) = it.next() else {
            out.push('%');
            break;
        };
        let num = |v: i64, width: usize| {
            if nopad {
                v.to_string()
            } else {
                format!("{v:0width$}")
            }
        };
        match spec {
            'Y' => out.push_str(&y.to_string()),
            'y' => out.push_str(&num(y.rem_euclid(100), 2)),
            'm' => out.push_str(&num(m.into(), 2)),
            'd' => out.push_str(&num(d.into(), 2)),
            'e' => out.push_str(&if nopad {
                d.to_string()
            } else {
                format!("{d:2}")
            }),
            'j' => out.push_str(&num(yday.into(), 3)),
            'H' => out.push_str(&num(hh, 2)),
            'I' => out.push_str(&num(if hh % 12 == 0 { 12 } else { hh % 12 }, 2)),
            'M' => out.push_str(&num(mm, 2)),
            'S' => out.push_str(&num(ss, 2)),
            'p' => out.push_str(if hh < 12 { "AM" } else { "PM" }),
            'B' => out.push_str(MONTHS[m as usize - 1]),
            'b' | 'h' => out.push_str(&MONTHS[m as usize - 1][..3]),
            'A' => out.push_str(DAYS[wday]),
            'a' => out.push_str(&DAYS[wday][..3]),
            'F' => out.push_str(&format!("{y}-{m:02}-{d:02}")),
            'T' => out.push_str(&format!("{hh:02}:{mm:02}:{ss:02}")),
            'Z' => out.push_str("UTC"),
            'z' => out.push_str("+0000"),
            '%' => out.push('%'),
            other => {
                out.push('%');
                out.push(other);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-01-15 12:34:56 UTC, a Thursday.
    const T: i64 = 1_768_478_400 + 34 * 60 + 56;

    #[test]
    fn strftime_formats() {
        assert_eq!(strftime("%Y-%m-%d", T), "2026-01-15");
        assert_eq!(strftime("%d %b %Y", T), "15 Jan 2026");
        assert_eq!(
            strftime("%A %a %B %-d|%e|%H:%M:%S", T),
            "Thursday Thu January 15|15|12:34:56"
        );
        assert_eq!(strftime("%j %y %p %I %%", T), "015 26 PM 12 %");
        assert_eq!(strftime("%-m/%#d", 0), "1/1");
        assert_eq!(strftime("%Y-%m-%d %a", 951_782_400), "2000-02-29 Tue");
        assert_eq!(strftime("%j", 951_782_400), "060");
        assert_eq!(strftime("%Y-%m-%d", -86_400), "1969-12-31");
    }

    #[test]
    fn tojson_matches_python() {
        let t = ChatTemplate::new(
            "{{ x|tojson }}|{{ x|tojson(indent=2) }}|{{ y|tojson(ensure_ascii=true) }}|{{ z|tojson(sort_keys=true) }}",
            None,
            None,
        )
        .unwrap();
        let mut kw = serde_json::Map::new();
        kw.insert(
            "x".into(),
            serde_json::json!({"a": [1, 2.5, null], "b": {}, "c": "<&'>é"}),
        );
        kw.insert("y".into(), serde_json::json!("é😀"));
        kw.insert("z".into(), serde_json::json!({"b": 1, "a": 2}));
        let out = t.render(&[], &[], false, &kw).unwrap();
        let expect = concat!(
            r#"{"a": [1, 2.5, null], "b": {}, "c": "<&'>é"}|"#,
            "{\n  \"a\": [\n    1,\n    2.5,\n    null\n  ],\n  \"b\": {},\n  \"c\": \"<&'>é\"\n}|",
            "\"\\u00e9\\ud83d\\ude00\"|{\"a\": 2, \"b\": 1}",
        );
        assert_eq!(out, expect);
    }

    #[test]
    fn raise_exception_carries_message() {
        let t = ChatTemplate::new("{{ raise_exception('nope') }}", None, None).unwrap();
        let e = t
            .render(&[], &[], false, &serde_json::Map::new())
            .unwrap_err();
        assert!(matches!(&e, TemplateError::Render(m) if m == "nope"), "{e}");
    }

    #[test]
    fn syntax_error_is_reported() {
        assert!(matches!(
            ChatTemplate::new("{% if %}", None, None),
            Err(TemplateError::Syntax(_))
        ));
    }
}
