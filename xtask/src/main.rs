//! Repository automation: `cargo xtask <task>` (alias in `.cargo/config.toml`).
//!
//! - `fetch-model`: download the Qwen3-0.6B Q8_0 test model into `.hf_cache/`.
//! - `llama-tools`: build `llama-simple` and `llama-bench` from `vendor/llama.cpp`.
//! - `golden`: record the llama-simple reference text and our bench-format golden (ADR 0026).
//!
//! Only std and external tools (curl, cmake, cargo): no extra dependencies.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const MODEL_URL: &str =
    "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf";
const MODEL_DIR: &str = "Qwen__Qwen3-0.6B-GGUF";
const MODEL_FILE: &str = "Qwen3-0.6B-Q8_0.gguf";
const MODEL_ID: &str = "Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf";
const LLAMA_COMMIT: &str = "2e0d86be";
const N_PREDICT: u32 = 32;
const PROMPTS: [(&str, &str); 3] = [
    ("capital", "The capital of France is"),
    ("fibonacci", "def fibonacci(n):"),
    ("story", "Once upon a time"),
];

type Res<T> = Result<T, String>;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace root")
        .to_path_buf()
}

/// `TETRAEDER_MODELS_DIR`, else the nearest `.hf_cache` at or above the repo root (a git
/// worktree under `.claude/worktrees/` finds the main checkout's cache).
fn models_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("TETRAEDER_MODELS_DIR") {
        return PathBuf::from(d);
    }
    let r = root();
    r.ancestors()
        .map(|a| a.join(".hf_cache"))
        .find(|p| p.is_dir())
        .unwrap_or_else(|| r.join(".hf_cache"))
}

fn run(cmd: &mut Command) -> Res<()> {
    eprintln!("+ {cmd:?}");
    let st = cmd
        .status()
        .map_err(|e| format!("cannot start {cmd:?}: {e}"))?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("{cmd:?} failed: {st}"))
    }
}

fn fetch_model() -> Res<()> {
    let dir = root().join(".hf_cache").join(MODEL_DIR);
    let dest = dir.join(MODEL_FILE);
    if dest.is_file() || models_dir().join(MODEL_DIR).join(MODEL_FILE).is_file() {
        eprintln!("model present, nothing to do");
        return Ok(());
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let part = dir.join(format!("{MODEL_FILE}.part"));
    run(Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "-o"])
        .arg(&part)
        .arg(MODEL_URL))?;
    std::fs::rename(&part, &dest).map_err(|e| e.to_string())
}

fn tools_dir() -> PathBuf {
    root().join("target").join("llama-tools")
}

fn llama_tools() -> Res<()> {
    let build = tools_dir();
    let mut cfg = Command::new("cmake");
    cfg.arg("-S")
        .arg(root().join("vendor/llama.cpp"))
        .arg("-B")
        .arg(&build)
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF",
            // Same ggml options as crates/ggml-sys/build.rs so the reference matches our build.
            "-DGGML_OPENMP=OFF",
            "-DLLAMA_BUILD_COMMON=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
            "-DLLAMA_BUILD_APP=OFF",
            "-DLLAMA_BUILD_MTMD=OFF",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_OPENSSL=OFF",
        ]);
    if let Ok(native) = std::env::var("TETRAEDER_GGML_NATIVE") {
        cfg.arg(format!("-DGGML_NATIVE={native}"));
    }
    run(&mut cfg)?;

    let mut b = Command::new("cmake");
    b.arg("--build").arg(&build).args([
        "--config",
        "Release",
        "--target",
        "llama-simple",
        "llama-bench",
    ]);
    if let Ok(j) = std::env::var("CMAKE_BUILD_PARALLEL_LEVEL") {
        b.args(["--parallel", &j]);
    }
    run(&mut b)
}

fn tool_path(name: &str) -> Res<PathBuf> {
    let exe = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    let bin = tools_dir().join("bin");
    [bin.join("Release").join(&exe), bin.join(&exe)]
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| format!("{name} not built; run `cargo xtask llama-tools`"))
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn golden() -> Res<()> {
    let model = models_dir().join(MODEL_DIR).join(MODEL_FILE);
    if !model.is_file() {
        return Err(format!(
            "{} missing; run `cargo xtask fetch-model`",
            model.display()
        ));
    }
    let simple = tool_path("llama-simple")?;
    let mut cases = Vec::new();
    for (id, prompt) in PROMPTS {
        let out = Command::new(&simple)
            .arg("-m")
            .arg(&model)
            .args(["-n", &N_PREDICT.to_string(), "-ngl", "0", prompt])
            .output()
            .map_err(|e| format!("cannot run llama-simple: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "llama-simple failed: {}\n{}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        // Windows stdout is a text-mode stream: "\n" arrives as "\r\n".
        let text = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        // llama-simple ends the output with one "\n".
        let text = text.strip_suffix('\n').unwrap_or(&text).to_owned();
        eprintln!("{id}: {text:?}");
        cases.push(format!(
            "    {{\"id\": {}, \"prompt\": {}, \"text\": {}}}",
            json_str(id),
            json_str(prompt),
            json_str(&text)
        ));
    }
    let json = format!(
        "{{\n  \"schema\": 1,\n  \"model\": {},\n  \"engine\": {},\n  \"backend\": \"cpu\",\n  \"n_predict\": {N_PREDICT},\n  \"cases\": [\n{}\n  ]\n}}\n",
        json_str(MODEL_ID),
        json_str(&format!("llama.cpp@{LLAMA_COMMIT} llama-simple")),
        cases.join(",\n")
    );
    let dir = root().join("bench/golden");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("qwen3-0.6b-q8_0.llama-simple.json");
    std::fs::write(&path, json).map_err(|e| e.to_string())?;
    eprintln!("wrote {}", path.display());

    run(
        Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .args([
                "test",
                "-p",
                "tetraeder-engine",
                "--test",
                "golden",
                "--",
                "--ignored",
                "record_golden",
                "--nocapture",
            ])
            .env("TETRAEDER_RECORD_GOLDEN", "1")
            .env("TETRAEDER_MODELS_DIR", models_dir())
            .current_dir(root()),
    )
}

fn main() -> ExitCode {
    let task = std::env::args().nth(1).unwrap_or_default();
    let res = match task.as_str() {
        "fetch-model" => fetch_model(),
        "llama-tools" => llama_tools(),
        "golden" => golden(),
        _ => Err("usage: cargo xtask <fetch-model|llama-tools|golden>".into()),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask: {e}");
            ExitCode::FAILURE
        }
    }
}
