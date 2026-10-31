//! Exposes the git commit as `TETRAEDER_GIT_SHA` for the `bench` rows (overridable by env;
//! falls back to `GIT_SHA`, filled by `git archive`).
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-env-changed=TETRAEDER_GIT_SHA");
    if std::env::var_os("TETRAEDER_GIT_SHA").is_some() {
        return;
    }
    // `git archive` (Vast boxes) has no .git; export-subst fills GIT_SHA at archive time.
    println!("cargo:rerun-if-changed=GIT_SHA");
    // `--git-path` also resolves inside linked worktrees, where `.git` is a file.
    if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    let archived = std::fs::read_to_string("GIT_SHA")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty() && !s.starts_with("$Format"));
    if let Some(sha) = git(&["rev-parse", "--short", "HEAD"])
        .filter(|s| !s.is_empty())
        .or(archived)
    {
        println!("cargo:rustc-env=TETRAEDER_GIT_SHA={sha}");
    }
}
