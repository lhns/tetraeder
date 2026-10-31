//! Exposes the git commit as `TETRAEDER_GIT_SHA` for the `bench` rows (overridable by env).
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
    // `--git-path` also resolves inside linked worktrees, where `.git` is a file.
    if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    if let Some(sha) = git(&["rev-parse", "--short", "HEAD"]).filter(|s| !s.is_empty()) {
        println!("cargo:rustc-env=TETRAEDER_GIT_SHA={sha}");
    }
}
