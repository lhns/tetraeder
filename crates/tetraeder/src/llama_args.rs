//! llama-server's single-dash long flags (`-ctk`, `-nkvo`, `-hf`, ...), which clap cannot take
//! as flags (a short flag is one character and `-hf` would read as `-h -f`). They are rewritten
//! to their double-dash names before parsing (ADR 0090); `--ngl` and its aliases keep the
//! spelling they have.

use std::ffi::OsString;

/// llama-server's spelling and ours.
const RENAMED: [(&str, &str); 9] = [
    ("-nkvo", "--no-kv-offload"),
    ("-kvo", "--kv-offload"),
    ("-ctk", "--cache-type-k"),
    ("-ctv", "--cache-type-v"),
    ("-cmoe", "--cpu-moe"),
    ("-ncmoe", "--n-cpu-moe"),
    ("-hf", "--hf-repo"),
    ("-hfr", "--hf-repo"),
    ("-hff", "--hf-file"),
];

/// The arguments with every llama-server single-dash flag renamed; what follows a `--` is left alone.
pub fn rewrite(args: impl IntoIterator<Item = OsString>) -> Vec<OsString> {
    let mut literal = false;
    args.into_iter()
        .map(|a| {
            if literal {
                return a;
            }
            if a == "--" {
                literal = true;
                return a;
            }
            match RENAMED.iter().find(|(from, _)| a == *from) {
                Some((_, to)) => OsString::from(*to),
                None => a,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Vec<String> {
        rewrite(args.iter().map(OsString::from))
            .into_iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    // hardware: none
    #[test]
    fn llama_server_flags_get_their_long_names() {
        assert_eq!(
            run(&[
                "serve", "-ctk", "q8_0", "-nkvo", "-hf", "a/b", "-cmoe", "-ncmoe", "3"
            ]),
            [
                "serve",
                "--cache-type-k",
                "q8_0",
                "--no-kv-offload",
                "--hf-repo",
                "a/b",
                "--cpu-moe",
                "--n-cpu-moe",
                "3"
            ]
        );
    }

    // hardware: none
    #[test]
    fn other_arguments_and_literals_stay() {
        assert_eq!(
            run(&["serve", "-m", "x.gguf", "-t", "2", "--ctk", "--", "-ctk"]),
            ["serve", "-m", "x.gguf", "-t", "2", "--ctk", "--", "-ctk"]
        );
    }
}
