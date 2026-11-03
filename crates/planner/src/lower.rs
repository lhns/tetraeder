//! A candidate as llama-bench arguments.

use crate::candidate::{Candidate, Placement};
use crate::layout::ModelLayout;

/// llama-bench `-ngl` for a dense placement. llama.cpp offloads the last `n - 1` layers and the
/// output head for `-ngl n` (the input layer never), so `g` layers plus the head are `g + 1`;
/// `L + 1` is everything.
pub fn dense_ngl(gpu_layers: u32, n_layer: u32) -> u32 {
    if gpu_layers == 0 {
        0
    } else {
        (gpu_layers + 1).min(n_layer + 1)
    }
}

/// Regex alternatives matching exactly the decimal numbers `lo..=hi` (no leading zeros), for
/// example `24..=47` gives `2[4-9]|3[0-9]|4[0-7]`. Full decades next to each other share a
/// class.
pub fn range_regex(lo: u32, hi: u32) -> String {
    assert!(lo <= hi, "empty range {lo}..={hi}");
    let mut parts = Vec::new();
    let mut start = lo;
    let mut width = digits(lo);
    while start <= hi {
        // numbers of this width: up to 10^width - 1
        let top = 10u32.saturating_pow(width) - 1;
        let end = hi.min(top);
        parts.extend(same_width(&to_digits(start, width), &to_digits(end, width)));
        start = end + 1;
        width += 1;
    }
    parts.join("|")
}

fn digits(n: u32) -> u32 {
    n.checked_ilog10().unwrap_or(0) + 1
}

fn to_digits(n: u32, width: u32) -> Vec<u8> {
    format!("{n:0w$}", w = width as usize)
        .bytes()
        .map(|b| b - b'0')
        .collect()
}

fn class(a: u8, b: u8) -> String {
    if a == b {
        a.to_string()
    } else {
        format!("[{a}-{b}]")
    }
}

/// Patterns for the digit strings between `lo` and `hi` (same length, leading zeros allowed).
fn same_width(lo: &[u8], hi: &[u8]) -> Vec<String> {
    if lo.len() == 1 {
        return vec![class(lo[0], hi[0])];
    }
    let rest = lo.len() - 1;
    if lo[0] == hi[0] {
        return same_width(&lo[1..], &hi[1..])
            .into_iter()
            .map(|s| format!("{}{s}", lo[0]))
            .collect();
    }
    let zeros = vec![0u8; rest];
    let nines = vec![9u8; rest];
    let full_lo = lo[1..] == zeros[..];
    let full_hi = hi[1..] == nines[..];
    let mut out = Vec::new();
    if !full_lo {
        for s in same_width(&lo[1..], &nines) {
            out.push(format!("{}{s}", lo[0]));
        }
    }
    let mid_lo = if full_lo { lo[0] } else { lo[0] + 1 };
    let mid_hi = if full_hi { hi[0] } else { hi[0] - 1 };
    if mid_lo <= mid_hi {
        out.push(format!("{}{}", class(mid_lo, mid_hi), "[0-9]".repeat(rest)));
    }
    if !full_hi {
        for s in same_width(&zeros, &hi[1..]) {
            out.push(format!("{}{s}", hi[0]));
        }
    }
    out
}

/// The `-ot` pattern that puts the stacked experts of layers `from..n_layer` on the CPU.
pub fn cpu_experts_pattern(from: u32, n_layer: u32) -> String {
    if from == 0 {
        return "exps=CPU".to_string();
    }
    format!(
        r"blk\.({})\.ffn_(up|down|gate)_exps=CPU",
        range_regex(from, n_layer - 1)
    )
}

/// Arguments of a candidate without `-t`, `-p`, `-n`, `-r` and `-o`, which the bench harness
/// owns: `-ngl`, `-ot`, `-b`, `-ub`, `-nopo`, `-fa`.
///
/// MoE candidates use `-ngl 99` and move the experts of the later layers with one `-ot`
/// (llama-bench separates several patterns with `;`; one range needs only one). `-b` is
/// `max(2048, ub)`.
pub fn to_variant_args(c: &Candidate, layout: &ModelLayout) -> Vec<String> {
    let mut a = Vec::new();
    let mut push = |flag: &str, v: String| {
        a.push(flag.to_string());
        a.push(v);
    };
    match c.placement {
        Placement::Dense { gpu_layers } => {
            push("-ngl", dense_ngl(gpu_layers, layout.n_layer).to_string());
        }
        Placement::Moe { gpu_expert_layers } => {
            push("-ngl", "99".into());
            if gpu_expert_layers < layout.n_layer {
                push(
                    "-ot",
                    cpu_experts_pattern(gpu_expert_layers, layout.n_layer),
                );
            }
        }
    }
    let ub = c.ubatch();
    push("-b", ub.max(2048).to_string());
    push("-ub", ub.to_string());
    if !c.op_offload() {
        push("-nopo", "1".into());
    }
    push("-fa", "1".into());
    a
}

/// [`to_variant_args`] plus `-t`: a complete llama-bench command line apart from the model and
/// the test sizes.
pub fn to_llama_bench_args(c: &Candidate, layout: &ModelLayout) -> Vec<String> {
    let mut a = to_variant_args(c, layout);
    a.extend(["-t".to_string(), c.threads().to_string()]);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{Workload, build_candidate};
    use crate::layout::fixtures;

    /// Tiny matcher for the regex subset `range_regex` emits.
    fn matches(pattern: &str, n: u32) -> bool {
        let s = n.to_string();
        pattern.split('|').any(|alt| {
            let mut cls = Vec::new();
            let b = alt.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'[' {
                    cls.push((b[i + 1], b[i + 3]));
                    i += 5;
                } else {
                    cls.push((b[i], b[i]));
                    i += 1;
                }
            }
            cls.len() == s.len()
                && s.bytes()
                    .zip(&cls)
                    .all(|(d, &(lo, hi))| (lo..=hi).contains(&d))
        })
    }

    #[test]
    fn range_regex_known_forms() {
        assert_eq!(range_regex(24, 47), "2[4-9]|3[0-9]|4[0-7]");
        assert_eq!(range_regex(40, 63), "[4-5][0-9]|6[0-3]");
        assert_eq!(range_regex(7, 7), "7");
        assert_eq!(range_regex(3, 9), "[3-9]");
        assert_eq!(range_regex(8, 12), "[8-9]|1[0-2]");
        assert_eq!(range_regex(0, 47), "[0-9]|[1-3][0-9]|4[0-7]");
        assert_eq!(range_regex(98, 101), "9[8-9]|10[0-1]");
    }

    #[test]
    fn range_regex_is_exact() {
        for lo in 0..130 {
            for hi in lo..130 {
                let p = range_regex(lo, hi);
                for n in 0..1000 {
                    assert_eq!(
                        matches(&p, n),
                        (lo..=hi).contains(&n),
                        "{lo}..={hi} {p} {n}"
                    );
                }
            }
        }
    }

    #[test]
    fn moe_args() {
        let l = fixtures::moe();
        let w = Workload {
            prompt: 2048,
            r#gen: 128,
        };
        let c = build_candidate(
            "c00",
            &l,
            Placement::Moe {
                gpu_expert_layers: 24,
            },
            (15, 512, false),
            &w,
        );
        let a = to_llama_bench_args(&c, &l);
        let want = [
            "-ngl",
            "99",
            "-ot",
            r"blk\.(2[4-9]|3[0-9]|4[0-7])\.ffn_(up|down|gate)_exps=CPU",
            "-b",
            "2048",
            "-ub",
            "512",
            "-nopo",
            "1",
            "-fa",
            "1",
            "-t",
            "15",
        ];
        assert_eq!(a, want);
        let c = build_candidate(
            "c01",
            &l,
            Placement::Moe {
                gpu_expert_layers: 0,
            },
            (7, 2048, true),
            &w,
        );
        assert_eq!(
            to_variant_args(&c, &l),
            [
                "-ngl", "99", "-ot", "exps=CPU", "-b", "2048", "-ub", "2048", "-fa", "1"
            ]
        );
        let c = build_candidate(
            "c02",
            &l,
            Placement::Moe {
                gpu_expert_layers: 48,
            },
            (7, 512, true),
            &w,
        );
        assert!(!to_variant_args(&c, &l).contains(&"-ot".to_string()));
    }

    #[test]
    fn dense_args() {
        let l = fixtures::dense();
        let w = Workload {
            prompt: 2048,
            r#gen: 128,
        };
        let c = build_candidate(
            "c00",
            &l,
            Placement::Dense { gpu_layers: 40 },
            (15, 2048, true),
            &w,
        );
        assert_eq!(
            to_llama_bench_args(&c, &l),
            [
                "-ngl", "41", "-b", "2048", "-ub", "2048", "-fa", "1", "-t", "15"
            ]
        );
        assert_eq!(dense_ngl(64, 64), 65);
        assert_eq!(dense_ngl(0, 64), 0);
    }
}
