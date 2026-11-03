//! Phase 3a exit test, quality part (docs/plan.md row 3a, ADR 0040 "Quality", ADR 0048).
//! Hardware: none. Reads the committed profiles and the committed `compare_mix.py` results.
//!
//! At the byte budget of the llama.cpp preset Q4_K_M the planner's format mix must be the mix
//! `compare_mix.py` measured as `greedy-at-Q4_K_M` (so that row's measured KL applies) and that
//! KL must be no higher than the preset's. The summed estimate, corrected, must stay within the
//! stated band of the measured KL of the profile's checks.

use std::collections::BTreeMap;
use tetraeder_plan::{Format, QualityProfile};
use tetraeder_planner::{Band, Correction, cli, estimate_mix, group_of_tensor, mix_at_budget};

struct Case {
    name: &'static str,
    moe: bool,
    profile: &'static str,
    result: &'static str,
    /// Stated band: largest `|measured - corrected| / measured` over the checks (in-sample).
    band: f64,
}

const CASES: [Case; 2] = [
    Case {
        name: "qwen3-8b",
        moe: false,
        profile: include_str!("../../../bench/quality/qwen3-8b.quality.json"),
        result: include_str!("../../../bench/results/20261103/quality-qwen3-8b.json"),
        band: 0.20,
    },
    Case {
        name: "qwen3-30b",
        moe: true,
        profile: include_str!("../../../bench/quality/qwen3-30b.quality.json"),
        result: include_str!("../../../bench/results/20261103/quality-qwen3-30b.json"),
        band: 0.31,
    },
];

fn row<'a>(res: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    res["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == name)
        .unwrap_or_else(|| panic!("no row {name}"))
}

#[test]
fn planner_mix_at_q4_k_m_bytes_is_the_measured_greedy_mix_and_beats_the_preset() {
    for c in &CASES {
        let profile = QualityProfile::from_json(c.profile).unwrap();
        let res: serde_json::Value = serde_json::from_str(c.result).unwrap();
        let (preset, greedy) = (row(&res, "preset-Q4_K_M"), row(&res, "greedy-at-Q4_K_M"));
        let budget = preset["bytes"].as_u64().unwrap();

        let mix = mix_at_budget(&profile, budget).unwrap();
        assert!(mix.bytes <= budget, "{}: {} > {budget}", c.name, mix.bytes);

        // The same formats, group by group, as the mix that was measured.
        let measured: BTreeMap<String, String> =
            serde_json::from_value(greedy["formats"].clone()).unwrap();
        let ours: BTreeMap<String, String> = mix
            .formats
            .iter()
            .map(|(g, f)| {
                let name = serde_json::to_value(f).unwrap();
                (g.clone(), name.as_str().unwrap().to_string())
            })
            .collect();
        assert_eq!(
            ours, measured,
            "{}: planner mix differs from greedy",
            c.name
        );
        let est = greedy["estimate"].as_f64().unwrap();
        assert!((mix.estimate - est).abs() < 1e-9, "{}", c.name);

        // So the measured row applies: no more bytes and no higher KL than the preset.
        assert!(greedy["bytes"].as_u64().unwrap() <= budget, "{}", c.name);
        let kl_mix = greedy["kl"].as_f64().unwrap();
        let kl_preset = preset["kl"].as_f64().unwrap();
        assert!(kl_mix <= kl_preset, "{}: {kl_mix} > {kl_preset}", c.name);
        eprintln!(
            "{}: planner {} B <= preset {budget} B; measured KL {kl_mix:.5} <= preset \
             {kl_preset:.5}; estimate {:.5}, corrected {:.5}",
            c.name, mix.bytes, mix.estimate, mix.corrected
        );
    }
}

#[test]
fn checks_stay_within_the_stated_band() {
    for c in &CASES {
        let p = QualityProfile::from_json(c.profile).unwrap();
        assert!(!p.checks.is_empty(), "{}", c.name);
        let corr = Correction::from_profile(&p);
        assert!(corr.b > 0.0, "{}: not monotone", c.name);
        for k in &p.checks {
            // The recorded estimate is the ADR 0048 sum.
            let e = estimate_mix(&p, &k.formats).unwrap();
            assert!((e - k.estimate).abs() < 1e-5, "{} {}", c.name, k.name);
            let err = (k.measured - corr.apply(e)).abs() / k.measured;
            assert!(
                err <= c.band,
                "{} {}: {err:.3} > {}",
                c.name,
                k.name,
                c.band
            );
        }
        let band = Band::of(&p);
        assert!(band.corrected <= c.band, "{}: {}", c.name, band.corrected);
        // The stated band is tight (within 0.02 of the computed one) and beats the raw sum.
        assert!(
            band.corrected > c.band - 0.02,
            "{}: {}",
            c.name,
            band.corrected
        );
        assert!(band.corrected < band.raw, "{}", c.name);
        eprintln!(
            "{}: band corrected {:.3}, raw {:.3}, leave-one-out {:?}; a={:.4} b={:.4}",
            c.name, band.corrected, band.raw, band.leave_one_out, corr.a, corr.b
        );
    }
}

#[test]
fn corrected_estimate_of_the_greedy_rows_is_reported() {
    // Not asserted: the greedy rows are held-out text and structured mixes; see ADR 0040.
    for c in &CASES {
        let p = QualityProfile::from_json(c.profile).unwrap();
        let corr = Correction::from_profile(&p);
        let res: serde_json::Value = serde_json::from_str(c.result).unwrap();
        for r in res["rows"].as_array().unwrap() {
            if let Some(est) = r.get("estimate").and_then(|e| e.as_f64()) {
                let m = r["kl"].as_f64().unwrap();
                eprintln!(
                    "{} {}: measured {m:.5}, raw {:+.0}%, corrected {:+.0}%",
                    c.name,
                    r["name"],
                    (est - m) / m * 100.0,
                    (corr.apply(est) - m) / m * 100.0
                );
            }
        }
    }
}

/// llama.cpp tensor names of a Qwen3 model with `layers` layers.
fn qwen3_tensors(layers: u32, moe: bool) -> Vec<String> {
    let mut v: Vec<String> = ["token_embd", "output_norm", "output"]
        .iter()
        .map(|t| format!("{t}.weight"))
        .collect();
    let ffn: &[&str] = if moe {
        &[
            "ffn_gate_inp",
            "ffn_gate_exps",
            "ffn_up_exps",
            "ffn_down_exps",
        ]
    } else {
        &["ffn_gate", "ffn_up", "ffn_down"]
    };
    let attn = [
        "attn_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_q_norm",
    ];
    for l in 0..layers {
        for t in attn.iter().chain(ffn) {
            v.push(format!("blk.{l}.{t}.weight"));
        }
    }
    v
}

#[test]
fn cli_mix_writes_the_compare_mix_shape() {
    for c in &CASES {
        let p = QualityProfile::from_json(c.profile).unwrap();
        let list = serde_json::to_string(&qwen3_tensors(p.model.layers, c.moe)).unwrap();
        let res: serde_json::Value = serde_json::from_str(c.result).unwrap();
        let budget = row(&res, "preset-Q4_K_M")["bytes"].as_u64().unwrap();
        let mix = mix_at_budget(&p, budget).unwrap();

        let (out, summary) = cli::mix_cmd(c.profile, budget, Some(&list)).unwrap();
        let by_tensor: BTreeMap<String, Format> = serde_json::from_str(&out).unwrap();
        let group = |t: &str| group_of_tensor(t, p.model.layers, p.layer_span);
        // Every tensor has its group's format; norms and the router are left out.
        for (t, f) in &by_tensor {
            assert_eq!(Some(f), group(t).and_then(|g| mix.formats.get(&g)), "{t}");
        }
        // Every group is reached by at least one tensor, so the names match the profile.
        for g in mix.formats.keys() {
            let hit = by_tensor.keys().any(|t| group(t).as_ref() == Some(g));
            assert!(hit, "{}: no tensor in {g}", c.name);
        }
        assert!(
            !by_tensor
                .keys()
                .any(|t| t.contains("norm") || t.contains("inp"))
        );
        assert!(summary.contains("corrected"));

        // Without a tensor list the output is the group mix.
        let (out, _) = cli::mix_cmd(c.profile, budget, None).unwrap();
        let by_group: BTreeMap<String, Format> = serde_json::from_str(&out).unwrap();
        assert_eq!(by_group, mix.formats);
    }
}
