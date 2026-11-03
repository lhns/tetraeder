//! `tetraeder advise`: ranked improvement actions for one observation (ADR 0041). Report only;
//! nothing is applied. The policy class says how a controller would treat each action.

use crate::doctor::{read_observation, snake};
use anyhow::{Context, Result};
use clap::Args;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use tetraeder_analysis::advise::{Action, advise};
use tetraeder_analysis::detect::detect;
use tetraeder_analysis::observation::Observation;
use tetraeder_plan::Hints;

#[derive(Args)]
pub struct AdviseArgs {
    /// Observation JSON (see crates/analysis/fixtures/observation_example.json).
    #[arg(long)]
    observation: PathBuf,
    /// Action kinds that may be applied without asking, comma separated (`hints.auto_apply`).
    #[arg(long, value_delimiter = ',')]
    auto_apply: Vec<String>,
    /// Hints TOML; its `auto_apply` is added to `--auto-apply`.
    #[arg(long)]
    hints: Option<PathBuf>,
    /// Print the actions as JSON.
    #[arg(long)]
    json: bool,
}

pub fn run(a: &AdviseArgs) -> Result<()> {
    let obs = read_observation(&a.observation)?;
    let mut auto = a.auto_apply.clone();
    if let Some(p) = &a.hints {
        auto.extend(hints_auto_apply(p)?);
    }
    print!("{}", output(&obs, &auto, a.json)?);
    Ok(())
}

fn hints_auto_apply(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let hints =
        Hints::from_toml(&text).with_context(|| format!("parse hints {}", path.display()))?;
    Ok(hints.auto_apply)
}

/// The ranked actions as text or JSON.
pub fn output(obs: &Observation, auto_apply: &[String], json: bool) -> Result<String> {
    let actions = advise(&detect(obs), obs, auto_apply);
    if json {
        Ok(serde_json::to_string_pretty(&actions)? + "\n")
    } else {
        Ok(text(&actions))
    }
}

fn text(actions: &[Action]) -> String {
    let mut o = String::from("actions, best first (report only, nothing is applied):\n");
    if actions.is_empty() {
        o.push_str("  none\n");
    }
    for (i, a) in actions.iter().enumerate() {
        let _ = writeln!(
            o,
            "  {}. [{}] {} on {}: gain {:.1}% of the step, switch cost {:.1} s, score {:.1} s",
            i + 1,
            snake(&a.policy),
            snake(&a.kind),
            a.target,
            a.predicted_gain_share * 100.0,
            a.switch_cost_s,
            a.score()
        );
        let _ = writeln!(o, "       {}", a.why);
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::tests::{example, faulty};
    use tetraeder_analysis::advise::PolicyClass;

    // hardware: none
    #[test]
    fn faulty_observation_gets_an_environment_fix() {
        let out = output(&faulty(), &[], false).unwrap();
        assert!(out.contains("fix_environment"), "{out}");
        assert!(out.contains("[report]"), "{out}");
        assert!(out.contains("report only"), "{out}");
    }

    // hardware: none
    #[test]
    fn json_lists_actions_best_first() {
        let out = output(&faulty(), &[], true).unwrap();
        let v: Vec<Action> = serde_json::from_str(&out).unwrap();
        assert!(!v.is_empty());
        assert!(v.windows(2).all(|w| w[0].score() >= w[1].score()));
        assert!(v.iter().any(|a| a.policy == PolicyClass::Report));
    }

    // hardware: none
    #[test]
    fn example_runs() {
        let out = output(&example(), &[], false).unwrap();
        assert!(out.starts_with("actions, best first"), "{out}");
    }

    // hardware: none
    #[test]
    fn hints_file_supplies_auto_apply() {
        let dir = std::env::temp_dir().join(format!("tetraeder-hints-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("hints.toml");
        std::fs::write(&p, "auto_apply = [\"grow_hot_set\", \"ubatch\"]\n").unwrap();
        assert_eq!(hints_auto_apply(&p).unwrap(), ["grow_hot_set", "ubatch"]);
        std::fs::write(&p, "auto_apply = 3\n").unwrap();
        assert!(hints_auto_apply(&p).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
