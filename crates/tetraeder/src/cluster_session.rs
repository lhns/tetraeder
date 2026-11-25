//! `tetraeder cluster merge` and `tetraeder cluster judged-round`: what a judged Case S session
//! runs on the head after `cluster profile` (ADR 0091, 0092).

use crate::node::ClusterArgs;
use anyhow::{Context, Result};
use clap::Args;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tetraeder::hw_merge;
use tetraeder::judged::{SegmentLabel, Workload, prompt_of_dump, run_round};
use tetraeder::pipeline::{
    PipelineBackend, plan_step_tokens, segment_assignments, start_remote_segments,
};
use tetraeder_control::client;
use tetraeder_scheduler::SegmentBackend;
use tetraeder_transport::{NodeId, SocketBuffers, Transport};

#[derive(Args)]
pub struct MergeArgs {
    /// The cluster graph (JSON, from `cluster profile`).
    pub hw: PathBuf,
    /// A node's fitted graph (`tetraeder-plan segment-predict --hw-out`), as `ID=FILE.toml`.
    #[arg(long = "node", value_name = "ID=FITTED.toml")]
    pub nodes: Vec<String>,
    /// Replace the graph's link curves and network links by this probe's, nothing else.
    #[arg(long)]
    pub swap_link: Option<PathBuf>,
    /// A node's measured RAM budget in bytes, as `ID=BYTES` (the CPU's capacity).
    #[arg(long = "ram-budget", value_name = "ID=BYTES")]
    pub ram_budget: Vec<String>,
    #[arg(short, long)]
    pub out: PathBuf,
}

#[derive(Args)]
pub struct JudgedArgs {
    #[command(flatten)]
    pub cluster: ClusterArgs,
    /// The cluster plan (TOML); its first replica runs, one segment per node.
    #[arg(long)]
    pub plan: PathBuf,
    /// The model's sha256: the nodes' models folders name the file `<sha256>.gguf`.
    #[arg(long)]
    pub model_sha256: String,
    /// The hidden-row dump of the frozen prompt (`TETRAEDER_EXIT_HIDDEN_OUT`); its prompt token
    /// ids are the request.
    #[arg(long)]
    pub hidden: PathBuf,
    /// Prompt tokens to use (default: the whole prompt of the dump).
    #[arg(long)]
    pub prompt_tokens: Option<usize>,
    /// Greedy tokens decoded after the prompt.
    #[arg(long = "gen", default_value_t = 128)]
    pub gen_tokens: usize,
    /// Untimed full requests before the timed one (ADR 0092 owner answer 1).
    #[arg(long, default_value_t = 1)]
    pub warm: usize,
    /// Seconds a worker may take to start (it loads its share of the model).
    #[arg(long, default_value_t = 1800)]
    pub startup_timeout_s: u64,
    /// Seconds a step may take before the worker is declared dead.
    #[arg(long, default_value_t = 600)]
    pub deadline_s: u64,
    /// Writes `cluster.json` and `seg-<node>.json` here.
    #[arg(long)]
    pub out: PathBuf,
}

/// `ID=BYTES` pairs.
pub fn id_values(specs: &[String]) -> Result<BTreeMap<String, u64>> {
    specs
        .iter()
        .map(|s| {
            let (id, v) = s
                .split_once('=')
                .with_context(|| format!("`{s}`: expected ID=BYTES"))?;
            Ok((
                id.to_owned(),
                v.parse::<u64>()
                    .with_context(|| format!("`{s}`: not a byte count"))?,
            ))
        })
        .collect()
}

fn read(p: &PathBuf) -> Result<String> {
    std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))
}

pub fn merge(a: &MergeArgs) -> Result<()> {
    let mut hw: serde_json::Value = serde_json::from_str(&read(&a.hw)?)
        .with_context(|| format!("{} is not a hardware graph", a.hw.display()))?;
    if !a.ram_budget.is_empty() {
        hw_merge::set_ram_budgets(&mut hw, &id_values(&a.ram_budget)?)
            .map_err(anyhow::Error::msg)?;
    }
    let mut fitted = BTreeMap::new();
    for spec in &a.nodes {
        let (n, f) = spec
            .split_once('=')
            .with_context(|| format!("`{spec}`: expected ID=FILE"))?;
        let text = read(&PathBuf::from(f))?;
        fitted.insert(
            n.to_owned(),
            hw_merge::read_fitted(&text).map_err(|e| anyhow::anyhow!("{f}: {e}"))?,
        );
    }
    for note in hw_merge::merge_fitted(&mut hw, &fitted).map_err(anyhow::Error::msg)? {
        println!("note: {note}");
    }
    if let Some(p) = &a.swap_link {
        let probe: serde_json::Value = serde_json::from_str(&read(p)?)
            .with_context(|| format!("{} is not a probe result", p.display()))?;
        hw_merge::swap_link(&mut hw, &probe).map_err(anyhow::Error::msg)?;
    }
    hw_merge::check_reads(&hw).map_err(anyhow::Error::msg)?;
    std::fs::write(&a.out, serde_json::to_string_pretty(&hw)?)
        .with_context(|| format!("write {}", a.out.display()))?;
    println!(
        "wrote {}: {} fitted node graph(s)",
        a.out.display(),
        fitted.len()
    );
    Ok(())
}

pub fn judged_round(a: &JudgedArgs) -> Result<()> {
    let (cfg, id) = a.cluster.load()?;
    anyhow::ensure!(
        id.node_id()? == cfg.head,
        "judged-round runs with the head's key and certificate (`{}`)",
        cfg.head
    );
    let t: Arc<dyn Transport> =
        Arc::new(a.cluster.transport(&cfg, &id, None, SocketBuffers::Auto)?);
    let plan = tetraeder_plan::Plan::from_toml(&read(&a.plan)?)
        .map_err(|e| anyhow::anyhow!("{} is not a plan: {e}", a.plan.display()))?;
    let mut prompt = prompt_of_dump(&a.hidden).map_err(anyhow::Error::msg)?;
    if let Some(n) = a.prompt_tokens {
        anyhow::ensure!(
            n <= prompt.len(),
            "the dump holds {} prompt tokens, {n} wanted",
            prompt.len()
        );
        prompt.truncate(n);
    }
    // The graph only maps the plan's devices to nodes.
    let hw = client::profile_cluster(t.as_ref(), &cfg, None).map_err(anyhow::Error::msg)?;
    let segs = segment_assignments(&plan, 0, &hw, &cfg).map_err(anyhow::Error::msg)?;
    let nodes: Vec<String> = segs.iter().map(|s| s.peer.id.0.clone()).collect();
    let distinct: std::collections::BTreeSet<&String> = nodes.iter().collect();
    anyhow::ensure!(
        distinct.len() == nodes.len(),
        "the judged plan has {} segments on {} nodes: Case S runs one segment per node",
        nodes.len(),
        distinct.len()
    );
    let node_ids: Vec<NodeId> = segs.iter().map(|s| s.peer.id.clone()).collect();
    let unreached = client::check_model(t.as_ref(), &cfg, &node_ids, &a.model_sha256)
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        unreached.is_empty(),
        "cannot reach node(s) {unreached:?} to check the model"
    );
    let remote = start_remote_segments(&t, 0, &segs, &a.model_sha256, &|c| {
        c.startup_timeout = Duration::from_secs(a.startup_timeout_s);
        c.deadline_floor = Duration::from_secs(a.deadline_s);
    })
    .map_err(anyhow::Error::msg)?;
    let mut pipe = PipelineBackend::new(
        remote
            .into_iter()
            .map(|r| Box::new(r) as Box<dyn SegmentBackend>)
            .collect(),
    )
    .map_err(anyhow::Error::msg)?;
    pipe.set_prefill_micro_batches(plan.knobs.prefill_micro_batches);
    let step_tokens = plan_step_tokens(&plan, None).unwrap_or(prompt.len());
    let labels: Vec<SegmentLabel> = nodes
        .into_iter()
        .map(|node| SegmentLabel { node })
        .collect();
    let w = Workload {
        prompt: &prompt,
        gen_tokens: a.gen_tokens,
        step_tokens,
        warm: a.warm,
    };
    let files = run_round(&mut pipe, &labels, &w).map_err(anyhow::Error::msg)?;
    std::fs::create_dir_all(&a.out).with_context(|| format!("create {}", a.out.display()))?;
    let put = |name: String, v: &serde_json::Value| -> Result<()> {
        let p = a.out.join(name);
        std::fs::write(&p, serde_json::to_string_pretty(v)?)
            .with_context(|| format!("write {}", p.display()))
    };
    put("cluster.json".into(), &files.cluster)?;
    for (node, v) in &files.segments {
        put(format!("seg-{node}.json"), v)?;
    }
    let r = &files.cluster["runs"][0];
    println!(
        "judged round: pp {:.1} tok/s, tg {:.2} tok/s, {} segments, written to {}",
        r["prefill_tok_s"].as_f64().unwrap_or(0.0),
        r["decode_tok_s"].as_f64().unwrap_or(0.0),
        files.segments.len(),
        a.out.display()
    );
    Ok(())
}
