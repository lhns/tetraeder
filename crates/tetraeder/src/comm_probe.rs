//! `tetraeder comm-probe`: times the executor's all-reduces and copies between GPUs (ADR 0069,
//! brief E5). The planner's collective curves (ADR 0070) come from its output.
//!
//! Every group the ADR 0068 candidates use gets an all-reduce curve: all GPUs and, on four, each
//! pair {0,1} and {2,3} (positions in `--gpus`). Every ordered pair gets a copy curve. Sizes run
//! from 4 KiB to 64 MiB in powers of 4; each point is the median of the repeats, each timed op
//! between two `Backend::synchronize` calls, through the path the executor uses.

use anyhow::{Result, bail};
use clap::Args;
use serde::Serialize;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;
use tetraeder_executor::{Executor, Op, RankDevice, RankId, Ranks};
use tetraeder_ggml_sys::ggml::{Context, OwnedTensor, Type};

#[derive(Args)]
pub struct CommProbeArgs {
    /// GPU indexes, comma separated (at least two, for example 0,1 or 0,1,2,3).
    #[arg(long, value_delimiter = ',', required = true)]
    gpus: Vec<usize>,
    /// Output JSON file.
    #[arg(long)]
    out: PathBuf,
    /// Timed repeats per size; each point is their median.
    #[arg(long, default_value_t = 20)]
    repeats: usize,
}

/// 4 KiB to 64 MiB in powers of 4.
pub const SIZES: [usize; 8] = [
    4 << 10,
    16 << 10,
    64 << 10,
    256 << 10,
    1 << 20,
    4 << 20,
    16 << 20,
    64 << 20,
];

/// Field names and variants as the planner's `CollectiveCurve` (brief E3).
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveOp {
    AllReduce,
    Copy,
}

#[derive(Serialize, Debug, Clone)]
pub struct Curve {
    pub op: CollectiveOp,
    /// Device names; a set for an all-reduce, an ordered (source, destination) pair for a copy.
    pub ranks: Vec<String>,
    /// The path the executor took: `nccl`, `internal` or `fallback` for an all-reduce, `copy`
    /// for a copy.
    pub comm: String,
    /// (bytes, seconds).
    pub points: Vec<(f64, f64)>,
}

#[derive(Serialize, Debug, Clone)]
pub struct Report {
    /// The path of an all-reduce over all ranks.
    pub comm_path: String,
    pub devices: Vec<String>,
    pub repeats: usize,
    pub curves: Vec<Curve>,
}

pub fn run(a: &CommProbeArgs) -> Result<()> {
    if a.gpus.len() < 2 {
        bail!("--gpus needs at least two GPUs");
    }
    let mut sorted = a.gpus.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        bail!("--gpus names a GPU twice");
    }
    let devices: Vec<RankDevice> = a.gpus.iter().map(|&i| RankDevice::Gpu(i)).collect();
    let names: Vec<String> = a.gpus.iter().map(|i| format!("gpu{i}")).collect();
    let report = probe(&devices, &names, a.repeats, &SIZES)?;
    for c in &report.curves {
        let last = c.points.last().map_or(0.0, |p| p.1);
        println!(
            "{:?} {} via {}: {} points, {:.3} ms at the largest size",
            c.op,
            c.ranks.join(","),
            c.comm,
            c.points.len(),
            last * 1e3
        );
    }
    std::fs::write(&a.out, serde_json::to_string_pretty(&report)?)?;
    println!(
        "wrote {} (all-reduce path {})",
        a.out.display(),
        report.comm_path
    );
    Ok(())
}

/// Times the curves of `devices` (one rank each, named `names`).
pub fn probe(
    devices: &[RankDevice],
    names: &[String],
    repeats: usize,
    sizes: &[usize],
) -> Result<Report> {
    let n = devices.len();
    let mut exec = Executor::new(Ranks::open(devices)?);
    let repeats = repeats.max(1);
    let mut groups: Vec<Vec<RankId>> = vec![(0..n).collect()];
    if n == 4 {
        groups.push(vec![0, 1]);
        groups.push(vec![2, 3]);
    }
    let mut curves = Vec::new();
    for g in &groups {
        let mut points = Vec::new();
        let mut comm = String::from("fallback");
        exec.take_counters();
        for &bytes in sizes {
            let bufs = buffers(&exec, g, bytes)?;
            let op = Op::AllReduce {
                tensors: g.iter().zip(&bufs).map(|(&r, t)| (r, t.get())).collect(),
            };
            points.push((bytes as f64, time_op(&mut exec, &op, g, repeats)?));
        }
        if let Some(&(_, path)) = exec
            .take_counters()
            .collective_bytes
            .keys()
            .find(|k| k.0 == "all_reduce")
        {
            comm = path.to_string();
        }
        curves.push(Curve {
            op: CollectiveOp::AllReduce,
            ranks: g.iter().map(|&r| names[r].clone()).collect(),
            comm,
            points,
        });
    }
    for src in 0..n {
        for dst in (0..n).filter(|&d| d != src) {
            let mut points = Vec::new();
            for &bytes in sizes {
                let pair = [src, dst];
                let bufs = buffers(&exec, &pair, bytes)?;
                let op = Op::Copy {
                    src: (src, bufs[0].get()),
                    dst: (dst, bufs[1].get()),
                };
                points.push((bytes as f64, time_op(&mut exec, &op, &pair, repeats)?));
            }
            curves.push(Curve {
                op: CollectiveOp::Copy,
                ranks: vec![names[src].clone(), names[dst].clone()],
                comm: "copy".to_string(),
                points,
            });
        }
    }
    Ok(Report {
        comm_path: exec.ranks().comm_path().to_string(),
        devices: names.to_vec(),
        repeats,
        curves,
    })
}

/// One zeroed F32 tensor of `bytes` bytes on each rank of `group`, each holding its own buffer.
/// Zeros keep repeated all-reduces from overflowing.
fn buffers(exec: &Executor, group: &[RankId], bytes: usize) -> Result<Vec<OwnedTensor>> {
    group
        .iter()
        .map(|&r| {
            let ctx = Rc::new(Context::new(1, 0)?);
            let t = ctx.new_persistent(Type::F32, &[(bytes / 4) as i64]);
            ctx.alloc_tensors(exec.ranks().backend(r))?;
            ctx.clear_buffer(0);
            Ok(t)
        })
        .collect()
}

fn time_op(exec: &mut Executor, op: &Op<'_>, ranks: &[RankId], repeats: usize) -> Result<f64> {
    let sync = |exec: &Executor| {
        for &r in ranks {
            exec.ranks().backend(r).synchronize();
        }
    };
    for _ in 0..2 {
        exec.transfer(op)?;
        sync(exec);
    }
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        sync(exec);
        let t0 = Instant::now();
        exec.transfer(op)?;
        sync(exec);
        times.push(t0.elapsed().as_secs_f64());
    }
    Ok(median(&mut times))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        args: CommProbeArgs,
    }

    #[test]
    fn parses_gpus() {
        let w = Wrapper::try_parse_from(["x", "--gpus", "0,1,2,3", "--out", "o.json"]).unwrap();
        assert_eq!((w.args.gpus, w.args.repeats), (vec![0, 1, 2, 3], 20));
        assert!(Wrapper::try_parse_from(["x", "--out", "o.json"]).is_err());
    }

    #[test]
    fn median_of_odd_and_even() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    fn cpu(n: usize) -> (Vec<RankDevice>, Vec<String>) {
        (
            vec![RankDevice::Cpu { threads: 1 }; n],
            (0..n).map(|i| format!("cpu{i}")).collect(),
        )
    }

    /// CPU ranks stand in for GPUs: every curve has a point per size, growing in bytes, with
    /// finite positive times, and the all-reduce takes the fallback.
    #[test]
    fn probe_on_two_cpu_ranks() {
        let (d, names) = cpu(2);
        let r = probe(&d, &names, 2, &[4 << 10, 16 << 10]).unwrap();
        assert_eq!(r.comm_path, "fallback");
        assert_eq!(r.curves.len(), 3);
        let ar = &r.curves[0];
        assert_eq!(
            (ar.op, ar.comm.as_str()),
            (CollectiveOp::AllReduce, "fallback")
        );
        assert_eq!(ar.ranks, ["cpu0", "cpu1"]);
        let copies: Vec<_> = r.curves[1..].iter().map(|c| c.ranks.join(">")).collect();
        assert_eq!(copies, ["cpu0>cpu1", "cpu1>cpu0"]);
        for c in &r.curves {
            let bytes: Vec<f64> = c.points.iter().map(|p| p.0).collect();
            assert_eq!(bytes, [4096.0, 16384.0]);
            assert!(
                c.points.iter().all(|p| p.1.is_finite() && p.1 > 0.0),
                "{c:?}"
            );
        }
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""op":"AllReduce""#) && json.contains(r#""comm_path":"fallback""#));
    }

    /// Four ranks: all of them and each pair for the all-reduce, every ordered pair for copies.
    #[test]
    fn probe_on_four_cpu_ranks_adds_the_pairs() {
        let (d, names) = cpu(4);
        let r = probe(&d, &names, 1, &[4 << 10]).unwrap();
        let ar: Vec<_> = r
            .curves
            .iter()
            .filter(|c| c.op == CollectiveOp::AllReduce)
            .map(|c| c.ranks.join(","))
            .collect();
        assert_eq!(ar, ["cpu0,cpu1,cpu2,cpu3", "cpu0,cpu1", "cpu2,cpu3"]);
        let copies = r.curves.iter().filter(|c| c.op == CollectiveOp::Copy);
        assert_eq!(copies.count(), 12);
    }
}
