use crate::fallback::Scratch;
use crate::ranks::{comm_name, group_comm};
use crate::trace::{ExecOptions, Trace};
use crate::{ExecError, Op, Program, RankGraph, RankId, Ranks, validate};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;
use tetraeder_ggml_sys::ggml::{Comm, Gallocr, RangeGraphs, Tensor, Type};
use tetraeder_telemetry::Metrics;

/// What the executor did since the last [`Executor::take_counters`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecCounters {
    /// Ranges launched, per rank.
    pub ranges: Vec<u64>,
    /// Bytes by (`kind`, `comm`): `kind` is `all_reduce` (the tensor bytes of every rank of the
    /// group) or `copy`; `comm` is `nccl`, `internal` or `fallback` for an all-reduce, and the
    /// all-reduce path over all ranks for a copy.
    pub collective_bytes: BTreeMap<(&'static str, &'static str), u64>,
    /// Graph allocations and fallback scratch sets built (zero when a program is run again).
    pub allocations: u64,
}

impl ExecCounters {
    /// Adds these counts to the `tetraeder.executor.*` metrics (ADR 0069).
    pub fn record(&self, metrics: &Metrics) {
        for (rank, &n) in self.ranges.iter().enumerate() {
            metrics.executor_ranges(rank, n);
        }
        for (&(kind, comm), &bytes) in &self.collective_bytes {
            metrics.executor_collective_bytes(kind, comm, bytes);
        }
    }
}

/// Cached range graphs of one rank: the graph id and cuts they were cut for.
struct RankRanges {
    graph: u64,
    cuts: Vec<usize>,
    ranges: RangeGraphs,
}

/// Issues step programs on a set of ranks from the calling thread.
pub struct Executor {
    ranks: Ranks,
    gallocs: Vec<Option<Gallocr>>,
    ranges: Vec<Option<RankRanges>>,
    /// Comms over groups other than all ranks in rank order, by group; None: ggml cannot.
    group_comms: HashMap<Vec<RankId>, Option<Comm>>,
    /// Fallback scratch by group and tensor shape.
    scratch: HashMap<(Vec<RankId>, [i64; 4]), Scratch>,
    counters: ExecCounters,
    opts: ExecOptions,
    trace: Option<Trace>,
}

impl Executor {
    /// An executor with the options of the environment ([`ExecOptions::from_env`]).
    pub fn new(ranks: Ranks) -> Executor {
        Executor::with_options(ranks, ExecOptions::from_env())
    }

    pub fn with_options(ranks: Ranks, opts: ExecOptions) -> Executor {
        let n = ranks.len();
        let trace = Trace::open(&opts);
        Executor {
            opts,
            trace,
            ranks,
            gallocs: (0..n).map(|_| None).collect(),
            ranges: (0..n).map(|_| None).collect(),
            group_comms: HashMap::new(),
            scratch: HashMap::new(),
            counters: ExecCounters {
                ranges: vec![0; n],
                ..ExecCounters::default()
            },
        }
    }

    pub fn ranks(&self) -> &Ranks {
        &self.ranks
    }

    /// True when builders should upload step inputs on each rank's stream
    /// (`Tensor::set_*_async`) rather than with one blocking copy per tensor.
    pub fn async_inputs(&self) -> bool {
        self.opts.async_inputs
    }

    /// Adds a builder record (input upload, logits read) to the current step's trace line, with
    /// the host nanoseconds since `since`. No-op without a trace.
    pub fn trace_note(&mut self, kind: &'static str, rank: RankId, since: Instant) {
        if let Some(t) = &mut self.trace {
            t.note(kind, rank, None, elapsed_ns(since));
        }
    }

    /// Writes the current step's trace line (the executor's records since the last flush and the
    /// builder's notes). No-op without a trace.
    pub fn trace_flush(&mut self) {
        if let Some(t) = &mut self.trace {
            t.flush();
        }
    }

    /// Allocates rank graphs not allocated yet and cuts their ranges. Both are kept while the
    /// same graphs come back (as `Sched`), so a repeated program allocates nothing and its range
    /// graphs keep their tensors' addresses (CUDA graphs replay).
    pub fn prepare(&mut self, p: &Program<'_>) -> Result<(), ExecError> {
        self.check_graphs(p)?;
        // Boundary tensors must survive their range: the allocator may not reuse their memory.
        for op in &p.ops {
            match op {
                Op::Compute { .. } => {}
                Op::AllReduce { tensors } => {
                    for &(_, t) in tensors {
                        t.set_output();
                    }
                }
                Op::Copy { src, dst } => {
                    src.1.set_output();
                    dst.1.set_input();
                }
            }
        }
        for rg in &p.graphs {
            let r = rg.rank;
            if self.gallocs[r].is_none() {
                self.gallocs[r] = Some(Gallocr::new(self.ranks.backend(r))?);
            }
            let mut allocated = false;
            if let Some(galloc) = self.gallocs[r].as_mut()
                && !galloc.is_allocated(&rg.graph)
            {
                galloc.alloc_graph(&rg.graph)?;
                self.counters.allocations += 1;
                allocated = true;
            }
            // A new allocation moves tensors: ranges with a uid promise unchanged nodes and
            // sources, so they are cut again with new uids.
            let fresh = allocated
                || !matches!(&self.ranges[r],
                Some(c) if c.graph == rg.graph.id() && c.cuts == rg.cuts);
            if fresh {
                self.ranges[r] = Some(RankRanges {
                    graph: rg.graph.id(),
                    cuts: rg.cuts.clone(),
                    ranges: RangeGraphs::new(&rg.graph, &rg.cuts, self.opts.range_uid)?,
                });
            }
        }
        for op in &p.ops {
            if let Op::AllReduce { tensors } = op {
                self.prepare_group(tensors)?;
            }
        }
        Ok(())
    }

    /// Validates (debug builds), issues every op in order and synchronizes the output ranks.
    /// After an error every rank is synchronized, so no work of the failed step is left queued.
    pub fn run(&mut self, p: &Program<'_>) -> Result<(), ExecError> {
        if cfg!(debug_assertions) {
            validate(p, self.ranks.len()).map_err(ExecError::Invalid)?;
        }
        self.prepare(p)?;
        let res = self.issue(p);
        if res.is_err() {
            for r in 0..self.ranks.len() {
                self.ranks.backend(r).synchronize();
            }
        }
        res?;
        for &r in &p.outputs {
            let t0 = Instant::now();
            self.ranks.backend(r).synchronize();
            if let Some(t) = &mut self.trace {
                t.note("sync", r, None, elapsed_ns(t0));
            }
        }
        Ok(())
    }

    /// Issues one all-reduce or copy outside a program (the comm probe, tests); same path and
    /// counters as in a program. Does not wait. A compute op needs a program.
    pub fn transfer(&mut self, op: &Op<'_>) -> Result<(), ExecError> {
        match op {
            Op::Compute { .. } => Err(ExecError::invalid("transfer: a compute op needs a program")),
            Op::AllReduce { tensors } => {
                self.prepare_group(tensors)?;
                self.all_reduce(tensors)
            }
            Op::Copy { src, dst } => self.copy(*src, *dst),
        }
    }

    /// Bytes of the rank's graph buffer (0 before its first graph).
    pub fn buffer_size(&self, rank: RankId) -> usize {
        self.gallocs
            .get(rank)
            .and_then(Option::as_ref)
            .map_or(0, Gallocr::buffer_size)
    }

    pub fn take_counters(&mut self) -> ExecCounters {
        let n = self.ranks.len();
        std::mem::replace(
            &mut self.counters,
            ExecCounters {
                ranges: vec![0; n],
                ..ExecCounters::default()
            },
        )
    }

    fn check_graphs(&self, p: &Program<'_>) -> Result<(), ExecError> {
        let mut seen = vec![false; self.ranks.len()];
        for g in &p.graphs {
            if g.rank >= seen.len() {
                return Err(ExecError::invalid(format!(
                    "graph for rank {}, but there are {} ranks",
                    g.rank,
                    seen.len()
                )));
            }
            if std::mem::replace(&mut seen[g.rank], true) {
                return Err(ExecError::invalid(format!(
                    "rank {} has more than one graph",
                    g.rank
                )));
            }
        }
        Ok(())
    }

    fn issue(&mut self, p: &Program<'_>) -> Result<(), ExecError> {
        let mut graph_of: Vec<Option<&RankGraph<'_>>> = vec![None; self.ranks.len()];
        for g in &p.graphs {
            graph_of[g.rank] = Some(g);
        }
        for op in &p.ops {
            let t0 = self.trace.is_some().then(Instant::now);
            match op {
                Op::Compute { rank, range } => {
                    let cached = self.ranges.get(*rank).and_then(Option::as_ref);
                    let (Some(g), Some(c)) = (graph_of.get(*rank).copied().flatten(), cached)
                    else {
                        return Err(ExecError::invalid(format!(
                            "compute on rank {rank}, which has no graph"
                        )));
                    };
                    let part = c.ranges.get(&g.graph, *range)?;
                    self.ranks.backend(*rank).compute_async(&part)?;
                    self.counters.ranges[*rank] += 1;
                }
                Op::AllReduce { tensors } => self.all_reduce(tensors)?,
                Op::Copy { src, dst } => self.copy(*src, *dst)?,
            }
            if let (Some(t0), Some(t)) = (t0, &mut self.trace) {
                // Sync mode: wait for the op's ranks, so the record holds its device time.
                let (kind, rank, peer, waits) = match op {
                    Op::Compute { rank, .. } => ("compute", *rank, None, vec![*rank]),
                    Op::AllReduce { tensors } => (
                        "all_reduce",
                        tensors[0].0,
                        None,
                        tensors.iter().map(|&(r, _)| r).collect(),
                    ),
                    Op::Copy { src, dst } => ("copy", src.0, Some(dst.0), vec![src.0, dst.0]),
                };
                if t.sync {
                    for r in waits {
                        self.ranks.backend(r).synchronize();
                    }
                }
                t.note(kind, rank, peer, elapsed_ns(t0));
            }
        }
        Ok(())
    }

    /// Makes the comm for the group's ranks, or its fallback scratch, so `all_reduce` allocates
    /// nothing in the step.
    fn prepare_group(&mut self, tensors: &[(RankId, Tensor<'_>)]) -> Result<(), ExecError> {
        let group: Vec<RankId> = tensors.iter().map(|&(r, _)| r).collect();
        if group.len() < 2 || group.iter().any(|&r| r >= self.ranks.len()) {
            return Err(ExecError::invalid(format!(
                "all-reduce over ranks {group:?} of {}",
                self.ranks.len()
            )));
        }
        if !is_all_ranks(&group, self.ranks.len()) && !self.group_comms.contains_key(&group) {
            let backends: Vec<_> = group
                .iter()
                .map(|&r| self.ranks.backend(r).clone())
                .collect();
            self.group_comms
                .insert(group.clone(), group_comm(&backends));
        }
        if find_comm(&self.ranks, &self.group_comms, &group).is_none() {
            self.scratch_for(&group, tensors[0].1)?;
        }
        Ok(())
    }

    fn scratch_for(&mut self, group: &[RankId], like: Tensor<'_>) -> Result<(), ExecError> {
        if like.ty() != Type::F32 {
            return Err(ExecError::invalid(
                "the fallback all-reduce supports F32 tensors only",
            ));
        }
        let key = (group.to_vec(), like.ne());
        if !self.scratch.contains_key(&key) {
            let s = Scratch::new(&self.ranks, group, like)?;
            self.scratch.insert(key, s);
            self.counters.allocations += 1;
        }
        Ok(())
    }

    fn all_reduce(&mut self, tensors: &[(RankId, Tensor<'_>)]) -> Result<(), ExecError> {
        let group: Vec<RankId> = tensors.iter().map(|&(r, _)| r).collect();
        let ts: Vec<Tensor<'_>> = tensors.iter().map(|&(_, t)| t).collect();
        let bytes: u64 = ts.iter().map(|t| t.nbytes() as u64).sum();
        let via_comm = find_comm(&self.ranks, &self.group_comms, &group)
            .filter(|comm| comm.all_reduce(&ts))
            .map(comm_name);
        let path = match via_comm {
            Some(path) => path,
            None => {
                // No comm, or ggml declined (nothing was summed).
                self.scratch_for(&group, ts[0])?;
                let key = (group.clone(), ts[0].ne());
                if let Some(s) = self.scratch.get_mut(&key) {
                    s.all_reduce(&self.ranks, &group, &ts)?;
                }
                "fallback"
            }
        };
        *self
            .counters
            .collective_bytes
            .entry(("all_reduce", path))
            .or_default() += bytes;
        Ok(())
    }

    fn copy(
        &mut self,
        src: (RankId, Tensor<'_>),
        dst: (RankId, Tensor<'_>),
    ) -> Result<(), ExecError> {
        if src.0 >= self.ranks.len() || dst.0 >= self.ranks.len() {
            return Err(ExecError::invalid(format!(
                "copy from rank {} to rank {} of {}",
                src.0,
                dst.0,
                self.ranks.len()
            )));
        }
        src.1
            .copy_tensor_async(self.ranks.backend(src.0), self.ranks.backend(dst.0), dst.1)?;
        *self
            .counters
            .collective_bytes
            .entry(("copy", self.ranks.comm_path()))
            .or_default() += src.1.nbytes() as u64;
        Ok(())
    }
}

fn elapsed_ns(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn is_all_ranks(group: &[RankId], n: usize) -> bool {
    group.len() == n && group.iter().copied().eq(0..n)
}

/// The comm for a group: the one over all ranks in rank order, else one made for this group.
fn find_comm<'x>(
    ranks: &'x Ranks,
    groups: &'x HashMap<Vec<RankId>, Option<Comm>>,
    group: &[RankId],
) -> Option<&'x Comm> {
    if is_all_ranks(group, ranks.len()) {
        ranks.comm.as_ref()
    } else {
        groups.get(group).and_then(Option::as_ref)
    }
}
