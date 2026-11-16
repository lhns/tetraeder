use crate::{ExecError, RankId};
use tetraeder_ggml_sys::ggml::{Backend, Comm, CommPath};

/// What a rank computes on. CPU ranks stand in for GPUs in CI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankDevice {
    /// The `n`-th GPU of ggml's device list ([`Backend::gpu`]). Two ranks may name the same GPU.
    Gpu(usize),
    Cpu {
        threads: i32,
    },
}

/// The backends of a plan's ranks in rank order, and the all-reduce path over all of them.
pub struct Ranks {
    backends: Vec<Backend>,
    pub(crate) comm: Option<Comm>,
    comm_path: &'static str,
}

impl Ranks {
    pub fn open(devices: &[RankDevice]) -> Result<Ranks, ExecError> {
        let backends = devices
            .iter()
            .map(|d| match *d {
                RankDevice::Gpu(i) => Backend::gpu(i),
                RankDevice::Cpu { threads } => Backend::cpu(threads),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let comm = group_comm(&backends);
        let comm_path = comm.as_ref().map_or("fallback", comm_name);
        if backends.len() > 1 {
            tracing::info!(
                ranks = backends.len(),
                comm = comm_path,
                "executor all-reduce path"
            );
        }
        Ok(Ranks {
            backends,
            comm,
            comm_path,
        })
    }

    pub fn backend(&self, r: RankId) -> &Backend {
        &self.backends[r]
    }

    pub fn len(&self) -> usize {
        self.backends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// "nccl", "internal" or "fallback" (copies through rank 0): the path of an all-reduce over
    /// all ranks. One rank, CPU ranks and several ranks on one GPU have no comm: "fallback".
    pub fn comm_path(&self) -> &'static str {
        self.comm_path
    }

    /// Ranks on a GPU. A plan with two or more of them on the fallback path is slow
    /// (alert `TetraederCollectiveFallback`).
    pub fn gpu_ranks(&self) -> usize {
        self.backends.iter().filter(|b| !b.is_cpu()).count()
    }

    /// Records the comm path as the `tetraeder.executor.comm` gauge when the plan has two or more
    /// GPU ranks (the alert reads `fallback` as a slow tensor-parallel group).
    pub fn record(&self, metrics: &tetraeder_telemetry::Metrics) {
        if self.gpu_ranks() >= 2 {
            metrics.executor_comm(self.comm_path);
        }
    }
}

pub(crate) fn comm_name(c: &Comm) -> &'static str {
    match c.path() {
        CommPath::Nccl => "nccl",
        CommPath::Internal => "internal",
        CommPath::None => "fallback",
    }
}

/// A comm over these backends, if they are two or more distinct GPUs that ggml can reduce over.
pub(crate) fn group_comm(backends: &[Backend]) -> Option<Comm> {
    if backends.len() < 2 || backends.iter().any(Backend::is_cpu) {
        return None;
    }
    let mut seen: Vec<_> = backends.iter().map(Backend::gpu_index).collect();
    seen.sort_unstable();
    // The comm inits one NCCL rank per device: ranks sharing a GPU cannot join.
    if seen.windows(2).any(|w| w[0] == w[1]) {
        return None;
    }
    Comm::new(backends)
}
