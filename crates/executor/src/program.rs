use tetraeder_ggml_sys::ggml::{Graph, Tensor};

/// Index of a rank in [`crate::Ranks`] (plan order).
pub type RankId = usize;

/// One step of a program.
#[derive(Debug, Clone)]
pub enum Op<'a> {
    /// Range `range` of the rank's graph (the rank's one [`RankGraph`]).
    Compute { rank: RankId, range: usize },
    /// In-place sum of one F32 tensor per rank of a group, same shape on each. Each tensor is
    /// the last node of a range of its rank.
    AllReduce { tensors: Vec<(RankId, Tensor<'a>)> },
    /// A pipeline hop: `src` (the last node of a range of its rank, or data that already
    /// exists) to `dst` (an input of its rank's graph), same type, shape and strides.
    Copy {
        src: (RankId, Tensor<'a>),
        dst: (RankId, Tensor<'a>),
    },
}

/// The graph of one rank for a step.
pub struct RankGraph<'a> {
    pub rank: RankId,
    pub graph: Graph<'a>,
    /// Node index ending each range, strictly increasing, the last is `graph.n_nodes()`.
    pub cuts: Vec<usize>,
}

/// A step: rank graphs, the ops in issue order and the ranks whose outputs the caller reads
/// (synchronized when the program ends).
pub struct Program<'a> {
    pub graphs: Vec<RankGraph<'a>>,
    pub ops: Vec<Op<'a>>,
    pub outputs: Vec<RankId>,
    /// The step carries more than one input row per sequence (prefill, chunked prefill,
    /// pipelined micro-batches): its range graphs are never captured as CUDA graphs (ADR 0081).
    /// Decode steps keep capture and replay.
    pub prefill: bool,
}
