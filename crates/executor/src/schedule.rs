use std::cmp::Reverse;

/// Puts ops tagged `(stage, micro-batch)` in GPipe wavefront order (ADR 0077): by
/// `stage + micro-batch`, then later stages first; ops of one tag keep their order. Stage s then
/// issues micro-batch m after m - 1 and after stage s - 1 issued m (with its copy), and stage
/// s + 1 starts m - 1 before stage s starts m + 1.
///
/// Later stages first because a copy enqueues the destination stream's wait for it when it is
/// issued (ggml-cuda `cpy_tensor_async`): the hop into stage s + 1 for m must be issued after
/// stage s + 1 issued m - 1, or that range waits for stage s to finish m (ADR 0077 History).
pub fn gpipe_sort<T>(mut ops: Vec<((usize, usize), T)>) -> Vec<T> {
    // Stable: the order inside a (stage, micro-batch) group is the builder's.
    ops.sort_by_key(|&((s, m), _)| (s + m, Reverse(s)));
    ops.into_iter().map(|(_, op)| op).collect()
}
