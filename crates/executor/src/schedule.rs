/// Puts ops tagged `(stage, micro-batch)` in GPipe wavefront order (ADR 0077): by
/// `stage + micro-batch`, then by stage; ops of one tag keep their order. Stage s then issues
/// micro-batch m after m - 1 and after stage s - 1 issued m (with its copy), and stage s + 1
/// starts m - 1 before stage s starts m + 1, so a range that blocks the host still overlaps.
pub fn gpipe_sort<T>(mut ops: Vec<((usize, usize), T)>) -> Vec<T> {
    // Stable: the order inside a (stage, micro-batch) group is the builder's.
    ops.sort_by_key(|&((s, m), _)| (s + m, s));
    ops.into_iter().map(|(_, op)| op).collect()
}
