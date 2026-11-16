//! Executor on CUDA ranks. Labels: `gpu_` needs one GPU, `gpu2_` two, `gpu4_` four. Each skips
//! with `skip: needs n GPUs` unless `TETRAEDER_REQUIRE_GPUS=<n>` asks for at least n, which makes
//! a missing GPU fail (Rule 9). Run on a box with
//! `cargo test -p tetraeder-executor --features cuda --test gpu -- --nocapture`;
//! `gpu_cuda_graphs_replay_kept_ranges` also needs `GGML_CUDA_GRAPH_DEBUG=1` in the environment.

mod common;

use common::{LEN, all_reduce_case, pipeline_case};
use std::cell::RefCell;
use tetraeder_executor::RankDevice::Gpu;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{CudaGraphLogCounts, cuda_graph_log_counts};

/// Two ranks on one GPU have no comm: they run the fallback all-reduce and a copy hop.
#[test]
fn gpu_two_backends_of_one_gpu_run_the_fallback_and_the_copy() {
    if !Backend::require_gpus(1) {
        return;
    }
    let [first, second] = all_reduce_case(&[Gpu(0), Gpu(0)], &[vec![0, 1]]);
    assert_eq!(
        first.collective_bytes.get(&("all_reduce", "fallback")),
        Some(&(2 * LEN as u64 * 4))
    );
    assert_eq!(second.allocations, 0);
    let c = pipeline_case([Gpu(0), Gpu(0)], &[3], &[3], 3, |_| {});
    assert_eq!(
        c.collective_bytes.get(&("copy", "fallback")),
        Some(&(3 * LEN as u64 * 4))
    );
    let cases: [(&[usize], &[usize]); 2] = [(&[1, 3], &[2, 3]), (&[1, 2, 3], &[3])];
    for (c0, c1) in cases {
        pipeline_case([Gpu(0), Gpu(0)], c0, c1, 3, |_| {});
    }
}

/// ADR 0036 for range graphs: a kept two-rank program captures one CUDA graph per range and
/// then replays it; range graphs have no `uid`, so a replay is silent in ggml's log and shows as
/// the absence of new warmups, resets and failed updates on every later step.
#[test]
fn gpu_cuda_graphs_replay_kept_ranges() {
    if !Backend::require_gpus(1) {
        return;
    }
    if std::env::var_os("GGML_CUDA_GRAPH_DEBUG").is_none() {
        let required = ["TETRAEDER_REQUIRE_GPU", "TETRAEDER_REQUIRE_GPUS"]
            .iter()
            .any(|k| std::env::var_os(k).is_some());
        assert!(!required, "set GGML_CUDA_GRAPH_DEBUG=1 for this test");
        eprintln!("skip: set GGML_CUDA_GRAPH_DEBUG=1");
        return;
    }
    if std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some() {
        eprintln!("skip: CUDA graphs are disabled");
        return;
    }
    let start = cuda_graph_log_counts();
    let seen: RefCell<Vec<CudaGraphLogCounts>> = RefCell::new(Vec::new());
    // Two ranks, two ranges each.
    pipeline_case([Gpu(0), Gpu(0)], &[1, 3], &[1, 3], 8, |_| {
        seen.borrow_mut().push(cuda_graph_log_counts());
    });
    let seen = seen.into_inner();
    eprintln!("start {start:?}");
    for (i, c) in seen.iter().enumerate() {
        eprintln!("step {i}: {c:?}");
    }
    let (warm, last) = (seen[3], seen[7]);
    assert!(
        warm.warmups - start.warmups >= 4,
        "one capture per range expected: {start:?} then {warm:?}"
    );
    assert_eq!(
        (warm.warmups, warm.resets, warm.update_failed),
        (last.warmups, last.resets, last.update_failed),
        "a kept range was captured again or reset after its warmup"
    );
}

/// An all-reduce through ggml's comm over two GPUs, and a peer copy hop.
#[test]
fn gpu2_all_reduce_through_the_comm() {
    if !Backend::require_gpus(2) {
        return;
    }
    let [first, second] = all_reduce_case(&[Gpu(0), Gpu(1)], &[vec![0, 1]]);
    eprintln!("collective bytes: {:?}", first.collective_bytes);
    assert!(
        !first
            .collective_bytes
            .contains_key(&("all_reduce", "fallback")),
        "ggml declined the all-reduce (GGML_CUDA_ALLREDUCE=none?): {:?}",
        first.collective_bytes
    );
    assert_eq!(second.allocations, 0);
    pipeline_case([Gpu(0), Gpu(1)], &[3], &[3], 3, |_| {});
    pipeline_case([Gpu(1), Gpu(0)], &[1, 3], &[2, 3], 3, |_| {});
}

/// Four GPUs: the all-reduce over all of them, over two pairs, and over a group that is not the
/// first ranks. Without NCCL ggml's internal path takes exactly two GPUs, so the four-GPU group
/// may run the fallback (printed).
#[test]
fn gpu4_all_reduce_over_groups() {
    if !Backend::require_gpus(4) {
        return;
    }
    let devices = [Gpu(0), Gpu(1), Gpu(2), Gpu(3)];
    let [all, _] = all_reduce_case(&devices, &[vec![0, 1, 2, 3]]);
    eprintln!("all four: {:?}", all.collective_bytes);
    let [pairs, _] = all_reduce_case(&devices, &[vec![0, 1], vec![2, 3]]);
    eprintln!("two pairs: {:?}", pairs.collective_bytes);
    assert!(
        !pairs
            .collective_bytes
            .contains_key(&("all_reduce", "fallback")),
        "pairs should use ggml's two-GPU path: {:?}",
        pairs.collective_bytes
    );
    all_reduce_case(&devices, &[vec![3, 1]]);
}
