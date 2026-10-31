//! `ggml_backend_sched`: assigns graph nodes to backends, allocates intermediates, computes.

use super::{Backend, GgmlError, Graph, Result, Tensor};
use crate::raw as sys;
use std::ptr::NonNull;

/// The backend scheduler. The last backend must be the CPU (ggml falls back to it).
///
/// The scheduler keeps pointers into the graph's context until the next [`Sched::reset`] or
/// [`Sched::alloc_graph`]. It remembers which graph it allocated and refuses to compute any other,
/// so a graph (and its context) that was dropped can never be computed by mistake.
pub struct Sched {
    ptr: NonNull<sys::ggml_backend_sched>,
    backends: Vec<Backend>,
    allocated: Option<u64>,
}

impl Sched {
    /// `graph_nodes` bounds the graph size; `op_offload` lets the scheduler run large ops of
    /// host-resident weights on a GPU backend.
    pub fn new(backends: &[Backend], graph_nodes: usize, op_offload: bool) -> Result<Sched> {
        if backends.last().is_none_or(|b| !b.is_cpu()) {
            return Err(GgmlError::Invalid(
                "the last backend must be the CPU".into(),
            ));
        }
        let mut raw: Vec<sys::ggml_backend_t> = backends.iter().map(Backend::raw).collect();
        // SAFETY: the backends are live and kept alive by `self.backends`; null bufts select
        // each backend's default buffer type.
        let ptr = unsafe {
            sys::ggml_backend_sched_new(
                raw.as_mut_ptr(),
                std::ptr::null_mut(),
                raw.len() as i32,
                graph_nodes,
                false,
                op_offload,
            )
        };
        let ptr = NonNull::new(ptr).ok_or(GgmlError::Alloc {
            what: "backend scheduler",
            bytes: 0,
        })?;
        Ok(Sched {
            ptr,
            backends: backends.to_vec(),
            allocated: None,
        })
    }

    /// Pins a node to a backend (must be one of this scheduler's). Call before `alloc_graph`.
    pub fn set_tensor_backend(&mut self, t: Tensor<'_>, backend: &Backend) -> Result<()> {
        if !self.backends.iter().any(|b| b.same(backend)) {
            return Err(GgmlError::Invalid(
                "backend is not part of the scheduler".into(),
            ));
        }
        // SAFETY: live scheduler, tensor and backend.
        unsafe {
            sys::ggml_backend_sched_set_tensor_backend(self.ptr.as_ptr(), t.raw(), backend.raw())
        };
        Ok(())
    }

    /// Sizes the compute buffers from a worst-case graph so later allocations do not grow them.
    pub fn reserve(&mut self, graph: &Graph<'_>) -> Result<()> {
        self.allocated = None;
        // SAFETY: live scheduler and graph.
        let ok = unsafe { sys::ggml_backend_sched_reserve(self.ptr.as_ptr(), graph.ptr.as_ptr()) };
        if ok {
            Ok(())
        } else {
            Err(GgmlError::Alloc {
                what: "scheduler compute buffers",
                bytes: 0,
            })
        }
    }

    /// Forgets the current allocation. Tensors of the previous graph lose their data.
    pub fn reset(&mut self) {
        // SAFETY: live scheduler.
        unsafe { sys::ggml_backend_sched_reset(self.ptr.as_ptr()) };
        self.allocated = None;
    }

    /// Resets, then splits and allocates `graph`. Inputs can be written afterwards.
    pub fn alloc_graph(&mut self, graph: &Graph<'_>) -> Result<()> {
        self.reset();
        // SAFETY: live scheduler and graph.
        let ok =
            unsafe { sys::ggml_backend_sched_alloc_graph(self.ptr.as_ptr(), graph.ptr.as_ptr()) };
        if !ok {
            return Err(GgmlError::Alloc {
                what: "graph tensors",
                bytes: 0,
            });
        }
        self.allocated = Some(graph.id);
        Ok(())
    }

    /// Computes the graph given to the last `alloc_graph` and waits for the result.
    pub fn compute(&mut self, graph: &Graph<'_>) -> Result<()> {
        if self.allocated != Some(graph.id) {
            return Err(GgmlError::Invalid(
                "compute needs the graph from the last alloc_graph".into(),
            ));
        }
        // SAFETY: live scheduler; the id check proves the graph is the allocated one and alive
        // (the graph borrows its context).
        let st =
            unsafe { sys::ggml_backend_sched_graph_compute(self.ptr.as_ptr(), graph.ptr.as_ptr()) };
        if st as i64 == sys::GGML_STATUS_SUCCESS as i64 {
            Ok(())
        } else {
            Err(GgmlError::Compute(st as i32))
        }
    }

    /// Splits of the last allocated graph (a split is a run on one backend).
    pub fn n_splits(&self) -> usize {
        // SAFETY: live scheduler.
        unsafe { sys::ggml_backend_sched_get_n_splits(self.ptr.as_ptr()) as usize }
    }
}

impl Drop for Sched {
    fn drop(&mut self) {
        // SAFETY: owned; the backends outlive it (held in `self.backends`, dropped after).
        unsafe { sys::ggml_backend_sched_free(self.ptr.as_ptr()) }
    }
}
