//! `ggml_gallocr`: allocates a graph's intermediates in one buffer of a backend's default
//! buffer type, without the scheduler (the executor's own rank programs, ADR 0069).

use super::{Backend, GgmlError, Graph, Result};
use crate::raw as sys;
use std::ptr::NonNull;

/// A graph allocator for one backend. Like [`super::Sched`] it remembers the graph it allocated.
/// Allocating another graph (or growing the buffer) invalidates the data of every earlier graph.
pub struct Gallocr {
    ptr: NonNull<sys::ggml_gallocr>,
    allocated: Option<u64>,
    // Keeps the backend (and its device context) alive for as long as the buffer.
    _backend: Backend,
}

impl Gallocr {
    pub fn new(backend: &Backend) -> Result<Gallocr> {
        // SAFETY: live backend; the buffer type is static per device.
        let ptr = unsafe {
            sys::ggml_gallocr_new(sys::ggml_backend_get_default_buffer_type(backend.raw()))
        };
        let ptr = NonNull::new(ptr).ok_or(GgmlError::Alloc {
            what: "graph allocator",
            bytes: 0,
        })?;
        Ok(Gallocr {
            ptr,
            allocated: None,
            _backend: backend.clone(),
        })
    }

    /// Sizes the buffer for a worst-case graph so later allocations do not grow it. Forgets the
    /// current allocation.
    pub fn reserve(&mut self, g: &Graph<'_>) -> Result<()> {
        self.allocated = None;
        // SAFETY: live allocator and graph.
        if unsafe { sys::ggml_gallocr_reserve(self.ptr.as_ptr(), g.ptr.as_ptr()) } {
            Ok(())
        } else {
            Err(GgmlError::Alloc {
                what: "graph allocator buffer",
                bytes: 0,
            })
        }
    }

    /// Assigns the graph's tensors their data. Inputs can be written afterwards.
    pub fn alloc_graph(&mut self, g: &Graph<'_>) -> Result<()> {
        self.allocated = None;
        // SAFETY: live allocator and graph.
        if !unsafe { sys::ggml_gallocr_alloc_graph(self.ptr.as_ptr(), g.ptr.as_ptr()) } {
            return Err(GgmlError::Alloc {
                what: "graph tensors",
                bytes: 0,
            });
        }
        self.allocated = Some(g.id);
        Ok(())
    }

    /// True if `g` is the graph of the last successful `alloc_graph`, so it can be computed again
    /// with new input data and no new allocation.
    pub fn is_allocated(&self, g: &Graph<'_>) -> bool {
        self.allocated == Some(g.id)
    }

    /// Bytes of the buffer.
    pub fn buffer_size(&self) -> usize {
        // SAFETY: live allocator; buffer 0 exists for a single-buffer allocator.
        unsafe { sys::ggml_gallocr_get_buffer_size(self.ptr.as_ptr(), 0) }
    }
}

impl Drop for Gallocr {
    fn drop(&mut self) {
        // SAFETY: owned; the backend outlives it (field dropped after).
        unsafe { sys::ggml_gallocr_free(self.ptr.as_ptr()) }
    }
}
