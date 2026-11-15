//! `ggml_backend_event`: orders work across backends (streams) without a host wait.

use super::{Backend, GgmlError, Result};
use crate::raw as sys;
use std::ptr::NonNull;

/// An event of one device. Only backends of that device may record or wait on it. The CPU device
/// has no events: [`Event::new`] fails for it.
pub struct Event {
    ptr: NonNull<sys::ggml_backend_event>,
    // Frees through the device, so the backend must outlive the event.
    _backend: Backend,
}

impl Event {
    pub fn new(backend: &Backend) -> Result<Event> {
        // SAFETY: live backend; its device handle is valid for the process. ggml returns null
        // when the device has no events.
        let ptr =
            unsafe { sys::ggml_backend_event_new(sys::ggml_backend_get_device(backend.raw())) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| GgmlError::Invalid(format!("`{}` has no events", backend.name())))?;
        Ok(Event {
            ptr,
            _backend: backend.clone(),
        })
    }

    /// Marks the point reached by the work queued so far on `backend`'s stream.
    pub fn record(&self, backend: &Backend) {
        // SAFETY: live event and backend.
        unsafe { sys::ggml_backend_event_record(self.ptr.as_ptr(), backend.raw()) }
    }

    /// Makes later work on `backend`'s stream wait for the recorded point (no host wait).
    pub fn wait(&self, backend: &Backend) {
        // SAFETY: live event and backend.
        unsafe { sys::ggml_backend_event_wait(backend.raw(), self.ptr.as_ptr()) }
    }

    /// Blocks the host until the recorded point is reached.
    pub fn synchronize(&self) {
        // SAFETY: live event.
        unsafe { sys::ggml_backend_event_synchronize(self.ptr.as_ptr()) }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: owned; the backend (and so the device) outlives it.
        unsafe { sys::ggml_backend_event_free(self.ptr.as_ptr()) }
    }
}
