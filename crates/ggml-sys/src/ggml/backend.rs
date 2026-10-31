//! Backend and buffer handles.

use super::{GgmlError, Result, cstr};
use crate::raw as sys;
use std::ptr::NonNull;
use std::rc::Rc;

struct BackendInner {
    ptr: NonNull<sys::ggml_backend>,
    is_cpu: bool,
}

impl Drop for BackendInner {
    fn drop(&mut self) {
        // SAFETY: owned; every Buffer and Sched holds an Rc, so none outlives this.
        unsafe { sys::ggml_backend_free(self.ptr.as_ptr()) }
    }
}

/// A ggml backend instance (CPU or one GPU). Clones share the instance.
#[derive(Clone)]
pub struct Backend {
    inner: Rc<BackendInner>,
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Backend({})", self.name())
    }
}

impl Backend {
    /// The CPU backend with `n_threads` compute threads.
    pub fn cpu(n_threads: i32) -> Result<Backend> {
        crate::llama::init();
        // SAFETY: plain constructor; null means failure.
        let ptr = unsafe { sys::ggml_backend_cpu_init() };
        let ptr = NonNull::new(ptr).ok_or_else(|| GgmlError::NoBackend("CPU".into()))?;
        // SAFETY: ptr is a live CPU backend.
        unsafe { sys::ggml_backend_cpu_set_n_threads(ptr.as_ptr(), n_threads.max(1)) };
        Ok(Backend {
            inner: Rc::new(BackendInner { ptr, is_cpu: true }),
        })
    }

    /// The `index`-th GPU device (discrete GPUs first, then integrated), if the build has one.
    pub fn gpu(index: usize) -> Result<Backend> {
        crate::llama::init();
        // SAFETY: the registry is initialized; device handles are valid for the process.
        let dev = unsafe {
            let n = sys::ggml_backend_dev_count();
            let is =
                |d, t: sys::ggml_backend_dev_type| sys::ggml_backend_dev_type(d) as i64 == t as i64;
            let devs: Vec<_> = (0..n).map(|i| sys::ggml_backend_dev_get(i)).collect();
            devs.iter()
                .copied()
                .filter(|&d| is(d, sys::GGML_BACKEND_DEVICE_TYPE_GPU))
                .chain(
                    devs.iter()
                        .copied()
                        .filter(|&d| is(d, sys::GGML_BACKEND_DEVICE_TYPE_IGPU)),
                )
                .nth(index)
        };
        let dev = dev.ok_or_else(|| GgmlError::NoBackend(format!("GPU {index}")))?;
        // SAFETY: dev is a valid device; null params select defaults.
        let ptr = unsafe { sys::ggml_backend_dev_init(dev, std::ptr::null()) };
        let ptr = NonNull::new(ptr).ok_or_else(|| GgmlError::NoBackend(format!("GPU {index}")))?;
        Ok(Backend {
            inner: Rc::new(BackendInner { ptr, is_cpu: false }),
        })
    }

    /// True when the build has at least one GPU device.
    pub fn gpu_available() -> bool {
        Backend::gpu_count() > 0
    }

    pub fn gpu_count() -> usize {
        crate::llama::init();
        // SAFETY: registry queries only.
        unsafe {
            (0..sys::ggml_backend_dev_count())
                .filter(|&i| {
                    let t = sys::ggml_backend_dev_type(sys::ggml_backend_dev_get(i)) as i64;
                    t == sys::GGML_BACKEND_DEVICE_TYPE_GPU as i64
                        || t == sys::GGML_BACKEND_DEVICE_TYPE_IGPU as i64
                })
                .count()
        }
    }

    pub fn name(&self) -> String {
        // SAFETY: live backend; static string.
        cstr(unsafe { sys::ggml_backend_name(self.inner.ptr.as_ptr()) })
    }

    pub fn is_cpu(&self) -> bool {
        self.inner.is_cpu
    }

    pub fn synchronize(&self) {
        // SAFETY: live backend.
        unsafe { sys::ggml_backend_synchronize(self.inner.ptr.as_ptr()) }
    }

    pub(crate) fn raw(&self) -> sys::ggml_backend_t {
        self.inner.ptr.as_ptr()
    }

    pub(crate) fn same(&self, other: &Backend) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

/// A backend buffer holding tensor data. Frees itself on drop.
pub struct Buffer {
    ptr: NonNull<sys::ggml_backend_buffer>,
    // Keeps the backend (and its device context) alive for as long as the buffer.
    _backend: Backend,
}

impl Buffer {
    /// Allocates every tensor of `ctx` in one buffer of the backend's default buffer type.
    pub(super) fn alloc_ctx_tensors(
        ctx: *mut sys::ggml_context,
        backend: &Backend,
    ) -> Result<Buffer> {
        // SAFETY: ctx is a live no_alloc context; the buffer type belongs to a live backend.
        let ptr = unsafe {
            let buft = sys::ggml_backend_get_default_buffer_type(backend.raw());
            sys::ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)
        };
        let ptr = NonNull::new(ptr).ok_or(GgmlError::Alloc {
            what: "tensor buffer",
            bytes: 0,
        })?;
        Ok(Buffer {
            ptr,
            _backend: backend.clone(),
        })
    }

    pub fn size(&self) -> usize {
        // SAFETY: live buffer.
        unsafe { sys::ggml_backend_buffer_get_size(self.ptr.as_ptr()) }
    }

    pub fn clear(&self, value: u8) {
        // SAFETY: live buffer.
        unsafe { sys::ggml_backend_buffer_clear(self.ptr.as_ptr(), value) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: owned; the owning Context frees it before its tensors are dropped.
        unsafe { sys::ggml_backend_buffer_free(self.ptr.as_ptr()) }
    }
}
