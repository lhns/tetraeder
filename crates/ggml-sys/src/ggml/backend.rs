//! Backend and buffer handles.

use super::{GgmlError, Result, cpu, cstr};
use crate::raw as sys;
use std::ptr::NonNull;
use std::rc::Rc;

struct BackendInner {
    ptr: NonNull<sys::ggml_backend>,
    is_cpu: bool,
    /// The CPU backend's own threadpool. Without one, ggml creates and joins its worker threads
    /// on every graph compute.
    pool: Option<NonNull<sys::ggml_threadpool>>,
}

impl Drop for BackendInner {
    fn drop(&mut self) {
        // SAFETY: owned; every Buffer and Sched holds an Rc, so none outlives this. The backend
        // goes first: it uses the pool until it is freed and never frees the pool itself.
        unsafe {
            sys::ggml_backend_free(self.ptr.as_ptr());
            if let (Some(pool), Ok(procs)) = (self.pool, cpu::procs()) {
                (procs.threadpool_free)(pool.as_ptr());
            }
        }
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

/// `TETRAEDER_REQUIRE_GPU` is set to anything but empty or `0`.
fn flag_set(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

fn require_gpu() -> bool {
    flag_set(std::env::var_os("TETRAEDER_REQUIRE_GPU").as_deref())
}

impl Backend {
    /// The CPU backend with `n_threads` compute threads in a threadpool of its own that lives as
    /// long as the backend (as libllama with an attached pool and llama-bench run).
    pub fn cpu(n_threads: i32) -> Result<Backend> {
        Self::cpu_with_poll(n_threads, None)
    }

    /// [`Backend::cpu`] with the threadpool's `poll` level (0 to 100, clamped): how long idle
    /// workers spin before they sleep. `None`: ggml's default (50).
    pub fn cpu_with_poll(n_threads: i32, poll: Option<u32>) -> Result<Backend> {
        let procs = cpu::procs()?;
        let n = n_threads.max(1);
        // SAFETY: registry lookup, then the plain constructor; null means failure.
        let ptr = unsafe {
            let dev = sys::ggml_backend_dev_by_type(sys::GGML_BACKEND_DEVICE_TYPE_CPU);
            if dev.is_null() {
                std::ptr::null_mut()
            } else {
                sys::ggml_backend_dev_init(dev, std::ptr::null())
            }
        };
        let ptr = NonNull::new(ptr).ok_or_else(|| GgmlError::NoBackend("CPU".into()))?;
        // SAFETY: `params` is a valid in-out struct; null means failure.
        let pool = NonNull::new(unsafe {
            let mut params = sys::ggml_threadpool_params_default(n);
            if let Some(p) = poll {
                params.poll = p.min(100);
            }
            (procs.threadpool_new)(&mut params)
        });
        // Owned from here, so a failure below frees both.
        let inner = BackendInner {
            ptr,
            is_cpu: true,
            pool,
        };
        let Some(pool) = pool else {
            return Err(GgmlError::NoBackend("CPU threadpool".into()));
        };
        // SAFETY: live CPU backend and pool; the pool outlives the backend (freed after it).
        unsafe {
            (procs.set_n_threads)(ptr.as_ptr(), n);
            (procs.set_threadpool)(ptr.as_ptr(), pool.as_ptr());
        }
        Ok(Backend {
            inner: Rc::new(inner),
        })
    }

    /// Compute threads of the next graphs on a CPU backend; keep it within the threadpool's size.
    /// No effect on a GPU backend.
    pub fn set_cpu_threads(&self, n_threads: i32) {
        // A CPU backend exists only after `cpu::procs()` succeeded.
        if let (true, Ok(procs)) = (self.inner.is_cpu, cpu::procs()) {
            // SAFETY: live CPU backend; only the thread count of later graph computes changes.
            unsafe { (procs.set_n_threads)(self.inner.ptr.as_ptr(), n_threads.max(1)) };
        }
    }

    /// The `index`-th GPU device (discrete GPUs first, then integrated), if the build has one.
    pub fn gpu(index: usize) -> Result<Backend> {
        crate::llama::init_checked().map_err(GgmlError::NoBackend)?;
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
            inner: Rc::new(BackendInner {
                ptr,
                is_cpu: false,
                pool: None,
            }),
        })
    }

    /// True when the build has at least one GPU device. Tests use it to skip GPU checks; with
    /// `TETRAEDER_REQUIRE_GPU=1` (set on GPU boxes) a missing GPU panics instead, so a skip cannot pass for a run.
    pub fn gpu_available() -> bool {
        let present = Backend::gpu_count() > 0;
        assert!(
            present || !require_gpu(),
            "TETRAEDER_REQUIRE_GPU is set but this build or machine has no GPU device"
        );
        present
    }

    pub fn gpu_count() -> usize {
        // A failed module load leaves no GPU; the constructors report the error.
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

    /// Marks the buffer as holding weights. The scheduler runs an op on the backend whose buffer
    /// holds a weights-usage source, so weights in host memory compute on the CPU even when a
    /// GPU is the primary backend.
    pub fn set_usage_weights(&self) {
        // SAFETY: live buffer.
        unsafe {
            sys::ggml_backend_buffer_set_usage(
                self.ptr.as_ptr(),
                sys::GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            )
        }
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

#[cfg(test)]
mod flag_tests {
    use super::flag_set;
    use std::ffi::OsStr;

    #[test]
    fn require_gpu_flag_values() {
        assert!(!flag_set(None));
        assert!(!flag_set(Some(OsStr::new(""))));
        assert!(!flag_set(Some(OsStr::new("0"))));
        assert!(flag_set(Some(OsStr::new("1"))));
        assert!(flag_set(Some(OsStr::new("true"))));
    }
}
