//! ggml's multi-backend communication context (`ggml_backend_comm_*`, reached through the
//! backend registry): NCCL or an internal path for sums across GPUs (ADR 0069, Collectives).

use super::{Backend, Tensor};
use crate::raw as sys;
use std::ffi::{CStr, c_void};
use std::ptr::NonNull;

/// The all-reduce path ggml uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommPath {
    Nccl,
    /// ggml's own two-device path.
    Internal,
    /// ggml declines; the caller falls back to copies.
    None,
}

type InitFn = unsafe extern "C" fn(*mut sys::ggml_backend_t, usize) -> *mut c_void;
type FreeFn = unsafe extern "C" fn(*mut c_void);
type AllReduceFn = unsafe extern "C" fn(*mut c_void, *mut *mut sys::ggml_tensor) -> bool;

pub struct Comm {
    ctx: NonNull<c_void>,
    free: FreeFn,
    allreduce: AllReduceFn,
    // Kept alive: the context points at their streams and devices. Dropped after `free`.
    backends: Vec<Backend>,
}

impl Comm {
    /// None when the first backend's registry has no `ggml_backend_comm_init` (CPU) or it
    /// returns null (not all backends are of one kind).
    pub fn new(backends: &[Backend]) -> Option<Comm> {
        let first = backends.first()?;
        // SAFETY: live backend; its device and registry handles are valid for the process.
        let reg =
            unsafe { sys::ggml_backend_dev_backend_reg(sys::ggml_backend_get_device(first.raw())) };
        let proc = |name: &CStr| {
            // SAFETY: valid registry; null for an unknown name.
            let p = unsafe { sys::ggml_backend_reg_get_proc_address(reg, name.as_ptr()) };
            (!p.is_null()).then_some(p)
        };
        let init = proc(c"ggml_backend_comm_init")?;
        let free = proc(c"ggml_backend_comm_free")?;
        let allreduce = proc(c"ggml_backend_comm_allreduce_tensor")?;
        // SAFETY: the registry returns these entry points with the signatures of the comm
        // typedefs in ggml-backend.h, as ggml-backend-meta.cpp casts them.
        let (init, free, allreduce) = unsafe {
            (
                std::mem::transmute::<*mut c_void, InitFn>(init),
                std::mem::transmute::<*mut c_void, FreeFn>(free),
                std::mem::transmute::<*mut c_void, AllReduceFn>(allreduce),
            )
        };
        let mut raw: Vec<sys::ggml_backend_t> = backends.iter().map(Backend::raw).collect();
        // SAFETY: live backends, kept alive by the returned `Comm`; null means no comm.
        let ctx = NonNull::new(unsafe { init(raw.as_mut_ptr(), raw.len()) })?;
        Some(Comm {
            ctx,
            free,
            allreduce,
            backends: backends.to_vec(),
        })
    }

    /// In-place sum of one tensor per backend, in backend order. False: ggml declined (or the
    /// call does not fit: wrong tensor count, tensors without a buffer), nothing was summed.
    pub fn all_reduce(&self, tensors: &[Tensor<'_>]) -> bool {
        if tensors.len() != self.backends.len() {
            return false;
        }
        let mut raw: Vec<*mut sys::ggml_tensor> = tensors.iter().map(|t| t.raw()).collect();
        // SAFETY: the tensors are live; only fields are read.
        let unallocated = raw
            .iter()
            .any(|&t| unsafe { (*t).buffer.is_null() && (*t).view_src.is_null() });
        if unallocated {
            return false;
        }
        // SAFETY: live comm context; `raw` holds one tensor per backend, as the context expects.
        unsafe { (self.allreduce)(self.ctx.as_ptr(), raw.as_mut_ptr()) }
    }

    /// What ggml will use, derived from the build and `GGML_CUDA_ALLREDUCE` (ggml does not
    /// report it). An NCCL init failure falls back to internal inside ggml and is not seen here.
    pub fn path(&self) -> CommPath {
        derive_path(
            cfg!(tetraeder_nccl),
            cfg!(target_os = "linux"),
            std::env::var("GGML_CUDA_ALLREDUCE").ok().as_deref(),
            self.backends.len(),
        )
    }
}

/// Mirrors `ggml_backend_cuda_comm_init` in ggml-cuda.cu: unset means NCCL on Linux and internal
/// elsewhere; without NCCL built in, internal, which needs exactly two devices.
fn derive_path(nccl_built: bool, linux: bool, mode: Option<&str>, n: usize) -> CommPath {
    let internal = if n == 2 {
        CommPath::Internal
    } else {
        CommPath::None
    };
    let nccl = if nccl_built { CommPath::Nccl } else { internal };
    match mode {
        None if linux => nccl,
        None | Some("internal") => internal,
        Some("nccl") => nccl,
        // `none` and unknown values.
        Some(_) => CommPath::None,
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        // SAFETY: owned context, freed once; the backends are still alive (fields drop after).
        unsafe { (self.free)(self.ctx.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::{CommPath, derive_path};

    #[test]
    fn path_rules() {
        let (nccl, internal, none) = (CommPath::Nccl, CommPath::Internal, CommPath::None);
        assert_eq!(derive_path(true, true, None, 4), nccl);
        assert_eq!(derive_path(true, true, Some("nccl"), 2), nccl);
        assert_eq!(derive_path(false, true, None, 2), internal);
        assert_eq!(derive_path(false, true, None, 4), none);
        assert_eq!(derive_path(true, false, None, 2), internal);
        assert_eq!(derive_path(true, true, Some("internal"), 2), internal);
        assert_eq!(derive_path(true, true, Some("none"), 2), none);
        assert_eq!(derive_path(true, true, Some("bogus"), 2), none);
    }
}
