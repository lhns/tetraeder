//! Backend devices and buffer types.

use super::{LlamaError, Result, cstr_lossy, init};
use crate::raw as sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    IGpu,
    Accel,
    Other,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    pub kind: DeviceKind,
    pub memory_free: u64,
    pub memory_total: u64,
}

/// All ggml backend devices (CPU, GPUs, accelerators).
pub fn devices() -> Vec<DeviceInfo> {
    init();
    // SAFETY: the registry is initialized; device handles stay valid for the process.
    unsafe {
        (0..sys::ggml_backend_dev_count())
            .map(|i| {
                let d = sys::ggml_backend_dev_get(i);
                let (mut free, mut total) = (0usize, 0usize);
                sys::ggml_backend_dev_memory(d, &mut free, &mut total);
                let t = sys::ggml_backend_dev_type(d) as i64;
                let kind = if t == sys::GGML_BACKEND_DEVICE_TYPE_CPU as i64 {
                    DeviceKind::Cpu
                } else if t == sys::GGML_BACKEND_DEVICE_TYPE_GPU as i64 {
                    DeviceKind::Gpu
                } else if t == sys::GGML_BACKEND_DEVICE_TYPE_IGPU as i64 {
                    DeviceKind::IGpu
                } else if t == sys::GGML_BACKEND_DEVICE_TYPE_ACCEL as i64 {
                    DeviceKind::Accel
                } else {
                    DeviceKind::Other
                };
                DeviceInfo {
                    name: cstr_lossy(sys::ggml_backend_dev_name(d)),
                    description: cstr_lossy(sys::ggml_backend_dev_description(d)),
                    kind,
                    memory_free: free as u64,
                    memory_total: total as u64,
                }
            })
            .collect()
    }
}

/// A ggml buffer type (where tensors live), for example `CPU` or `CUDA0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferType(pub(super) sys::ggml_backend_buffer_type_t);

// SAFETY: buffer types are immutable objects owned by the backend registry for the whole
// process; this wrapper only reads their name and hands the pointer back to libllama.
unsafe impl Send for BufferType {}
// SAFETY: see `Send`.
unsafe impl Sync for BufferType {}

impl BufferType {
    pub fn name(&self) -> String {
        // SAFETY: the handle is non-null and registry-owned.
        cstr_lossy(unsafe { sys::ggml_backend_buft_name(self.0) })
    }
}

fn all_buffer_types() -> Vec<BufferType> {
    init();
    let mut out = Vec::new();
    // SAFETY: registry handles are valid; the host buffer type may be null.
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let d = sys::ggml_backend_dev_get(i);
            for b in [
                sys::ggml_backend_dev_buffer_type(d),
                sys::ggml_backend_dev_host_buffer_type(d),
            ] {
                if !b.is_null() && !out.contains(&BufferType(b)) {
                    out.push(BufferType(b));
                }
            }
        }
    }
    out
}

/// Names accepted by [`buffer_type`].
pub fn buffer_type_names() -> Vec<String> {
    all_buffer_types().iter().map(BufferType::name).collect()
}

/// Looks a buffer type up by name, case-insensitively, over every device's buffer type and
/// host buffer type (the set `llama-cli --override-tensor` accepts, plus host buffers).
pub fn buffer_type(name: &str) -> Option<BufferType> {
    all_buffer_types()
        .into_iter()
        .find(|b| b.name().eq_ignore_ascii_case(name))
}

/// Like [`buffer_type`], with an error that lists the known names.
pub fn resolve_buffer_type(name: &str) -> Result<BufferType> {
    buffer_type(name).ok_or_else(|| LlamaError::UnknownBufferType {
        name: name.to_owned(),
        known: buffer_type_names(),
    })
}
