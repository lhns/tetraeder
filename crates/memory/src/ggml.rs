//! [`Allocator`] on ggml backends, through the raw bindings. One backend buffer per arena, from
//! the device's default buffer type.

use crate::arena::{Allocator, ArenaBuffer, ArenaError};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use tetraeder_ggml_sys::raw;

struct Dev {
    dev: raw::ggml_backend_dev_t,
    /// Created by `init`, freed on drop.
    backend: Cell<raw::ggml_backend_t>,
}

/// Maps plan device ids to ggml devices. Not `Sync`: backend creation is lazy and unguarded.
pub struct GgmlAllocator {
    devices: BTreeMap<String, Dev>,
}

impl Default for GgmlAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl GgmlAllocator {
    pub fn new() -> Self {
        Self {
            devices: BTreeMap::new(),
        }
    }

    /// Names of the ggml devices of this build (`CPU`, `CUDA0`, ...).
    pub fn available() -> Vec<String> {
        // SAFETY: device handles are static registry entries; the name is a valid C string.
        unsafe {
            (0..raw::ggml_backend_dev_count())
                .map(|i| {
                    let name = raw::ggml_backend_dev_name(raw::ggml_backend_dev_get(i));
                    CStr::from_ptr(name).to_string_lossy().into_owned()
                })
                .collect()
        }
    }

    /// Serve plan device `id` from the ggml device called `ggml_name`.
    pub fn bind(&mut self, id: &str, ggml_name: &str) -> Result<(), ArenaError> {
        let c = CString::new(ggml_name).map_err(|_| ArenaError::UnknownDevice(ggml_name.into()))?;
        // SAFETY: c is a valid C string for the call.
        let dev = unsafe { raw::ggml_backend_dev_by_name(c.as_ptr()) };
        self.insert(id, dev, ggml_name)
    }

    /// Serve plan device `id` from ggml's CPU device.
    pub fn bind_cpu(&mut self, id: &str) -> Result<(), ArenaError> {
        // SAFETY: plain lookup in the registry.
        let dev = unsafe { raw::ggml_backend_dev_by_type(raw::GGML_BACKEND_DEVICE_TYPE_CPU) };
        self.insert(id, dev, "CPU")
    }

    fn insert(
        &mut self,
        id: &str,
        dev: raw::ggml_backend_dev_t,
        name: &str,
    ) -> Result<(), ArenaError> {
        if dev.is_null() {
            return Err(ArenaError::UnknownDevice(name.to_string()));
        }
        self.devices.insert(
            id.to_string(),
            Dev {
                dev,
                backend: Cell::new(std::ptr::null_mut()),
            },
        );
        Ok(())
    }

    fn dev(&self, id: &str) -> Result<&Dev, ArenaError> {
        self.devices
            .get(id)
            .ok_or_else(|| ArenaError::UnknownDevice(id.to_string()))
    }

    fn buft(&self, id: &str) -> Result<raw::ggml_backend_buffer_type_t, ArenaError> {
        // SAFETY: the device handle is valid for the process lifetime.
        Ok(unsafe { raw::ggml_backend_dev_buffer_type(self.dev(id)?.dev) })
    }
}

impl Drop for GgmlAllocator {
    fn drop(&mut self) {
        for d in self.devices.values() {
            let b = d.backend.get();
            if !b.is_null() {
                // SAFETY: created by ggml_backend_dev_init in `init`, freed once.
                unsafe { raw::ggml_backend_free(b) };
            }
        }
    }
}

/// An owned ggml backend buffer.
#[derive(Debug)]
pub struct GgmlBuffer {
    buf: raw::ggml_backend_buffer_t,
}

impl ArenaBuffer for GgmlBuffer {
    fn size(&self) -> u64 {
        // SAFETY: buf is live until drop.
        unsafe { raw::ggml_backend_buffer_get_size(self.buf) as u64 }
    }

    fn base(&self) -> *mut u8 {
        // SAFETY: as above.
        unsafe { raw::ggml_backend_buffer_get_base(self.buf).cast() }
    }
}

impl Drop for GgmlBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated by ggml_backend_buft_alloc_buffer, freed once.
        unsafe { raw::ggml_backend_buffer_free(self.buf) };
    }
}

impl Allocator for GgmlAllocator {
    type Buffer = GgmlBuffer;

    fn alignment(&self, device: &str) -> Result<u64, ArenaError> {
        // SAFETY: buft is a valid buffer type of a live device.
        Ok(unsafe { raw::ggml_backend_buft_get_alignment(self.buft(device)?) as u64 })
    }

    fn max_size(&self, device: &str) -> Result<u64, ArenaError> {
        // SAFETY: as above. ggml reports SIZE_MAX for "no limit".
        Ok(unsafe { raw::ggml_backend_buft_get_max_size(self.buft(device)?) as u64 })
    }

    fn alloc(&self, device: &str, size: u64) -> Result<Option<GgmlBuffer>, ArenaError> {
        let size = usize::try_from(size).map_err(|_| ArenaError::Alloc {
            device: device.to_string(),
            purpose: crate::Purpose::Slots,
            size,
        })?;
        // SAFETY: the buffer type is valid; a null result means the allocation failed.
        let buf = unsafe { raw::ggml_backend_buft_alloc_buffer(self.buft(device)?, size) };
        Ok((!buf.is_null()).then_some(GgmlBuffer { buf }))
    }

    fn memory(&self, device: &str) -> Result<(u64, u64), ArenaError> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both out pointers are valid for the call.
        unsafe { raw::ggml_backend_dev_memory(self.dev(device)?.dev, &mut free, &mut total) };
        Ok((free as u64, total as u64))
    }

    fn init(&self, device: &str) -> Result<(), ArenaError> {
        let d = self.dev(device)?;
        if d.backend.get().is_null() {
            // SAFETY: the device is valid; a null params string means defaults.
            let b = unsafe { raw::ggml_backend_dev_init(d.dev, std::ptr::null()) };
            if b.is_null() {
                return Err(ArenaError::UnknownDevice(device.to_string()));
            }
            d.backend.set(b);
        }
        Ok(())
    }
}
