//! A device-less [`Allocator`] for tests: fixed capacities, an alignment, a one-time init cost
//! and a per-allocation overhead.

use crate::arena::{Allocator, ArenaBuffer, ArenaError};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy)]
pub struct FakeDevice {
    pub total: u64,
    /// Taken by someone else before we start.
    pub used_by_others: u64,
    pub alignment: u64,
    pub max_size: u64,
    /// Taken once by `init`.
    pub init_cost: u64,
    /// Taken on top of every allocation.
    pub alloc_overhead: u64,
}

impl FakeDevice {
    pub fn new(total: u64) -> Self {
        Self {
            total,
            used_by_others: 0,
            alignment: 256,
            max_size: u64::MAX,
            init_cost: 0,
            alloc_overhead: 0,
        }
    }
}

#[derive(Debug)]
struct State {
    dev: FakeDevice,
    used: u64,
    inited: bool,
}

#[derive(Debug, Clone)]
pub struct FakeAllocator {
    devices: Arc<Mutex<BTreeMap<String, State>>>,
}

impl FakeAllocator {
    pub fn new(devices: impl IntoIterator<Item = (String, FakeDevice)>) -> Self {
        let map = devices
            .into_iter()
            .map(|(k, dev)| {
                let used = dev.used_by_others;
                let state = State {
                    dev,
                    used,
                    inited: false,
                };
                (k, state)
            })
            .collect();
        Self {
            devices: Arc::new(Mutex::new(map)),
        }
    }

    /// Bytes in use on a device, including what others hold.
    pub fn used(&self, device: &str) -> u64 {
        self.devices.lock().unwrap()[device].used
    }

    fn with<T>(&self, device: &str, f: impl FnOnce(&mut State) -> T) -> Result<T, ArenaError> {
        let mut m = self.devices.lock().unwrap();
        m.get_mut(device)
            .map(f)
            .ok_or_else(|| ArenaError::UnknownDevice(device.to_string()))
    }
}

#[derive(Debug)]
pub struct FakeBuffer {
    size: u64,
    device: String,
    charged: u64,
    owner: Arc<Mutex<BTreeMap<String, State>>>,
}

impl ArenaBuffer for FakeBuffer {
    fn size(&self) -> u64 {
        self.size
    }

    fn base(&self) -> *mut u8 {
        std::ptr::null_mut()
    }
}

impl Drop for FakeBuffer {
    fn drop(&mut self) {
        if let Some(s) = self.owner.lock().unwrap().get_mut(&self.device) {
            s.used -= self.charged;
        }
    }
}

impl Allocator for FakeAllocator {
    type Buffer = FakeBuffer;

    fn alignment(&self, device: &str) -> Result<u64, ArenaError> {
        self.with(device, |s| s.dev.alignment)
    }

    fn max_size(&self, device: &str) -> Result<u64, ArenaError> {
        self.with(device, |s| s.dev.max_size)
    }

    fn alloc(&self, device: &str, size: u64) -> Result<Option<FakeBuffer>, ArenaError> {
        let charged = self.with(device, |s| {
            let charged = size + s.dev.alloc_overhead;
            (s.used + charged <= s.dev.total).then(|| {
                s.used += charged;
                charged
            })
        })?;
        Ok(charged.map(|charged| FakeBuffer {
            size,
            device: device.to_string(),
            charged,
            owner: self.devices.clone(),
        }))
    }

    fn memory(&self, device: &str) -> Result<(u64, u64), ArenaError> {
        self.with(device, |s| (s.dev.total - s.used, s.dev.total))
    }

    fn init(&self, device: &str) -> Result<(), ArenaError> {
        self.with(device, |s| {
            if !std::mem::replace(&mut s.inited, true) {
                s.used = (s.used + s.dev.init_cost).min(s.dev.total);
            }
        })
    }
}
