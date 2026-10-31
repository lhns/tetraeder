//! Per-device arenas (ADR 0031): one backend buffer per (device, purpose), sized from the
//! expanded plan. Allocation goes through [`Allocator`], so tests run on a fake device and the
//! real thing runs on a ggml backend (`ggml` feature).

use std::collections::BTreeMap;
use std::fmt;
use tetraeder_plan::{BufferKind, Expanded, PhaseBytes};
use tetraeder_profiler::HardwareGraph;

/// What an arena holds. `Reserve` is accounting only: the measured headroom that is subtracted
/// from capacity, never a live buffer (a live one would count twice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Purpose {
    Kv,
    Slots,
    Workspace,
    Reserve,
}

impl Purpose {
    /// The `tetraeder.memory.kind` value.
    pub fn as_str(self) -> &'static str {
        match self {
            Purpose::Kv => "kv",
            Purpose::Slots => "slots",
            Purpose::Workspace => "workspace",
            Purpose::Reserve => "reserve",
        }
    }
}

impl From<BufferKind> for Purpose {
    fn from(k: BufferKind) -> Self {
        match k {
            BufferKind::Kv => Purpose::Kv,
            BufferKind::Workspace => Purpose::Workspace,
            BufferKind::SlotCache => Purpose::Slots,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArenaError {
    /// The arenas of a device do not fit what is left of it.
    OverBudget {
        device: String,
        needed: u64,
        available: u64,
    },
    /// One arena is larger than the backend allows in one buffer.
    TooLarge {
        device: String,
        purpose: Purpose,
        size: u64,
        max: u64,
    },
    /// The backend refused the allocation.
    Alloc {
        device: String,
        purpose: Purpose,
        size: u64,
    },
    UnknownDevice(String),
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ArenaError {}

/// A device buffer. `base` is null for devices without host-visible memory.
pub trait ArenaBuffer {
    fn size(&self) -> u64;
    fn base(&self) -> *mut u8;
}

/// Allocates buffers on named devices (the plan's device ids).
pub trait Allocator {
    type Buffer: ArenaBuffer;

    /// Buffer size granularity and base alignment of the device.
    fn alignment(&self, device: &str) -> Result<u64, ArenaError>;
    /// Largest single buffer; `u64::MAX` when unlimited.
    fn max_size(&self, device: &str) -> Result<u64, ArenaError>;
    /// `None` when the backend refuses.
    fn alloc(&self, device: &str, size: u64) -> Result<Option<Self::Buffer>, ArenaError>;
    /// `(free, total)` bytes of the device.
    fn memory(&self, device: &str) -> Result<(u64, u64), ArenaError>;
    /// Bring the device backend up (context creation), so its cost shows in [`Self::memory`].
    fn init(&self, device: &str) -> Result<(), ArenaError>;
}

/// Requested bytes per (device, purpose), summed over replicas and stages.
pub fn arena_sizes(e: &Expanded) -> BTreeMap<(String, Purpose), u64> {
    let mut sizes: BTreeMap<(String, Purpose), u64> = BTreeMap::new();
    for (k, b) in &e.buffers {
        let s = sizes.entry((k.device.clone(), k.kind.into())).or_default();
        *s = s.saturating_add(*b);
    }
    sizes
}

/// Bytes each device can give to arenas: capacity minus the measured reserve minus the weights
/// homed on it. Devices without weights or buffers are left out.
pub fn available_for_arenas(
    e: &Expanded,
    hw: &HardwareGraph,
    reserve: &BTreeMap<String, u64>,
) -> BTreeMap<String, u64> {
    let mut weights: BTreeMap<&str, u64> = BTreeMap::new();
    for p in e.shards.values() {
        let w = weights.entry(&p.home).or_default();
        *w = w.saturating_add(p.bytes);
    }
    let mut out = BTreeMap::new();
    for (device, _) in arena_sizes(e).keys() {
        if let Some(d) = hw.device(device) {
            let used = reserve
                .get(device)
                .copied()
                .unwrap_or(0)
                .saturating_add(weights.get(device.as_str()).copied().unwrap_or(0));
            out.insert(device.clone(), d.capacity_bytes.saturating_sub(used));
        }
    }
    out
}

/// Slot and workspace bytes the plan gives `phase` on `device`, summed over stages.
pub fn phase_bytes(e: &Expanded, device: &str, phase: Phase) -> PhaseBytes {
    let mut total = PhaseBytes::default();
    for (k, b) in &e.budgets {
        if k.device == device {
            let p = match phase {
                Phase::Prefill => b.prefill,
                Phase::Decode => b.decode,
            };
            total.slots = total.slots.saturating_add(p.slots);
            total.workspace = total.workspace.saturating_add(p.workspace);
        }
    }
    total
}

/// One allocated arena.
#[derive(Debug)]
pub struct Arena<B> {
    pub device: String,
    pub purpose: Purpose,
    /// Bytes the plan asked for.
    pub requested: u64,
    pub buffer: B,
}

/// All arenas of a plan.
#[derive(Debug)]
pub struct Arenas<B> {
    arenas: BTreeMap<(String, Purpose), Arena<B>>,
}

impl<B: ArenaBuffer> Arenas<B> {
    /// Allocate every arena of `e`. Sizes are rounded up to the device alignment; each device's
    /// total must fit `available` (see [`available_for_arenas`]) before anything is allocated.
    /// Zero-size arenas are skipped.
    pub fn allocate<A: Allocator<Buffer = B>>(
        alloc: &A,
        e: &Expanded,
        available: &BTreeMap<String, u64>,
    ) -> Result<Self, ArenaError> {
        let mut want: Vec<(String, Purpose, u64, u64)> = Vec::new();
        let mut needed: BTreeMap<&str, u64> = BTreeMap::new();
        let sizes = arena_sizes(e);
        for ((device, purpose), &requested) in &sizes {
            if requested == 0 {
                continue;
            }
            let align = alloc.alignment(device)?.max(1);
            let size = requested.next_multiple_of(align);
            let max = alloc.max_size(device)?;
            if size > max {
                return Err(ArenaError::TooLarge {
                    device: device.clone(),
                    purpose: *purpose,
                    size,
                    max,
                });
            }
            let n = needed.entry(device).or_default();
            *n = n.saturating_add(size);
            want.push((device.clone(), *purpose, requested, size));
        }
        for (device, needed) in needed {
            let avail = available.get(device).copied().unwrap_or(u64::MAX);
            if needed > avail {
                return Err(ArenaError::OverBudget {
                    device: device.to_string(),
                    needed,
                    available: avail,
                });
            }
        }
        let mut arenas = BTreeMap::new();
        for (device, purpose, requested, size) in want {
            let buffer = alloc.alloc(&device, size)?.ok_or(ArenaError::Alloc {
                device: device.clone(),
                purpose,
                size,
            })?;
            arenas.insert(
                (device.clone(), purpose),
                Arena {
                    device,
                    purpose,
                    requested,
                    buffer,
                },
            );
        }
        Ok(Self { arenas })
    }

    pub fn get(&self, device: &str, purpose: Purpose) -> Option<&Arena<B>> {
        self.arenas.get(&(device.to_string(), purpose))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arena<B>> {
        self.arenas.values()
    }

    /// Allocated bytes per device.
    pub fn allocated(&self) -> BTreeMap<String, u64> {
        let mut out: BTreeMap<String, u64> = BTreeMap::new();
        for a in self.arenas.values() {
            *out.entry(a.device.clone()).or_default() += a.buffer.size();
        }
        out
    }
}
