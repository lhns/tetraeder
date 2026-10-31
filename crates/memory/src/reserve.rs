//! Reserve measurement (ADR 0031): what a device cannot give to the plan. Measured at startup
//! from free memory before and after backend init plus a test allocation, reported as the
//! value the plan validator's reserve map takes (`tetraeder_plan::Context::reserve`).

use crate::arena::{Allocator, ArenaBuffer, ArenaError};
use std::collections::BTreeMap;

/// Raw readings of one device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReserveReport {
    pub device: String,
    pub total: u64,
    pub free_before: u64,
    pub free_after_init: u64,
    pub free_after_alloc: u64,
    /// Size of the test allocation (rounded up to the device alignment).
    pub test_bytes: u64,
}

impl ReserveReport {
    /// Memory already taken before we started: other processes, the desktop compositor.
    pub fn others(&self) -> u64 {
        self.total.saturating_sub(self.free_before)
    }

    /// What bringing the backend up cost: context, workspaces, graph pools.
    pub fn init_cost(&self) -> u64 {
        self.free_before.saturating_sub(self.free_after_init)
    }

    /// What the allocator took beyond the bytes asked for (page granularity, metadata).
    pub fn alloc_overhead(&self) -> u64 {
        self.free_after_init
            .saturating_sub(self.free_after_alloc)
            .saturating_sub(self.test_bytes)
    }

    /// Bytes to subtract from capacity before planning.
    pub fn reserve(&self) -> u64 {
        self.others()
            .saturating_add(self.init_cost())
            .saturating_add(self.alloc_overhead())
    }
}

/// Measure one device: read free memory, bring the backend up, allocate `test_bytes` (freed
/// again on return) and read again.
pub fn measure_reserve<A: Allocator>(
    alloc: &A,
    device: &str,
    test_bytes: u64,
) -> Result<ReserveReport, ArenaError> {
    let (free_before, total) = alloc.memory(device)?;
    alloc.init(device)?;
    let (free_after_init, _) = alloc.memory(device)?;
    let size = test_bytes.next_multiple_of(alloc.alignment(device)?.max(1));
    let buf = alloc.alloc(device, size)?.ok_or(ArenaError::Alloc {
        device: device.to_string(),
        purpose: crate::Purpose::Reserve,
        size,
    })?;
    let (free_after_alloc, _) = alloc.memory(device)?;
    let test_bytes = buf.size();
    drop(buf);
    Ok(ReserveReport {
        device: device.to_string(),
        total,
        free_before,
        free_after_init,
        free_after_alloc,
        test_bytes,
    })
}

/// The map the validator takes.
pub fn reserve_map(reports: &[ReserveReport]) -> BTreeMap<String, u64> {
    reports
        .iter()
        .map(|r| (r.device.clone(), r.reserve()))
        .collect()
}
