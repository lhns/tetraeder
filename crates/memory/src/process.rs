//! Resident set size of this process: the number that shows a run stayed inside the box.

/// Peak resident set size in bytes since the process started (`VmHWM`, `PeakWorkingSetSize`).
/// `None` where the platform gives no number.
pub fn peak_rss() -> Option<u64> {
    imp::read(true)
}

/// Current resident set size in bytes (`VmRSS`, `WorkingSetSize`).
pub fn current_rss() -> Option<u64> {
    imp::read(false)
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn read(peak: bool) -> Option<u64> {
        let key = if peak { "VmHWM:" } else { "VmRSS:" };
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let kb: u64 = status
            .lines()
            .find_map(|l| l.strip_prefix(key))?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        Some(kb * 1024)
    }
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    pub fn read(peak: bool) -> Option<u64> {
        // SAFETY: an all-zero PROCESS_MEMORY_COUNTERS is a valid value (plain integers).
        let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        c.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        // SAFETY: the pseudo handle of the current process is always valid, and `c` is a live
        // struct whose size is passed in `cb`.
        let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        (ok != 0).then_some(if peak {
            c.PeakWorkingSetSize
        } else {
            c.WorkingSetSize
        } as u64)
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod imp {
    pub fn read(_peak: bool) -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn rss_is_reported_and_peak_is_at_least_current() {
        let ballast = vec![1u8; 8 << 20];
        let cur = current_rss().expect("current rss");
        let peak = peak_rss().expect("peak rss");
        assert!(cur > 0 && peak >= cur, "current {cur}, peak {peak}");
        assert!(ballast.iter().all(|&b| b == 1));
    }
}
