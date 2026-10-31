//! Overlapped reads on a handle opened with `FILE_FLAG_OVERLAPPED`: up to `depth` reads in
//! flight, each with its own event, collected in submission order.

use super::{Engine, Job};
use std::fs::File;
use std::io;
use std::os::windows::io::AsRawHandle;
use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, FALSE, GetLastError, HANDLE, TRUE,
};
use windows_sys::Win32::Storage::FileSystem::ReadFile;
use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::Threading::CreateEventW;

pub(super) struct Overlapped {
    depth: usize,
}

impl Overlapped {
    pub fn new(depth: usize) -> Self {
        Self {
            depth: if depth == 0 { 32 } else { depth },
        }
    }
}

/// A read in flight. The OVERLAPPED is boxed so its address stays fixed while the kernel
/// holds it.
struct Pending {
    ov: Box<OVERLAPPED>,
    event: HANDLE,
}

impl Drop for Pending {
    fn drop(&mut self) {
        // SAFETY: the event was created by CreateEventW and is closed once.
        unsafe { CloseHandle(self.event) };
    }
}

fn last_error() -> io::Error {
    // SAFETY: plain FFI call.
    io::Error::from_raw_os_error(unsafe { GetLastError() } as i32)
}

impl Engine for Overlapped {
    fn name(&self) -> &'static str {
        "overlapped"
    }

    fn read(&self, file: &File, jobs: &mut [Job]) -> io::Result<()> {
        let handle = file.as_raw_handle() as HANDLE;
        if jobs.iter().any(|j| u32::try_from(j.len).is_err()) {
            return Err(io::Error::other("read larger than 4 GiB"));
        }
        let mut error: Option<io::Error> = None;
        for batch in jobs.chunks_mut(self.depth) {
            let mut pending: Vec<Option<Pending>> = Vec::with_capacity(batch.len());
            for j in batch.iter() {
                if error.is_some() {
                    pending.push(None);
                    continue;
                }
                // SAFETY: plain FFI call; manual-reset event, initially not signaled.
                let event =
                    unsafe { CreateEventW(std::ptr::null(), TRUE, FALSE, std::ptr::null()) };
                if event.is_null() {
                    error = Some(last_error());
                    pending.push(None);
                    continue;
                }
                // SAFETY: OVERLAPPED is plain data; all zero is its documented initial state.
                let mut ov: Box<OVERLAPPED> = Box::new(unsafe { std::mem::zeroed() });
                ov.hEvent = event;
                ov.Anonymous.Anonymous.Offset = j.offset as u32;
                ov.Anonymous.Anonymous.OffsetHigh = (j.offset >> 32) as u32;
                let p = Pending { ov, event };
                // SAFETY: the buffer is valid for len bytes until this call returns (Job::new)
                // and every read issued here is waited for below, also when another one failed.
                let ok = unsafe {
                    ReadFile(
                        handle,
                        j.ptr.cast(),
                        j.len as u32,
                        std::ptr::null_mut(),
                        (&*p.ov as *const OVERLAPPED).cast_mut(),
                    )
                };
                if ok == 0 {
                    let code = unsafe { GetLastError() };
                    if code != ERROR_IO_PENDING {
                        if code != ERROR_HANDLE_EOF {
                            error = Some(io::Error::from_raw_os_error(code as i32));
                        }
                        // Not in flight: nothing to wait for. EOF at the offset: filled stays 0.
                        pending.push(None);
                        continue;
                    }
                }
                pending.push(Some(p));
            }
            for (j, p) in batch.iter_mut().zip(pending) {
                let Some(p) = p else { continue };
                let mut n = 0u32;
                // SAFETY: waits for this read; ov and the buffer are still alive.
                let ok = unsafe { GetOverlappedResult(handle, &*p.ov, &mut n, TRUE) };
                if ok != 0 {
                    j.filled = n as usize;
                } else {
                    let code = unsafe { GetLastError() };
                    if code != ERROR_HANDLE_EOF {
                        error.get_or_insert(io::Error::from_raw_os_error(code as i32));
                    }
                }
            }
            if let Some(e) = error {
                return Err(e);
            }
        }
        Ok(())
    }
}
