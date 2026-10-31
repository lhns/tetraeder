//! io_uring engine. Probed at startup: a kernel without io_uring (ENOSYS) or a container
//! whose seccomp profile blocks it (EPERM, Docker's default) makes `new` fail and the caller
//! use the thread pool. Not exercised by the Windows host; CI containers usually take the pool.

use super::{Engine, Job};
use io_uring::{IoUring, opcode, types};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::Mutex;

pub(super) struct Uring {
    ring: Mutex<IoUring>,
    depth: usize,
}

impl Uring {
    pub fn new(depth: usize) -> io::Result<Self> {
        let depth = if depth == 0 { 64 } else { depth };
        let ring = IoUring::new(depth as u32)?;
        Ok(Self {
            ring: Mutex::new(ring),
            depth,
        })
    }
}

impl Engine for Uring {
    fn name(&self) -> &'static str {
        "io_uring"
    }

    fn read(&self, file: &File, jobs: &mut [Job]) -> io::Result<()> {
        if jobs.iter().any(|j| u32::try_from(j.len).is_err()) {
            return Err(io::Error::other("read larger than 4 GiB"));
        }
        let fd = types::Fd(file.as_raw_fd());
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        for batch in jobs.chunks_mut(self.depth) {
            for (i, j) in batch.iter().enumerate() {
                let sqe = opcode::Read::new(fd, j.ptr, j.len as u32)
                    .offset(j.offset)
                    .build()
                    .user_data(i as u64);
                // SAFETY: the buffer is valid until this call returns (Job::new), and every
                // entry pushed here is waited for below before returning. The batch fits the
                // queue, so the push cannot fail.
                unsafe { ring.submission().push(&sqe) }.expect("batch is at most the queue depth");
            }
            let mut done = 0;
            let mut error = None;
            while done < batch.len() {
                match ring.submit_and_wait(batch.len() - done) {
                    Ok(_) => {}
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
                for cqe in ring.completion() {
                    done += 1;
                    let res = cqe.result();
                    if res < 0 {
                        error.get_or_insert(io::Error::from_raw_os_error(-res));
                    } else {
                        batch[cqe.user_data() as usize].filled = res as usize;
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
