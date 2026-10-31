//! Thread-pool engine: positional reads on scoped threads. Used on Linux when io_uring is
//! unavailable (old kernel, seccomp in containers) and on other Unixes.

use super::{Engine, Job};
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct Pool {
    threads: usize,
}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let threads = if threads == 0 { 8 } else { threads };
        Self { threads }
    }
}

fn read_job(file: &File, j: &mut Job) -> io::Result<()> {
    // SAFETY: Job::new's contract: ptr is valid for len bytes and ours alone until we return.
    let buf = unsafe { std::slice::from_raw_parts_mut(j.ptr, j.len) };
    let mut at = 0;
    while at < buf.len() {
        match file.read_at(&mut buf[at..], j.offset + at as u64) {
            Ok(0) => break,
            Ok(n) => at += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    j.filled = at;
    Ok(())
}

impl Engine for Pool {
    fn name(&self) -> &'static str {
        "thread-pool"
    }

    fn read(&self, file: &File, jobs: &mut [Job]) -> io::Result<()> {
        let workers = self.threads.min(jobs.len());
        if workers <= 1 {
            return jobs.iter_mut().try_for_each(|j| read_job(file, j));
        }
        let next = AtomicUsize::new(0);
        let first_error: Mutex<Option<io::Error>> = Mutex::new(None);
        let slots: Vec<Mutex<&mut Job>> = jobs.iter_mut().map(Mutex::new).collect();
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(slot) = slots.get(i) else { break };
                        let mut job = slot.lock().unwrap_or_else(|e| e.into_inner());
                        if let Err(e) = read_job(file, &mut job) {
                            first_error
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .get_or_insert(e);
                            break;
                        }
                    }
                });
            }
        });
        match first_error.into_inner().unwrap_or_else(|e| e.into_inner()) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
