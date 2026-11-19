//! Spare buffers for KV snapshots (ADR 0074). A snapshot is tens of MB, above the allocator's
//! mmap threshold, so a fresh `Vec` page-faults on every byte of the save, with every live
//! sequence stalled. A background thread keeps a few buffers ready (allocated, every page
//! touched); a save takes one, and the prefix cache hands released buffers back. When none is
//! ready a save allocates plainly, without any advice: advised memory can stall the faulting
//! thread in direct compaction.
//!
//! The target size only grows (the longest snapshot hinted so far), usually by a few percent per
//! request. The preparer grows a spare that fell below it and touches only the new tail; it frees
//! a spare only when the cache budget no longer holds them all at the target. Frees run on the
//! preparer thread, in pieces: a multi-GB free in one go stalled the steps it overlapped by up to
//! 0.7 s, and replacing every spare at each new target left saves without one (ttft6-20261119).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use tetraeder_ggml_sys::ggml::advise_hugepages;

/// Env: spare buffers kept ready (default 2; 0 disables the pool).
pub const SPARES_ENV: &str = "TETRAEDER_SNAPSHOT_SPARES";
/// Env: `1` advises huge pages on the buffers the preparer makes (default off: on a host with
/// fragmented memory the advice made each spare take 0.5 s of direct compaction, and with the
/// pages touched off the admission path it gains nothing; `20261117h`, ADR 0074).
pub const THP_ENV: &str = "TETRAEDER_SNAPSHOT_THP";

/// Bytes a free gives back per piece: about 7 ms of unmapping at the 9.5 GB/s measured on the
/// ttft6-20261119 host, the longest one free holds the address space.
pub const FREE_PIECE: usize = 64 << 20;

#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Spares kept at most; 0 turns the pool off (every save allocates plainly, as before).
    pub spares: usize,
    /// Advise huge pages on the buffers the preparer makes.
    pub thp: bool,
    /// Buffers below this capacity are never pooled.
    pub min_bytes: usize,
    /// Capacities are rounded up to a multiple of this.
    pub round: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            spares: 2,
            thp: false,
            min_bytes: 1 << 20,
            round: 2 << 20,
        }
    }
}

impl PoolConfig {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Some(n) = std::env::var(SPARES_ENV)
            .ok()
            .and_then(|v| v.trim().parse().ok())
        {
            c.spares = n;
        }
        c.thp = std::env::var(THP_ENV).is_ok_and(|v| v.trim() == "1");
        c
    }
}

/// Counters since the pool was made, and what is ready now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Saves served by a ready spare.
    pub from_spare: u64,
    /// Saves that allocated plainly.
    pub fallback: u64,
    /// Released buffers taken back into the pool.
    pub recycled: u64,
    /// Bytes the preparer touched (new spares and grown tails).
    pub touched_bytes: u64,
    /// Bytes the preparer freed (spares over the budget, released buffers without room).
    pub freed_bytes: u64,
    pub ready: usize,
    pub ready_bytes: usize,
}

/// Work for the preparer thread.
enum Job {
    /// Free this buffer, in pieces.
    Free(Vec<u8>),
    /// Touch this buffer up to the given length (a new spare starts empty).
    Fill(Vec<u8>, usize),
}

#[derive(Default)]
struct State {
    /// Spares; the first `len()` bytes of each are touched (the contents are junk).
    ready: Vec<Vec<u8>>,
    /// Released buffers without room, to free on the preparer thread.
    discard: Vec<Vec<u8>>,
    /// Capacity of the spare the preparer is filling, if any (counted as held).
    filling: Option<usize>,
    /// Capacity of the buffers to prepare: the largest snapshot asked for, rounded up.
    target: usize,
    /// Most bytes worth holding (the prefix cache budget); 0 means no limit yet.
    max_bytes: usize,
    started: bool,
    shutdown: bool,
}

impl State {
    fn ready_bytes(&self) -> usize {
        self.ready.iter().map(Vec::capacity).sum()
    }

    /// Bytes held as spares, including the one being filled.
    fn held_bytes(&self) -> usize {
        self.ready_bytes() + self.filling.unwrap_or(0)
    }

    fn fits(&self, bytes: usize) -> bool {
        self.max_bytes == 0 || bytes <= self.max_bytes
    }

    fn has_room(&self, spares: usize, cap: usize) -> bool {
        self.ready.len() + usize::from(self.filling.is_some()) < spares
            && self.fits(self.held_bytes() + cap)
    }

    /// Spares worth holding at the target: `spares`, fewer when the budget holds fewer.
    fn keep(&self, spares: usize) -> usize {
        match (self.max_bytes, self.target) {
            (0, _) | (_, 0) => spares,
            (max, t) => spares.min(max / t),
        }
    }

    fn take_smallest(&mut self) -> Vec<u8> {
        let i = (0..self.ready.len())
            .min_by_key(|&i| self.ready[i].capacity())
            .expect("a spare to take");
        self.ready.swap_remove(i)
    }

    /// The preparer's next job: free a released buffer, free a spare the budget no longer
    /// holds, grow the least touched spare to the target, or make a new spare.
    fn next_job(&mut self, spares: usize) -> Option<Job> {
        if let Some(v) = self.discard.pop() {
            return Some(Job::Free(v));
        }
        let t = self.target;
        if t == 0 {
            return None;
        }
        if !self.ready.is_empty() && self.ready.len() > self.keep(spares) {
            return Some(Job::Free(self.take_smallest()));
        }
        let short = (0..self.ready.len())
            .filter(|&i| self.ready[i].len() < t)
            .min_by_key(|&i| self.ready[i].len());
        if let Some(i) = short {
            let grow = t.saturating_sub(self.ready[i].capacity());
            if !self.fits(self.held_bytes() + grow) {
                // Growing it needs the bytes of one spare.
                return Some(Job::Free(self.take_smallest()));
            }
            let v = self.ready.swap_remove(i);
            self.filling = Some(v.capacity().max(t));
            return Some(Job::Fill(v, t));
        }
        if self.has_room(spares, t) {
            self.filling = Some(t);
            return Some(Job::Fill(Vec::new(), t));
        }
        None
    }
}

struct Shared {
    cfg: PoolConfig,
    /// Spares kept at most; starts as `cfg.spares`, see [`SnapshotPool::set_default_spares`].
    spares: AtomicUsize,
    state: Mutex<State>,
    wake: Condvar,
    from_spare: AtomicU64,
    fallback: AtomicU64,
    recycled: AtomicU64,
    touched: AtomicU64,
    freed: AtomicU64,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct SnapshotPool {
    shared: Arc<Shared>,
}

impl SnapshotPool {
    pub fn new(cfg: PoolConfig) -> Self {
        Self {
            shared: Arc::new(Shared {
                spares: AtomicUsize::new(cfg.spares),
                cfg,
                state: Mutex::new(State::default()),
                wake: Condvar::new(),
                from_spare: AtomicU64::new(0),
                fallback: AtomicU64::new(0),
                recycled: AtomicU64::new(0),
                touched: AtomicU64::new(0),
                freed: AtomicU64::new(0),
            }),
        }
    }

    pub fn enabled(&self) -> bool {
        self.shared.spares.load(Ordering::Relaxed) > 0
    }

    /// Sets the number of spares unless `TETRAEDER_SNAPSHOT_SPARES` is set: one admission pass
    /// can save as many sequences as the scheduler runs in parallel.
    pub fn set_default_spares(&self, n: usize) {
        if std::env::var(SPARES_ENV).is_err() {
            self.shared.spares.store(n, Ordering::Relaxed);
            self.shared.wake.notify_all();
        }
    }

    /// Capacity the preparer aims for (0 before any hint or take).
    pub fn target(&self) -> usize {
        self.shared.lock().target
    }

    /// Caps spare buffers (each one and all together) at `bytes`: the prefix cache budget.
    pub fn set_max_bytes(&self, bytes: usize) {
        self.shared.lock().max_bytes = bytes;
        self.shared.wake.notify_all();
    }

    fn round(&self, n: usize) -> usize {
        n.next_multiple_of(self.shared.cfg.round.max(1))
    }

    /// Tells the pool a snapshot of about `bytes` will be saved later, so the preparer has a spare
    /// of that size ready by then. Cheap; call it as soon as the size is known, off the save path.
    pub fn expect(&self, bytes: usize) {
        if !self.enabled() {
            return;
        }
        let mut st = self.shared.lock();
        self.raise_target(&mut st, self.round(bytes));
        drop(st);
        self.shared.wake.notify_all();
    }

    /// Grows the target to `cap` unless the prefix cache would not keep a buffer that large, and
    /// starts the preparer thread on first use. Returns whether `cap` is worth preparing for.
    fn raise_target(&self, st: &mut State, cap: usize) -> bool {
        if st.max_bytes != 0 && cap > st.max_bytes {
            return false;
        }
        st.target = st.target.max(cap);
        if !st.started {
            st.started = true;
            let s = Arc::clone(&self.shared);
            if let Err(e) = std::thread::Builder::new()
                .name("snapshot-spares".into())
                .spawn(move || prepare_loop(&s))
            {
                tracing::warn!("cannot start the snapshot buffer thread: {e}");
            }
        }
        true
    }

    /// An empty buffer of capacity at least `need`, for a snapshot of about `need` bytes: a ready
    /// spare of at most about twice that size (the cache budget counts contents, not capacity),
    /// else the largest spare that is too small (the read grows it by realloc, which mostly
    /// extends the mapping, so only the tail pages fault), else a plain allocation (no advice,
    /// pages untouched). Also tells the preparer the size to keep ready and wakes it.
    pub fn take(&self, need: usize) -> Vec<u8> {
        let sh = &self.shared;
        if !self.enabled() {
            return Vec::with_capacity(need);
        }
        let cap = self.round(need);
        let mut st = sh.lock();
        let worth = self.raise_target(&mut st, cap);
        let limit = 2 * need + sh.cfg.round;
        let pick = |small: bool| {
            let spares = st.ready.iter().enumerate();
            if small {
                spares
                    .filter(|(_, v)| v.capacity() < need)
                    .max_by_key(|(_, v)| v.capacity())
                    .map(|(i, _)| i)
            } else {
                spares
                    .filter(|(_, v)| v.capacity() >= need && v.capacity() <= limit)
                    .min_by_key(|(_, v)| v.capacity())
                    .map(|(i, _)| i)
            }
        };
        let (i, source) = match (pick(false), pick(true)) {
            (Some(i), _) => (Some(i), "fit"),
            (None, Some(i)) => (Some(i), "small"),
            (None, None) => (None, "fallback"),
        };
        let spare = i.map(|i| st.ready.swap_remove(i));
        drop(st);
        sh.wake.notify_all();
        let mut v = match spare {
            Some(v) => {
                sh.from_spare.fetch_add(1, Ordering::Relaxed);
                v
            }
            None => {
                sh.fallback.fetch_add(1, Ordering::Relaxed);
                Vec::with_capacity(if worth { cap } else { need })
            }
        };
        if crate::qwen3::step_trace() {
            tracing::info!(target: "tetraeder::trace", need, cap = v.capacity(),
                touched = v.len(), source, "snapshot take");
        }
        v.clear();
        v
    }

    /// Takes a buffer nobody needs any more (an evicted or replaced snapshot) back while the
    /// pool has room, whatever its size (the preparer grows it to the target); the rest is
    /// freed on the preparer thread.
    pub fn recycle(&self, v: Vec<u8>) {
        let sh = &self.shared;
        if !self.enabled() || v.capacity() < sh.cfg.min_bytes {
            return;
        }
        let mut st = sh.lock();
        if st.has_room(sh.spares.load(Ordering::Relaxed), v.capacity()) {
            // Its contents are touched pages.
            st.ready.push(v);
            sh.recycled.fetch_add(1, Ordering::Relaxed);
        } else if st.started {
            st.discard.push(v);
        } else {
            drop(st);
            drop(v);
            return;
        }
        drop(st);
        sh.wake.notify_all();
    }

    pub fn stats(&self) -> PoolStats {
        let sh = &self.shared;
        let st = sh.lock();
        PoolStats {
            from_spare: sh.from_spare.load(Ordering::Relaxed),
            fallback: sh.fallback.load(Ordering::Relaxed),
            recycled: sh.recycled.load(Ordering::Relaxed),
            touched_bytes: sh.touched.load(Ordering::Relaxed),
            freed_bytes: sh.freed.load(Ordering::Relaxed),
            ready: st.ready.len(),
            ready_bytes: st.ready_bytes(),
        }
    }
}

impl Drop for SnapshotPool {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.wake.notify_all();
    }
}

/// The pool of this process, configured from the environment.
pub fn global() -> Arc<SnapshotPool> {
    static POOL: OnceLock<Arc<SnapshotPool>> = OnceLock::new();
    Arc::clone(POOL.get_or_init(|| {
        let cfg = PoolConfig::from_env();
        tracing::info!(spares = cfg.spares, thp = cfg.thp, "snapshot buffer pool");
        Arc::new(SnapshotPool::new(cfg))
    }))
}

/// Grows, touches and frees spares until shutdown, one job at a time, outside the lock.
fn prepare_loop(sh: &Shared) {
    loop {
        let job = {
            let mut st = sh.lock();
            loop {
                if st.shutdown {
                    return;
                }
                if let Some(j) = st.next_job(sh.spares.load(Ordering::Relaxed)) {
                    break j;
                }
                st = sh.wake.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        };
        match job {
            Job::Free(v) => {
                let bytes = v.capacity();
                let t = std::time::Instant::now();
                let pieces = release(v, FREE_PIECE);
                sh.freed.fetch_add(bytes as u64, Ordering::Relaxed);
                if crate::qwen3::step_trace() {
                    tracing::info!(target: "tetraeder::trace", bytes, pieces,
                        ms = t.elapsed().as_millis() as u64, "snapshot spare freed");
                }
            }
            Job::Fill(mut v, target) => {
                let t = std::time::Instant::now();
                let touched = fill(&mut v, target, sh.cfg.thp);
                sh.touched.fetch_add(touched as u64, Ordering::Relaxed);
                if crate::qwen3::step_trace() {
                    tracing::info!(target: "tetraeder::trace", bytes = touched,
                        cap = v.capacity(), ms = t.elapsed().as_millis() as u64,
                        "snapshot spare touched");
                }
                let mut st = sh.lock();
                st.filling = None;
                if st.shutdown {
                    return;
                }
                // Over the count or the budget by now (a recycle, a new budget)? The next job
                // frees the excess.
                st.ready.push(v);
                tracing::debug!(
                    bytes = target,
                    ready = st.ready.len(),
                    "snapshot spare ready"
                );
            }
        }
    }
}

/// Touches `v` up to `target` bytes (its length): grows the capacity by realloc, which moves
/// the touched pages along, and writes zeros over the new tail only. Returns the bytes touched.
fn fill(v: &mut Vec<u8>, target: usize, thp: bool) -> usize {
    let before = v.len();
    if before >= target {
        return 0;
    }
    if v.capacity() < target {
        v.reserve_exact(target - before);
    }
    if thp {
        advise_hugepages(v.as_ptr() as usize, v.capacity());
    }
    v.resize(target, 0);
    target - before
}

/// Frees `v` in pieces of `piece` bytes from its end, yielding between pieces, so no single
/// unmap runs long (glibc shrinks a mapped block in place by `mremap`, which unmaps the tail).
/// Elsewhere it is one free. Returns the number of pieces.
fn release(mut v: Vec<u8>, piece: usize) -> usize {
    let mut pieces = 1;
    if cfg!(target_os = "linux") && piece > 0 {
        v.clear();
        while v.capacity() > piece {
            v.shrink_to(v.capacity() - piece);
            pieces += 1;
            std::thread::yield_now();
        }
    }
    drop(v);
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const KB: usize = 1 << 10;

    fn pool(spares: usize) -> SnapshotPool {
        SnapshotPool::new(PoolConfig {
            spares,
            thp: false,
            min_bytes: 1,
            round: 4 * KB,
        })
    }

    fn wait_for(p: &SnapshotPool, what: &str, f: impl Fn(PoolStats) -> bool) -> PoolStats {
        let t = Instant::now();
        loop {
            let s = p.stats();
            if f(s) {
                return s;
            }
            assert!(t.elapsed() < Duration::from_secs(10), "{what}: {s:?}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// hardware: cpu. With nothing ready a save falls back to a plain buffer.
    #[test]
    fn falls_back_when_no_spare_is_ready() {
        let p = pool(2);
        let v = p.take(10 * KB);
        assert!(v.is_empty() && v.capacity() >= 10 * KB);
        let s = p.stats();
        assert_eq!((s.from_spare, s.fallback), (0, 1));
    }

    /// hardware: cpu. The preparer fills to the number of spares; a take uses one and the
    /// preparer refills.
    #[test]
    fn the_preparer_fills_to_k_and_refills() {
        let p = pool(2);
        drop(p.take(10 * KB));
        let s = wait_for(&p, "fill", |s| s.ready == 2);
        assert!(s.ready_bytes >= 2 * 12 * KB);
        let v = p.take(10 * KB);
        assert!(v.is_empty() && v.capacity() >= 12 * KB);
        assert_eq!(p.stats().from_spare, 1);
        wait_for(&p, "refill", |s| s.ready == 2);
    }

    /// hardware: cpu. The target follows the largest snapshot; spares grow with it and none is
    /// freed (ttft6-20261119: replacing them freed 2.6 to 8 GB at a time, overlapping steps).
    #[test]
    fn the_target_grows_and_spares_grow_with_it() {
        let p = pool(2);
        drop(p.take(10 * KB));
        wait_for(&p, "fill", |s| s.ready == 2);
        drop(p.take(100 * KB));
        let s = wait_for(&p, "regrow", |s| {
            s.ready == 2 && s.ready_bytes >= 2 * 100 * KB
        });
        assert_eq!(s.freed_bytes, 0, "{s:?}");
        // A somewhat smaller request is served by a bigger spare (the take above used a
        // too-small one).
        let v = p.take(60 * KB);
        assert!(v.is_empty() && v.capacity() >= 100 * KB);
        assert_eq!(p.stats().from_spare, 2);
    }

    /// hardware: cpu. A target a few percent higher touches only the new tails, not whole
    /// spares (ttft6-20261119: each new longest prompt re-touched every spare, 1.3 to 4 s each).
    #[test]
    fn a_small_rise_touches_only_the_tail() {
        let p = pool(2);
        p.expect(100 * KB);
        let s0 = wait_for(&p, "fill", |s| s.ready == 2 && s.ready_bytes >= 200 * KB);
        p.expect(104 * KB);
        let s = wait_for(&p, "grow", |s| s.ready == 2 && s.ready_bytes >= 208 * KB);
        assert_eq!(
            s.touched_bytes - s0.touched_bytes,
            2 * 4 * KB as u64,
            "{s:?}"
        );
        assert_eq!(s.freed_bytes, 0);
        let v = p.take(104 * KB);
        assert!(v.is_empty() && v.capacity() >= 104 * KB);
    }

    /// hardware: cpu. With no spare that fits, the largest too-small one is taken, not a plain
    /// allocation; a spare that is too large is never taken.
    #[test]
    fn a_too_small_spare_beats_a_fresh_allocation() {
        let p = pool(2);
        let small = |n: usize| Vec::<u8>::with_capacity(n);
        p.recycle(small(8 * KB));
        p.recycle(small(16 * KB));
        let v = p.take(64 * KB);
        assert_eq!(v.capacity(), 16 * KB, "the largest of the too-small spares");
        assert_eq!((p.stats().from_spare, p.stats().fallback), (1, 0));
    }

    /// hardware: cpu. A tiny snapshot does not pin a huge spare: it allocates plainly.
    #[test]
    fn a_spare_far_larger_than_the_snapshot_is_not_used() {
        let p = pool(2);
        drop(p.take(100 * KB));
        wait_for(&p, "fill", |s| s.ready == 2);
        let v = p.take(10 * KB);
        assert!(v.capacity() < 20 * KB);
        assert_eq!(p.stats().from_spare, 0);
        assert_eq!(p.stats().ready, 2);
    }

    /// hardware: cpu. A hint prepares spares before the first save, so that save is served by one.
    #[test]
    fn expect_prepares_before_the_first_take() {
        let p = pool(2);
        p.expect(50 * KB);
        wait_for(&p, "fill", |s| s.ready == 2);
        let v = p.take(50 * KB);
        assert!(v.capacity() >= 50 * KB);
        assert_eq!((p.stats().from_spare, p.stats().fallback), (1, 0));
        // Without spares a hint does nothing.
        let off = pool(0);
        off.expect(50 * KB);
        assert!(!off.shared.lock().started);
    }

    /// hardware: cpu. Released buffers come back up to the number of spares and are reused.
    #[test]
    fn recycle_is_capped_and_reused() {
        let p = pool(2);
        let bufs: Vec<Vec<u8>> = (0..3)
            .map(|_| {
                let mut v = Vec::with_capacity(64 * KB);
                v.extend_from_slice(&[7; 100]);
                v
            })
            .collect();
        let ptrs: Vec<_> = bufs.iter().map(|v| v.as_ptr()).collect();
        for v in bufs {
            p.recycle(v);
        }
        let s = p.stats();
        assert_eq!((s.recycled, s.ready), (2, 2));
        let v = p.take(64 * KB);
        assert!(v.is_empty(), "a recycled buffer comes back empty");
        assert!(ptrs[..2].contains(&v.as_ptr()));
        assert_eq!(p.stats().from_spare, 1);
    }

    /// hardware: cpu. Buffers below the minimum are not kept; a released buffer smaller than the
    /// target is kept and grown, not freed.
    #[test]
    fn small_buffers_go_and_smaller_released_ones_are_grown() {
        let p = SnapshotPool::new(PoolConfig {
            spares: 2,
            thp: false,
            min_bytes: 8 * KB,
            round: 4 * KB,
        });
        p.recycle(Vec::with_capacity(4 * KB));
        assert_eq!(p.stats().ready, 0);
        let mut v = Vec::with_capacity(16 * KB);
        v.resize(16 * KB, 1);
        p.recycle(v);
        assert_eq!(p.stats().recycled, 1);
        p.expect(100 * KB);
        let s = wait_for(&p, "grow", |s| s.ready == 2 && s.ready_bytes >= 200 * KB);
        assert_eq!(s.freed_bytes, 0, "{s:?}");
        assert_eq!(s.touched_bytes, (100 - 16 + 100) as u64 * KB as u64);
    }

    /// hardware: cpu. A request above the budget is served plainly and prepares nothing; a
    /// released buffer without room is freed by the preparer, not the caller.
    #[test]
    fn the_budget_caps_spares() {
        let p = pool(2);
        p.set_max_bytes(32 * KB);
        let v = p.take(100 * KB);
        assert_eq!(v.capacity(), 100 * KB);
        assert_eq!(p.stats().fallback, 1);
        drop(p.take(16 * KB));
        let s = wait_for(&p, "fill", |s| s.ready == 2);
        assert!(s.ready_bytes <= 32 * KB, "{s:?}");
        p.recycle(Vec::with_capacity(16 * KB));
        let s = wait_for(&p, "free", |s| s.freed_bytes == 16 * KB as u64);
        assert_eq!(s.ready, 2);
    }

    /// hardware: cpu. When the budget no longer holds every spare at the target, the smallest
    /// goes and the rest grow (ttft6-20261119: 4 spares, 8 GiB budget, snapshots of 5 GB).
    #[test]
    fn spares_over_the_budget_are_freed() {
        let p = pool(2);
        p.set_max_bytes(250 * KB);
        p.expect(100 * KB);
        wait_for(&p, "fill", |s| s.ready == 2 && s.ready_bytes >= 200 * KB);
        p.expect(200 * KB);
        let s = wait_for(&p, "shrink", |s| s.ready == 1 && s.ready_bytes >= 200 * KB);
        assert_eq!(s.freed_bytes, 100 * KB as u64, "{s:?}");
        assert!(s.ready_bytes <= 250 * KB);
    }

    /// hardware: cpu. Without spares the pool is today's `Vec::with_capacity` and nothing else.
    #[test]
    fn no_spares_is_a_plain_allocation() {
        let p = pool(0);
        let v = p.take(10 * KB);
        assert_eq!(v.capacity(), 10 * KB);
        p.recycle(Vec::with_capacity(64 * KB));
        assert_eq!(p.stats(), PoolStats::default());
        assert!(!p.shared.lock().started, "no thread");
    }

    /// hardware: cpu. A fill touches up to the target once, then only what the target grew by.
    #[test]
    fn fill_touches_only_the_tail() {
        let mut v = Vec::new();
        assert_eq!(fill(&mut v, 64 * KB, true), 64 * KB);
        assert_eq!((v.len(), v.capacity()), (64 * KB, 64 * KB));
        assert_eq!(fill(&mut v, 80 * KB, false), 16 * KB);
        assert!(v.len() == 80 * KB && v.capacity() >= 80 * KB);
        assert_eq!(fill(&mut v, 80 * KB, false), 0);
    }

    /// hardware: cpu. On Linux a free gives the memory back piece by piece.
    #[test]
    fn release_frees_in_pieces() {
        let mut v = Vec::with_capacity(100 * KB);
        v.resize(100 * KB, 3u8);
        let want = if cfg!(target_os = "linux") { 7 } else { 1 };
        assert_eq!(release(v, 16 * KB), want);
        assert_eq!(release(Vec::new(), 16 * KB), 1);
    }
}
