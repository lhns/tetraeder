//! Spare buffers for KV snapshots (ADR 0074). A snapshot is tens of MB, above the allocator's
//! mmap threshold, so a fresh `Vec` page-faults on every byte of the save, with every live
//! sequence stalled. A background thread keeps a few buffers ready (allocated, advised for huge
//! pages, every page touched); a save takes one, and the prefix cache hands released buffers
//! back. When none is ready a save allocates plainly, without any advice: advised memory can
//! stall the faulting thread in direct compaction.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use tetraeder_ggml_sys::ggml::advise_hugepages;

/// Env: spare buffers kept ready (default 2; 0 disables the pool).
pub const SPARES_ENV: &str = "TETRAEDER_SNAPSHOT_SPARES";
/// Env: `0` skips the huge-page advice in the preparer (default on).
pub const THP_ENV: &str = "TETRAEDER_SNAPSHOT_THP";

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
            thp: true,
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
        c.thp = std::env::var(THP_ENV)
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
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
    pub ready: usize,
    pub ready_bytes: usize,
}

#[derive(Default)]
struct State {
    ready: Vec<Vec<u8>>,
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

    fn has_room(&self, spares: usize, cap: usize) -> bool {
        self.ready.len() < spares
            && (self.max_bytes == 0 || self.ready_bytes() + cap <= self.max_bytes)
    }
}

struct Shared {
    cfg: PoolConfig,
    state: Mutex<State>,
    wake: Condvar,
    from_spare: AtomicU64,
    fallback: AtomicU64,
    recycled: AtomicU64,
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
                cfg,
                state: Mutex::new(State::default()),
                wake: Condvar::new(),
                from_spare: AtomicU64::new(0),
                fallback: AtomicU64::new(0),
                recycled: AtomicU64::new(0),
            }),
        }
    }

    pub fn enabled(&self) -> bool {
        self.shared.cfg.spares > 0
    }

    /// Caps spare buffers (each one and all together) at `bytes`: the prefix cache budget.
    pub fn set_max_bytes(&self, bytes: usize) {
        self.shared.lock().max_bytes = bytes;
        self.shared.wake.notify_all();
    }

    fn round(&self, n: usize) -> usize {
        n.next_multiple_of(self.shared.cfg.round.max(1))
    }

    /// An empty buffer of capacity at least `need`, for a snapshot of about `need` bytes: a ready
    /// spare, else a plain allocation (no advice, pages untouched). Also tells the preparer the
    /// size to keep ready and wakes it.
    pub fn take(&self, need: usize) -> Vec<u8> {
        let sh = &self.shared;
        if !self.enabled() {
            return Vec::with_capacity(need);
        }
        let cap = self.round(need);
        let mut st = sh.lock();
        if st.max_bytes != 0 && cap > st.max_bytes {
            // The prefix cache would not keep it; nothing to prepare.
            drop(st);
            sh.fallback.fetch_add(1, Ordering::Relaxed);
            return Vec::with_capacity(need);
        }
        st.target = st.target.max(cap);
        let pick = st
            .ready
            .iter()
            .enumerate()
            .filter(|(_, v)| v.capacity() >= need)
            .min_by_key(|(_, v)| v.capacity())
            .map(|(i, _)| i);
        let spare = pick.map(|i| st.ready.swap_remove(i));
        let target = st.target;
        if !st.started {
            st.started = true;
            let s = Arc::clone(sh);
            if let Err(e) = std::thread::Builder::new()
                .name("snapshot-spares".into())
                .spawn(move || prepare_loop(&s))
            {
                tracing::warn!("cannot start the snapshot buffer thread: {e}");
            }
        }
        drop(st);
        sh.wake.notify_all();
        match spare {
            Some(v) => {
                sh.from_spare.fetch_add(1, Ordering::Relaxed);
                v
            }
            None => {
                sh.fallback.fetch_add(1, Ordering::Relaxed);
                Vec::with_capacity(target)
            }
        }
    }

    /// Takes a buffer nobody needs any more (an evicted or replaced snapshot) back, up to the
    /// number of spares; the rest is freed. Only buffers as large as the target are kept.
    pub fn recycle(&self, mut v: Vec<u8>) {
        let sh = &self.shared;
        if !self.enabled() || v.capacity() < sh.cfg.min_bytes {
            return;
        }
        v.clear();
        let mut st = sh.lock();
        if v.capacity() >= st.target && st.has_room(sh.cfg.spares, v.capacity()) {
            st.ready.push(v);
            sh.recycled.fetch_add(1, Ordering::Relaxed);
        } else {
            // Freed outside the lock.
            drop(st);
            drop(v);
        }
    }

    pub fn stats(&self) -> PoolStats {
        let sh = &self.shared;
        let st = sh.lock();
        PoolStats {
            from_spare: sh.from_spare.load(Ordering::Relaxed),
            fallback: sh.fallback.load(Ordering::Relaxed),
            recycled: sh.recycled.load(Ordering::Relaxed),
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

/// Keeps the ready list at `spares` buffers of the target capacity until shutdown.
fn prepare_loop(sh: &Shared) {
    loop {
        let cap = {
            let mut st = sh.lock();
            let stale = loop {
                if st.shutdown {
                    return;
                }
                let target = st.target;
                // Buffers smaller than the target will not fit the next save.
                let (keep, stale): (Vec<_>, Vec<_>) = std::mem::take(&mut st.ready)
                    .into_iter()
                    .partition(|v| v.capacity() >= target);
                st.ready = keep;
                if target > 0 && st.has_room(sh.cfg.spares, target) {
                    break stale;
                }
                if !stale.is_empty() {
                    drop(st);
                    drop(stale);
                    st = sh.lock();
                    continue;
                }
                st = sh.wake.wait(st).unwrap_or_else(|e| e.into_inner());
            };
            let target = st.target;
            drop(st);
            drop(stale);
            target
        };
        let v = prepare(cap, sh.cfg.thp);
        let mut st = sh.lock();
        if st.shutdown {
            return;
        }
        if v.capacity() >= st.target && st.has_room(sh.cfg.spares, v.capacity()) {
            st.ready.push(v);
            tracing::debug!(bytes = cap, ready = st.ready.len(), "snapshot spare ready");
        } else {
            drop(st);
            drop(v);
        }
    }
}

/// A buffer of `cap` bytes whose every page is already faulted in.
fn prepare(cap: usize, thp: bool) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(cap);
    if thp {
        advise_hugepages(v.as_ptr() as usize, v.capacity());
    }
    // Writing zeros touches every page; clearing keeps the capacity.
    v.resize(v.capacity(), 0);
    v.clear();
    v
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

    /// hardware: cpu. The target follows the largest snapshot; smaller spares are dropped.
    #[test]
    fn the_target_grows_and_stale_spares_go() {
        let p = pool(2);
        drop(p.take(10 * KB));
        wait_for(&p, "fill", |s| s.ready == 2);
        drop(p.take(100 * KB));
        wait_for(&p, "regrow", |s| {
            s.ready == 2 && s.ready_bytes >= 2 * 100 * KB
        });
        // A smaller request is served by a bigger spare.
        let v = p.take(10 * KB);
        assert!(v.capacity() >= 100 * KB);
        assert_eq!(p.stats().from_spare, 1);
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

    /// hardware: cpu. Buffers below the minimum or the target are not kept.
    #[test]
    fn small_and_stale_buffers_are_not_kept() {
        let p = SnapshotPool::new(PoolConfig {
            spares: 2,
            thp: false,
            min_bytes: 8 * KB,
            round: 4 * KB,
        });
        p.recycle(Vec::with_capacity(4 * KB));
        assert_eq!(p.stats().ready, 0);
        drop(p.take(100 * KB));
        wait_for(&p, "fill", |s| s.ready == 2);
        p.recycle(Vec::with_capacity(16 * KB));
        assert_eq!(p.stats().recycled, 0, "smaller than the target");
    }

    /// hardware: cpu. A request above the budget is served plainly and prepares nothing.
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
        assert_eq!(p.stats().ready, 2);
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

    /// hardware: cpu. Prepared buffers are empty and keep their capacity through a use.
    #[test]
    fn prepared_buffers_are_empty() {
        let v = prepare(64 * KB, true);
        assert!(v.is_empty() && v.capacity() == 64 * KB);
    }
}
