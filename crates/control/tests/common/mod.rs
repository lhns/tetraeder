//! Shared helpers of the node tests: a cluster config, in-process workers and the node graph.

#![allow(dead_code)]

use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tetraeder_control::ClusterConfig;
use tetraeder_control::node::{Assignment, Launcher, WorkerIo};
use tetraeder_profiler::HardwareGraph;
use tetraeder_transport::NodeId;
use tetraeder_transport::frame::{self, Frame};

pub fn id(s: &str) -> NodeId {
    NodeId::new(s)
}

/// A node graph with a `cpu0` and two GPUs.
pub fn node_graph_json() -> String {
    let g = HardwareGraph::from_toml(include_str!(
        "../../../costmodel/fixtures/two_3090_hoststaged.toml"
    ))
    .unwrap();
    g.to_json().unwrap()
}

/// A config of nodes `names` (the first is the head) at `127.0.0.1:<port>` of `ports`.
pub fn config(down_ms: u64, names: &[&str], ports: &[u16]) -> ClusterConfig {
    let mut t = format!(
        "cluster = \"sim\"\nhead = \"{}\"\nnode_down_after_ms = {down_ms}\n",
        names[0]
    );
    for (n, p) in names.iter().zip(ports) {
        t += &format!("[[nodes]]\nid = \"{n}\"\naddr = \"127.0.0.1:{p}\"\nanchor = \"{n}.pem\"\n");
    }
    ClusterConfig::parse(&t).unwrap()
}

/// Polls `f` until it holds or `within` passes; returns the time it took.
pub fn wait_until(within: Duration, mut f: impl FnMut() -> bool) -> Option<Duration> {
    let t0 = Instant::now();
    while t0.elapsed() < within {
        if f() {
            return Some(t0.elapsed());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    f().then(|| t0.elapsed())
}

pub struct ChanWriter(Sender<Vec<u8>>);

impl Write for ChanWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .send(buf.to_vec())
            .map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub struct ChanReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    dead: Arc<AtomicBool>,
}

impl Read for ChanReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.buf.is_empty() {
            if self.dead.load(Ordering::SeqCst) {
                return Ok(0);
            }
            match self.rx.recv_timeout(Duration::from_millis(10)) {
                Ok(b) => self.buf = b,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(0),
            }
        }
        let n = out.len().min(self.buf.len());
        out[..n].copy_from_slice(&self.buf[..n]);
        self.buf.drain(..n);
        Ok(n)
    }
}

fn pipe(dead: &Arc<AtomicBool>) -> (ChanWriter, ChanReader) {
    let (tx, rx) = mpsc::channel();
    (
        ChanWriter(tx),
        ChanReader {
            rx,
            buf: Vec::new(),
            dead: Arc::clone(dead),
        },
    )
}

/// A worker that answers every frame with a frame holding the same payload and the header
/// `{"echo": <n-th frame of this worker process>}`; one "process" per launch.
#[derive(Default)]
pub struct EchoLauncher {
    pub launches: AtomicUsize,
    /// The `dead` flag of the running worker: set it to crash it.
    pub current: Mutex<Option<Arc<AtomicBool>>>,
    /// Launches beyond this many fail (for the restart cap).
    pub fail_after: Option<usize>,
}

impl EchoLauncher {
    pub fn crash(&self) {
        if let Some(d) = self.current.lock().unwrap().as_ref() {
            d.store(true, Ordering::SeqCst);
        }
    }
}

impl Launcher for EchoLauncher {
    fn launch(&self, _a: &Assignment) -> io::Result<WorkerIo> {
        let n = self.launches.fetch_add(1, Ordering::SeqCst);
        if self.fail_after.is_some_and(|f| n >= f) {
            return Err(io::Error::other("no worker"));
        }
        let dead = Arc::new(AtomicBool::new(false));
        *self.current.lock().unwrap() = Some(Arc::clone(&dead));
        let ((in_tx, mut in_rx), (out_tx, out_rx)) = (pipe(&dead), pipe(&dead));
        std::thread::spawn(move || {
            let mut out = out_tx;
            let mut count = 0u32;
            while let Ok(Some(Frame { payload, .. })) = frame::read_frame(&mut in_rx) {
                count += 1;
                if frame::write_frame(&mut out, &serde_json::json!({ "echo": count }), &payload)
                    .is_err()
                {
                    break;
                }
            }
        });
        let d = Arc::clone(&dead);
        Ok(WorkerIo {
            stdin: Box::new(in_tx),
            stdout: Box::new(out_rx),
            kill: Arc::new(move || d.store(true, Ordering::SeqCst)),
        })
    }
}

/// A replica "process": `ReplicaServer` over pipes, one per launch, each over a fresh engine from
/// `make`. `crash` ends the running one as `kill -9` ends a process.
pub struct ReplicaLauncher {
    #[allow(clippy::type_complexity)]
    pub make: Box<dyn Fn() -> Arc<dyn tetraeder_gateway::engine::Engine> + Send + Sync>,
    pub cap: u32,
    pub launches: AtomicUsize,
    pub current: Mutex<Option<Arc<AtomicBool>>>,
    /// Prefix hooks of the replicas this launcher starts.
    #[allow(clippy::type_complexity)]
    pub kv: Mutex<Option<Arc<dyn Fn() -> tetraeder_control::replica::KvHooks + Send + Sync>>>,
    /// How long a launched "process" takes to load its engine before it reads a frame: the pipes
    /// are open at once, so what the daemon relays queues up, as with a real process.
    pub load: Mutex<Duration>,
}

impl ReplicaLauncher {
    pub fn new(
        cap: u32,
        make: impl Fn() -> Arc<dyn tetraeder_gateway::engine::Engine> + Send + Sync + 'static,
    ) -> Self {
        Self {
            make: Box::new(make),
            cap,
            launches: AtomicUsize::new(0),
            current: Mutex::new(None),
            kv: Mutex::new(None),
            load: Mutex::new(Duration::ZERO),
        }
    }

    pub fn crash(&self) {
        if let Some(d) = self.current.lock().unwrap().as_ref() {
            d.store(true, Ordering::SeqCst);
        }
    }
}

impl Launcher for ReplicaLauncher {
    fn launch(&self, _a: &Assignment) -> io::Result<WorkerIo> {
        self.launches.fetch_add(1, Ordering::SeqCst);
        let dead = Arc::new(AtomicBool::new(false));
        *self.current.lock().unwrap() = Some(Arc::clone(&dead));
        let ((in_tx, in_rx), (out_tx, out_rx)) = (pipe(&dead), pipe(&dead));
        let kv = self.kv.lock().unwrap().as_ref().map(|f| f());
        let server = tetraeder_control::replica::ReplicaServer::new((self.make)(), self.cap, kv);
        let load = *self.load.lock().unwrap();
        std::thread::spawn(move || {
            std::thread::sleep(load);
            server.serve(in_rx, out_tx);
        });
        let d = Arc::clone(&dead);
        Ok(WorkerIo {
            stdin: Box::new(in_tx),
            stdout: Box::new(out_rx),
            kill: Arc::new(move || d.store(true, Ordering::SeqCst)),
        })
    }
}
