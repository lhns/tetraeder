# Phase 6 interfaces (ADR 0091)

The interfaces Phase 6 touches (rule 1), as Rust signatures. Sketches: names and shapes are the decision; the implementing step may adjust details that no test pins and says so in the ADR. Everything here is additive; existing signatures not listed are unchanged.

## `crates/transport`

```rust
/// Node id, the `tetraeder-node:<id>` SAN of its certificate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub String);

/// What a connection carries; one connection per purpose (no head-of-line blocking).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "purpose", rename_all = "snake_case")]
pub enum Purpose {
    Control,
    /// Head to a replica leader: `Generate` requests and their events.
    Replica { replica: u32 },
    /// Leader to a segment worker: today's backend calls.
    Segment { replica: u32, segment: u32 },
    /// Leader to leader: prefix snapshot fetch.
    Prefix,
}

pub struct Peer {
    pub id: NodeId,
    /// Address to dial: `host:port` (on Vast the public IP and the mapped port).
    pub addr: String,
}

/// An authenticated, bidirectional byte stream, split so a reader thread and a writer
/// thread can run at once (as `RemoteBackend` drives a child's pipes).
pub struct Conn {
    pub peer: NodeId,
    pub purpose: Purpose,
    pub rx: Box<dyn std::io::Read + Send>,
    pub tx: Box<dyn std::io::Write + Send>,
    /// Closes both halves from any thread; a blocked read or write then fails at once.
    /// Membership calls it on every connection of a node that goes `Down`.
    pub close: std::sync::Arc<dyn Fn() + Send + Sync>,
}

pub trait Transport: Send + Sync + 'static {
    /// Connects and authenticates; the peer's certificate must name `peer.id`.
    fn connect(&self, peer: &Peer, purpose: Purpose) -> std::io::Result<Conn>;
    /// Next authenticated inbound connection. Refused handshakes are counted
    /// (`tetraeder.cluster.tls.rejected`) and never returned.
    fn accept(&self) -> std::io::Result<Conn>;
    /// `tls` or `mem`, the `transport` attribute of the link metrics.
    fn name(&self) -> &'static str;
}

/// TCP + TLS 1.3 (rustls, ring), mutual authentication.
pub struct TlsTransport { /* listener, client and server configs */ }
impl TlsTransport {
    pub fn new(identity: &Identity, anchors: &[CertificateDer<'static>], nodes: &[NodeId],
               listen: Option<&str>, buffers: SocketBuffers) -> anyhow::Result<Self>;
}
/// Kernel autotuning, or explicit sizes (Linux caps them at `net.core.{r,w}mem_max` and
/// turns autotuning off). The probe picks one per link; the effective sizes are read back
/// with `getsockopt` and logged.
pub enum SocketBuffers { Auto, Explicit { send: usize, recv: usize } }
pub fn effective_buffers(s: &std::net::TcpStream) -> std::io::Result<(usize, usize)>;
pub struct Identity { pub cert: CertificateDer<'static>, pub key: PrivateKeyDer<'static> }

/// In-process transport for simulated nodes; every link follows its fault script.
pub struct MemTransport { /* shared switchboard */ }
impl MemTransport {
    pub fn switchboard() -> std::sync::Arc<Switchboard>;
}
pub struct Switchboard { /* nodes, links, clock */ }
impl Switchboard {
    pub fn node(self: &std::sync::Arc<Self>, id: &NodeId) -> MemTransport;
    pub fn fault(&self, from: &NodeId, to: &NodeId, f: LinkFault);
}
pub enum LinkFault {
    None,
    /// Close every connection (a crashed process: the peer sees a reset).
    Close,
    /// Pass no bytes, close nothing (a silent node or a partition, one direction).
    Blackhole,
    /// Delay and bandwidth limit per frame.
    Slow { alpha: std::time::Duration, bytes_per_s: f64 },
}

// Moved from `tetraeder::remote_backend`, unchanged: the frame format.
pub mod frame {
    pub const MAX_FRAME: u64 = 1 << 30;
    pub struct Frame { pub header: Vec<u8>, pub payload: Vec<u8> }
    pub fn write_frame<W: std::io::Write, T: serde::Serialize>(w: &mut W, header: &T, payload: &[u8]) -> Result<(), String>;
    pub fn read_frame<R: std::io::Read>(r: &mut R) -> Result<Option<Frame>, String>;
}
```

## `crates/control`: cluster config and membership

```rust
/// `cluster.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct ClusterConfig {
    pub cluster: String,
    pub head: NodeId,
    /// 2000 (ADR 0091).
    #[serde(default = "two_seconds_ms")]
    pub node_down_after_ms: u64,
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeConfig {
    pub id: NodeId,
    /// Address others dial.
    pub addr: String,
    /// Address to bind, when it differs (NAT, Vast's mapped ports).
    pub listen: Option<String>,
    /// PEM: this node's self-signed certificate or the cluster CA (ADR 0091, owner question 2).
    pub anchor: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState { Joining, Up, Draining, Down }

pub enum NodeEvent {
    /// Any frame from the node, heartbeat replies included.
    Heard(NodeId),
    /// A call failed: reset, EOF, malformed frame, refused handshake.
    Failed(NodeId),
    Drain(NodeId),
}

/// Pure state machine; the caller supplies time, as `replan.rs` does.
pub struct Membership { /* per node: state, last heard */ }
impl Membership {
    pub fn new(cfg: &ClusterConfig) -> Self;
    pub fn on_event(&mut self, now: Instant, e: NodeEvent) -> Vec<(NodeId, NodeState)>;
    /// Nodes silent for `node_down_after` become `Down`.
    pub fn tick(&mut self, now: Instant) -> Vec<(NodeId, NodeState)>;
    pub fn state(&self, id: &NodeId) -> NodeState;
}

/// Picks a replica for each new request.
pub struct Router { /* per replica: health, outstanding, cap, prefix index */ }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicaId(pub u32);
impl Router {
    /// `caps[r]` is replica r's `n_seq_max`: no more requests are sent to it at once.
    pub fn new(caps: &[u32]) -> Self;
    /// An `Up` replica with room: first the one holding the longest prefix of `prompt`
    /// (live or cached), else the one with the fewest outstanding requests. `None`: the
    /// request waits at the router (and can still go to any replica).
    pub fn route(&mut self, prompt: &[TokenId]) -> Option<ReplicaId>;
    /// Prefix index, fed by the replicas' save and evict events.
    pub fn prefix_saved(&mut self, r: ReplicaId, key: PrefixKey, tokens: u32);
    pub fn prefix_evicted(&mut self, r: ReplicaId, key: PrefixKey);
    pub fn finished(&mut self, r: ReplicaId);
    /// A replica joins (loaded and probed) or leaves (a node of it went `Down`).
    pub fn set_up(&mut self, r: ReplicaId, up: bool);
}

// `replan.rs`: one new trigger.
pub enum Trigger { /* existing */ NodeChange { node: NodeId, state: NodeState } }
```

## Control messages (node daemon, `Purpose::Control`)

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum NodeRequest {
    Hello { cluster: String },
    Heartbeat { n: u64 },
    /// The node's hardware graph (profiler JSON) in the reply.
    Profile,
    /// Start a replica engine (`tetraeder replica`) led by this node; `plan` is the replica's plan TOML.
    StartReplica { replica: u32, plan: String, model_sha256: String },
    /// Start a worker for a segment; `plan` is the segment's single-node plan TOML.
    Assign { replica: u32, segment: u32, plan: String, model_sha256: String },
    Release { replica: u32, segment: u32 },
    /// Reply with an empty frame; the payload is `bytes` long (link probe).
    Probe { bytes: u64 },
    Drain,
}

#[derive(Serialize, Deserialize)]
pub enum NodeReply {
    Hello { node: NodeId, version: String },
    Alive { n: u64, workers: u32 },
    Profile,
    Started,
    Assigned,
    Released,
    Probe,
    Error { msg: String },
}
```

## Replica protocol (`Purpose::Replica`, `Purpose::Prefix`)

```rust
/// The gateway's `Engine` for a replica led by another node. Tokenization and templates
/// stay on the head; sampling and constraints run on the leader.
pub struct RemoteEngine { /* transport, peer, replica, vocab fetched once */ }
impl gateway::Engine for RemoteEngine { /* `generate` sends `Generate`, streams `GenEvent`s */ }

/// `GenerateRequest` on the wire: the constraint as its source, compiled on the leader,
/// because `TokenConstraint` is a trait object.
#[derive(Serialize, Deserialize)]
pub struct WireGenerate {
    pub model: String,
    pub prompt: Vec<TokenId>,
    pub params: SamplingParams,
    pub constraint: Option<ConstraintSpec>,
    pub trace: Option<String>,
}
#[derive(Serialize, Deserialize)]
pub enum ConstraintSpec { Grammar(String), JsonSchema(serde_json::Value), Regex(String) }
/// One frame per event: `GenEvent` (token id, probabilities when asked, finish, error),
/// a few bytes per generated token. Cancel is a `Cancel` frame on the same connection.

/// Leader to leader. Reply payload: the snapshot bytes (C5), or an empty `Gone` reply.
#[derive(Serialize, Deserialize)]
pub enum PrefixRequest { Fetch { key: PrefixKey } }

/// The fetch rule: fetch when `hop_time(bytes) + restore_s(bytes)` times `RESTORE_MIN_GAIN`
/// beats recomputing `tokens` at the replica's predicted prefill rate, else recompute.
pub fn fetch_beats_recompute(hw: &HardwareGraph, from: &str, to: &str, bytes: u64,
                             tokens: u32, predicted_pp_tps: f64) -> bool;
```

## Worker protocol (`remote_backend::wire`)

```rust
pub struct RemoteConfig {
    /* existing fields */
    /// Where the worker runs. `Child` is today's behavior.
    pub endpoint: Endpoint,
}
pub enum Endpoint {
    Child { program: PathBuf, args: Vec<OsString> },
    /// A worker on a node: connect with `Purpose::Segment`, after `Assign` on the control link.
    /// The node's daemon restarts its worker; the head reconnects and re-assigns when the node
    /// is `Up` again. The restart cap calls `on_give_up`, which for a node endpoint takes the
    /// replica out of the router and never exits the head.
    Node { transport: Arc<dyn Transport>, peer: Peer, replica: u32, segment: u32 },
}

pub enum Request {
    /* existing variants unchanged */
    Decode {
        inputs: Vec<WireInput>,
        /// W3C trace context of the step.
        #[serde(default)]
        trace: Option<String>,
    },
}

pub struct Info {
    /* existing fields */
    /// `None` for a whole model (today).
    #[serde(default)]
    pub segment: Option<SegmentInfo>,
}
pub struct SegmentInfo { pub embed: bool, pub head: bool, pub n_embd: usize, pub layers: (u32, u32) }
```

A `decode` to a segment without `embed` carries the hidden rows of every input token as its payload (f32, in input order); a reply from a segment without `head` carries the hidden rows of every input token and `spans` in rows.

## `crates/scheduler`: segment steps

```rust
/// What a step consumes and produces when the backend is a segment.
pub enum Feed<'a> { Tokens, Hidden(&'a [f32]) }
pub enum StepOut<'a> { Logits(Outputs<'a>), Hidden { data: &'a [f32], spans: &'a [Range<usize>] } }

/// A backend that runs part of the layers. `BatchBackend::decode` is
/// `step(inputs, Feed::Tokens)` on a segment with embed and head.
pub trait SegmentBackend: BatchBackend {
    fn segment(&self) -> SegmentInfo;
    fn step(&mut self, inputs: &[SeqInput<'_>], feed: Feed<'_>) -> Result<StepOut<'_>, BackendError>;
}
```

`tetraeder::PipelineBackend` holds one `RemoteBackend` per segment and implements `BatchBackend`: sequence calls go to every segment, `decode` walks the segments in order (micro-batches in flight for prefill, ADR 0077), and any segment's failure fails the step for all (ADR 0065 rollback): each worker advances its own KV when its segment succeeds, so on a later segment's failure the leader sends `seq_rm(seq, old_len)` for every input to the earlier segments; a segment that cannot roll back marks those sequences lost (`seq_lost`).

## `crates/profiler` and `crates/plan`

```rust
// profiler::Device
/// Node of this device; `None` is the local node (every single-node graph).
#[serde(default, skip_serializing_if = "Option::is_none")]
pub node: Option<String>,

impl HardwareGraph {
    /// Union of node graphs, ids prefixed `<node>/`, `node` set, plus network links and curves.
    pub fn merge_cluster(nodes: &[(String, HardwareGraph)], links: Vec<Link>,
                         curves: Vec<CollectiveCurve>) -> HardwareGraph;
    pub fn node_of(&self, device: &str) -> Option<&str>;
    /// The devices of one node with their original ids (for single-node `predict`).
    pub fn node_graph(&self, node: Option<&str>) -> HardwareGraph;
}

// plan::hints::Forbid
#[serde(default, skip_serializing_if = "Option::is_none")]
pub node: Option<String>,

// plan::segment
pub struct Segment {
    pub node: Option<String>,
    /// Stage indices of the replica; empty for a head-only segment.
    pub stages: Range<usize>,
    pub layers: Span,
    pub embed: bool,
    pub head: bool,
}
pub fn segments(plan: &Plan, replica: usize, hw: &HardwareGraph) -> Result<Vec<Segment>, Vec<String>>;
/// The single-node plan a node's worker runs: the segment's stages, device ids without the node prefix.
pub fn segment_plan(plan: &Plan, replica: usize, seg: &Segment) -> Plan;

// plan::validate: new issues
pub enum Issue { /* existing */
    StageSpansNodes { at: At, nodes: Vec<String> },
    ForbiddenNode { at: At, node: String },
}
```

## `crates/costmodel` and `crates/planner`

```rust
/// One network hop: one-way time of `bytes` from `from` to `to`. The probe curve
/// (`CollectiveOp::Copy`, ranks `[<a>/host, <b>/host]`, comm `tls`) if present, else
/// `alpha + bytes / beta` of the `Network` link.
pub fn hop_time(hw: &HardwareGraph, from: &str, to: &str, bytes: f64) -> Option<f64>;

pub struct SegmentFeatures { pub node: Option<String>, pub run: RunFeatures, pub embed: bool, pub head: bool }
pub struct ClusterFeatures {
    pub segments: Vec<SegmentFeatures>,
    /// Hidden bytes per token at each boundary, and the return bytes per output row.
    pub n_embd: u32,
    pub n_vocab: u32,
    pub prefill_micro_batches: u32,
}
pub struct ClusterPrediction { pub segments: Vec<Prediction>, pub hop_s_per_token: f64, pub pp_tps: f64, pub tg_tps: f64 }
pub fn predict_cluster(hw: &HardwareGraph, f: &ClusterFeatures, prompt: u32, gen_tokens: u32)
    -> Result<ClusterPrediction, PredictError>;

// planner
pub struct ClusterCandidate { pub id: String, pub plan: Plan, pub notes: Vec<String> }
pub fn enumerate_cluster(hw: &HardwareGraph, layout: &ModelLayout, hints: &Hints, phase: Phase)
    -> (Vec<ClusterCandidate>, Vec<(String, String)> /* pruned id, reason */);
```

## CLI

- `tetraeder node --cluster cluster.toml --id <id> [--key key.pem --cert cert.pem]`, `tetraeder node --drain`.
- `tetraeder cluster cert --id <id>` (run on the box; writes a key that never leaves it and a self-signed certificate; tooling for owner question 2).
- `tetraeder cluster profile --cluster cluster.toml -o hw.json`, `tetraeder cluster probe --cluster cluster.toml -o probe.json`.
- `tetraeder replica --plan p.toml` (started by the daemon; the engine of `serve` without HTTP), `tetraeder serve --cluster cluster.toml --plan p.toml`, `tetraeder-plan candidates --cluster`, `tetraeder-plan predict --cluster`.

## As built in step 2 (`cluster-planner`)

`crates/profiler`, `crates/plan`, `crates/costmodel` and `crates/planner` follow the signatures above except for the differences listed in the History of ADR 0091 (2026-11-23, step 2): `Device.os`, network links and probe curves between the nodes' `cpu0` devices (ranks are `[<a>/cpu0, <b>/cpu0]`, not `<node>/host`), `hop_time` on node names, `ClusterCandidate.prediction`, one stage per node, the head unit as a Head rule of its own on the leader. Step 3's `cluster probe` writes curves with those ranks.

## As built in step 3 (`cluster-discovery`)

`ClusterConfig`, `Membership`, the monitor, the daemon and the probe client are in `crates/control` (`cluster`, `membership`, `monitor`, `node`, `client`, `proto`); the differences from the sketches above are listed in the History of ADR 0091 (2026-11-24, step 3): `node_down_after_ms`, `Membership` takes a `Duration` clock and a `Hello` event, `NodeRequest::ProbePeer` and `NodeReply::Curve`, `Alive.draining`, `Purpose::Probe { session }` and `Purpose::max_frame`, `frame::read_frame_max`, a new connection for a `(peer, purpose)` replaces the old one, CLI `cluster profile|probe|watch`, `node --drain --id <node>` run with the head's key, and `node` takes `--cluster --key --cert` (the id is the certificate's).
