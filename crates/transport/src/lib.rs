//! Transport between nodes (ADR 0091 step 1; interfaces in `docs/notes/phase6-interfaces.md`).
//!
//! One trait, [`Transport`], opens authenticated byte streams ([`Conn`]) to peers, one per
//! [`Purpose`]. Two implementations: [`TlsTransport`] (TCP + TLS 1.3, mutual authentication
//! against trust anchors, ADR 0023) and [`MemTransport`] (in-process pipes with injected faults,
//! for simulated nodes). The worker frame format ([`frame`]) runs over the halves of a `Conn`.
//!
//! The crate also holds the Phase 0 transport spike (`bench`, `proto`, `tls`; ADR 0023).

use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod bench;
pub mod frame;
pub mod identity;
pub mod mem;
pub mod metrics;
pub mod proto;
pub mod tls;
pub mod tls_transport;

pub use identity::{Anchors, Identity, node_id_of};
pub use mem::{LinkFault, MemTransport, Switchboard};
pub use metrics::LinkMetrics;
pub use rustls::pki_types::CertificateDer;
pub use tls_transport::{SocketBuffers, TlsTransport, effective_buffers};

/// Node id, the `tetraeder-node:<id>` SAN of its certificate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl NodeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// What a connection carries; one connection per purpose (no head-of-line blocking).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "purpose", rename_all = "snake_case")]
pub enum Purpose {
    Control,
    /// Head to a replica leader: `Generate` requests and their events.
    Replica {
        replica: u32,
    },
    /// Leader to a segment worker: today's backend calls.
    Segment {
        replica: u32,
        segment: u32,
    },
    /// Leader to leader: prefix snapshot fetch.
    Prefix,
    /// Link probe: bulk frames answered by empty ones, on a connection of their own so a
    /// 64 MiB frame never delays a heartbeat (ADR 0091 step 3).
    Probe {
        session: u32,
    },
}

impl Purpose {
    /// The `tetraeder.purpose` attribute of the link metrics at connection level. The ADR's
    /// finer labels (`event`, `logits`) are set by the callers that know the frame.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Replica { .. } => "request",
            Self::Segment { .. } => "hidden",
            Self::Prefix => "kv",
            Self::Probe { .. } => "probe",
        }
    }

    /// Largest frame a reader of this purpose accepts (`frame::read_frame_max`): a control or
    /// request connection never carries bulk, so a corrupt or hostile length cannot ask for
    /// a gigabyte there.
    pub fn max_frame(&self) -> u64 {
        match self {
            Self::Control => 1 << 20,
            Self::Replica { .. } => 16 << 20,
            // 64 MiB of payload and the header.
            Self::Probe { .. } => (64 << 20) + (1 << 20),
            Self::Segment { .. } | Self::Prefix => frame::MAX_FRAME,
        }
    }
}

#[derive(Debug, Clone)]
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
    pub close: Arc<dyn Fn() + Send + Sync>,
}

pub trait Transport: Send + Sync + 'static {
    /// Connects and authenticates; the peer's certificate must name `peer.id`.
    fn connect(&self, peer: &Peer, purpose: Purpose) -> std::io::Result<Conn>;
    /// Next authenticated inbound connection. Refused handshakes are counted
    /// (`tetraeder.cluster.tls.rejected`) and never returned.
    fn accept(&self) -> std::io::Result<Conn>;
    /// `tls` or `mem`, the `transport` attribute of the link metrics.
    fn name(&self) -> &'static str;
    /// This node's id (the `source` attribute of the link metrics).
    fn local(&self) -> &NodeId;
    /// Stops accepting: a blocked or later `accept` fails. Open connections stay.
    fn shutdown(&self);
}
