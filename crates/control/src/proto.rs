//! Messages of the node daemon (ADR 0091): one frame per call, the header is JSON, a payload
//! carries bulk (the hardware graph, a probe frame).

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use tetraeder_transport::frame::{self, Frame};
use tetraeder_transport::{Conn, NodeId, Purpose};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum NodeRequest {
    Hello {
        cluster: String,
    },
    Heartbeat {
        n: u64,
    },
    /// The node's hardware graph (profiler JSON) as the reply payload.
    Profile,
    /// Start a worker for a segment; `plan` is the segment's single-node plan TOML.
    Assign {
        replica: u32,
        segment: u32,
        plan: String,
        model_sha256: String,
    },
    /// [`NodeRequest::Assign`] naming the node that leads the replica: besides the head, only it
    /// may connect to the segment's worker (with plain `Assign` only the head may).
    AssignLed {
        replica: u32,
        segment: u32,
        plan: String,
        model_sha256: String,
        leader: NodeId,
    },
    /// A replica engine is released with `segment` `u32::MAX`.
    Release {
        replica: u32,
        segment: u32,
    },
    /// Start the replica engine (`tetraeder replica`) this node leads; `plan` is the replica's
    /// plan TOML (empty for a whole model on one node) and `segments` the nodes of its
    /// segments, in order, when it spans nodes (the head assigned them already). Starting an
    /// engine that already runs with the same arguments changes nothing; `Assign` always starts
    /// the worker afresh.
    StartReplica {
        replica: u32,
        plan: String,
        model_sha256: String,
        #[serde(default)]
        segments: Vec<String>,
        /// The planner's decode step time of each segment, seconds (the residual's baseline).
        #[serde(default)]
        predicted_s: Vec<f64>,
    },
    /// The payload is `bytes` long; the reply is empty (link probe).
    Probe {
        bytes: u64,
    },
    /// Time frames from this node to `peer` through its own transport: for each size the median
    /// of `repeats` round trips, less half the empty round trip. `buffers` asks for explicit
    /// socket buffers (send, receive); `None` leaves the kernel's autotuning.
    ProbePeer {
        peer: NodeId,
        sizes: Vec<u64>,
        repeats: u32,
        #[serde(default)]
        buffers: Option<(usize, usize)>,
    },
    Drain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum NodeReply {
    Hello {
        node: NodeId,
        version: String,
    },
    Alive {
        n: u64,
        workers: u32,
        #[serde(default)]
        draining: bool,
    },
    /// The hardware graph is the payload.
    Profile,
    Assigned,
    Started,
    Released,
    Probe,
    /// One-way seconds by frame bytes, the empty round trip, and the socket buffers of the
    /// measuring connection when the transport knows them.
    Curve {
        points: Vec<(f64, f64)>,
        empty_rtt_s: f64,
    },
    Error {
        msg: String,
    },
}

/// A request and its payload as one frame.
pub fn send<W: Write, T: Serialize>(tx: &mut W, msg: &T, payload: &[u8]) -> std::io::Result<()> {
    frame::write_frame(tx, msg, payload).map_err(std::io::Error::other)
}

/// The next frame of a connection of `purpose`, within that purpose's size limit.
pub fn recv<R: Read>(rx: &mut R, purpose: &Purpose) -> std::io::Result<Option<Frame>> {
    frame::read_frame_max(rx, purpose.max_frame())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// The next request of an inbound connection of `purpose`, within its request limit.
pub fn recv_request<R: Read>(rx: &mut R, purpose: &Purpose) -> std::io::Result<Option<Frame>> {
    frame::read_frame_max(rx, purpose.max_request_frame())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// One call on the dialer's side: send, wait for the reply.
pub fn call(
    c: &mut Conn,
    req: &NodeRequest,
    payload: &[u8],
) -> std::io::Result<(NodeReply, Vec<u8>)> {
    send(&mut c.tx, req, payload)?;
    let f = recv(&mut c.rx, &c.purpose)?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "the node closed the link",
        )
    })?;
    let reply: NodeReply = frame::parse_header(&f)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok((reply, f.payload))
}
