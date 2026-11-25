//! `cluster.toml` (ADR 0091, "Discovery and membership: static first"): the cluster name, the
//! head, `node_down_after`, and per node its id, the address others dial, an optional listen
//! address and the trust anchor (the node's pinned certificate, or the cluster CA).
//!
//! The file never holds keys: unknown keys are refused, and an anchor file that contains a
//! private key is refused. Node ids prefix device ids (`<node>/gpu0`), so they follow the
//! certificate rule and cannot hold `/`, be empty, be `.` or `..`, or repeat.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tetraeder_transport::identity::{certs_from_pem, check_node_id, node_id_of};
use tetraeder_transport::{CertificateDer, NodeId, Peer};

/// The monitor sleeps in steps of this length, so a period can run this much long.
pub const PAUSE_STEP: Duration = Duration::from_millis(5);

const DEFAULT_DOWN_AFTER_MS: u64 = 2000;

fn two_seconds_ms() -> u64 {
    DEFAULT_DOWN_AFTER_MS
}

/// `cluster.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterConfig {
    pub cluster: String,
    pub head: NodeId,
    /// A node silent for this long is `Down` (2000, ADR 0091).
    #[serde(default = "two_seconds_ms")]
    pub node_down_after_ms: u64,
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeConfig {
    pub id: NodeId,
    /// Address others dial (on Vast the public IP and the mapped port).
    pub addr: String,
    /// Address to bind, when it differs (NAT, mapped ports); `addr` otherwise.
    #[serde(default)]
    pub listen: Option<String>,
    /// PEM: this node's self-signed certificate or the cluster CA.
    pub anchor: PathBuf,
}

impl ClusterConfig {
    /// Parses and validates; anchor paths stay as written.
    pub fn parse(text: &str) -> Result<Self, String> {
        let cfg: Self = toml::from_str(text).map_err(|e| format!("cluster.toml: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reads `path`; relative anchor paths are taken relative to its directory.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let mut cfg = Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if let Some(dir) = path.parent() {
            for n in &mut cfg.nodes {
                if n.anchor.is_relative() {
                    n.anchor = dir.join(&n.anchor);
                }
            }
        }
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        if self.cluster.trim().is_empty() {
            return Err("`cluster` is empty".into());
        }
        if self.nodes.is_empty() {
            return Err("no nodes".into());
        }
        if !(200..=60_000).contains(&self.node_down_after_ms) {
            return Err(format!(
                "node_down_after_ms {} is outside 200 to 60000",
                self.node_down_after_ms
            ));
        }
        for (i, n) in self.nodes.iter().enumerate() {
            check_node_id(&n.id.0).map_err(|e| e.to_string())?;
            if self.nodes[..i].iter().any(|o| o.id == n.id) {
                return Err(format!("node `{}` is listed twice", n.id));
            }
            if !n.addr.contains(':') {
                return Err(format!(
                    "node `{}`: addr `{}` is not host:port",
                    n.id, n.addr
                ));
            }
            if n.listen.as_deref().is_some_and(|l| !l.contains(':')) {
                return Err(format!("node `{}`: listen is not host:port", n.id));
            }
        }
        if self.node(&self.head).is_none() {
            return Err(format!("head `{}` is not a listed node", self.head));
        }
        Ok(())
    }

    pub fn node_down_after(&self) -> Duration {
        Duration::from_millis(self.node_down_after_ms)
    }

    /// Heartbeats run every `node_down_after / 4`.
    pub fn heartbeat_every(&self) -> Duration {
        self.node_down_after() / 4
    }

    /// The monitor's ticker (membership check and stale-connection cut) runs every
    /// `heartbeat_every / 4`, at least [`PAUSE_STEP`].
    pub fn tick_every(&self) -> Duration {
        (self.heartbeat_every() / 4).max(PAUSE_STEP)
    }

    /// Worst case from the last frame heard from a node that went silent (a partition, a stopped
    /// process: no connection closes) to the head's calls on it failing: the node is `Down` at
    /// the first tick with `node_down_after` of silence, and that tick closes the replica and
    /// segment connections (ADR 0091, Case F mode 2). The fault comes just after a reply, so
    /// the silence starts at the fault.
    pub fn silent_detect_bound(&self) -> Duration {
        self.node_down_after() + self.tick_every() + PAUSE_STEP
    }

    pub fn node(&self, id: &NodeId) -> Option<&NodeConfig> {
        self.nodes.iter().find(|n| &n.id == id)
    }

    pub fn ids(&self) -> Vec<NodeId> {
        self.nodes.iter().map(|n| n.id.clone()).collect()
    }

    pub fn peer(&self, id: &NodeId) -> Option<Peer> {
        self.node(id).map(|n| Peer {
            id: n.id.clone(),
            addr: n.addr.clone(),
        })
    }

    /// The address node `id` binds.
    pub fn listen_addr(&self, id: &NodeId) -> Option<&str> {
        self.node(id)
            .map(|n| n.listen.as_deref().unwrap_or(&n.addr))
    }

    /// The trust anchors: every node's certificate or the CA. A certificate that names a node
    /// must name the node it is listed under, and no anchor file may hold a private key.
    pub fn anchors(&self) -> Result<Vec<CertificateDer<'static>>, String> {
        let mut out = Vec::new();
        for n in &self.nodes {
            let pem = std::fs::read(&n.anchor)
                .map_err(|e| format!("node `{}`: cannot read {}: {e}", n.id, n.anchor.display()))?;
            if String::from_utf8_lossy(&pem).contains("PRIVATE KEY") {
                return Err(format!(
                    "node `{}`: {} holds a private key; the anchor is the public certificate",
                    n.id,
                    n.anchor.display()
                ));
            }
            for c in certs_from_pem(&pem).map_err(|e| format!("node `{}`: {e:#}", n.id))? {
                if let Ok(named) = node_id_of(&c)
                    && named != n.id
                {
                    return Err(format!(
                        "node `{}`: {} is the certificate of `{named}`",
                        n.id,
                        n.anchor.display()
                    ));
                }
                out.push(c);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OK: &str = r#"
cluster = "lab"
head = "a"
[[nodes]]
id = "a"
addr = "10.0.0.1:7000"
anchor = "a.pem"
[[nodes]]
id = "b"
addr = "203.0.113.5:41000"
listen = "0.0.0.0:7000"
anchor = "b.pem"
"#;

    #[test]
    fn parses_and_defaults() {
        let c = ClusterConfig::parse(OK).unwrap();
        assert_eq!(c.node_down_after(), Duration::from_secs(2));
        assert_eq!(c.heartbeat_every(), Duration::from_millis(500));
        assert_eq!(c.listen_addr(&NodeId::new("a")), Some("10.0.0.1:7000"));
        assert_eq!(c.listen_addr(&NodeId::new("b")), Some("0.0.0.0:7000"));
        assert_eq!(c.peer(&NodeId::new("b")).unwrap().addr, "203.0.113.5:41000");
    }

    #[test]
    fn a_silent_node_is_cut_off_within_the_3_s_rule_at_the_default() {
        let c = ClusterConfig::parse(OK).unwrap();
        assert_eq!(c.tick_every(), Duration::from_millis(125));
        assert_eq!(
            c.silent_detect_bound(),
            Duration::from_millis(2000) + Duration::from_millis(125) + PAUSE_STEP
        );
        assert!(c.silent_detect_bound() < Duration::from_secs(3));
        // The shortest `node_down_after` the file allows.
        let f = ClusterConfig::parse(&OK.replace(
            "head = \"a\"",
            "head = \"a\"
node_down_after_ms = 200",
        ))
        .unwrap();
        assert_eq!(f.tick_every(), Duration::from_micros(12_500));
    }

    #[test]
    fn refuses_bad_files() {
        let bad = |s: &str| ClusterConfig::parse(&OK.replace("id = \"b\"", s)).unwrap_err();
        assert!(bad("id = \"a\"").contains("twice"));
        assert!(bad("id = \"..\"").contains("not a node id"));
        assert!(bad("id = \"x/y\"").contains("not a node id"));
        assert!(bad("id = \"\"").contains("not a node id"));
        assert!(
            ClusterConfig::parse(&OK.replace("head = \"a\"", "head = \"z\""))
                .unwrap_err()
                .contains("head")
        );
        assert!(
            ClusterConfig::parse(&format!("{OK}\nkey = \"k.pem\"\n"))
                .unwrap_err()
                .contains("unknown field")
        );
        assert!(
            ClusterConfig::parse(&OK.replace("addr = \"10.0.0.1:7000\"", "addr = \"10.0.0.1\""))
                .unwrap_err()
                .contains("host:port")
        );
        assert!(
            ClusterConfig::parse(
                &OK.replace("head = \"a\"", "head = \"a\"\nnode_down_after_ms = 5")
            )
            .unwrap_err()
            .contains("node_down_after_ms")
        );
    }
}
