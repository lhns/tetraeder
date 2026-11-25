//! The operator's side of the node daemons: `tetraeder cluster profile`, `cluster probe` and
//! `node --drain` (ADR 0091 step 3). Each call dials the node as the head on a `Probe` connection
//! of session 0, which the daemon treats as privileged for the head only, so the control
//! connection that carries the heartbeats is never shared with a tool.

use crate::cluster::ClusterConfig;
use crate::proto::{self, NodeReply, NodeRequest};
use serde::{Deserialize, Serialize};
use tetraeder_profiler::{
    CollectiveCurve, CollectiveOp, HardwareGraph, Link, LinkKind, Measurement, Source,
};
use tetraeder_transport::{NodeId, Peer, Purpose, Transport};

/// Frame sizes of the probe: 4 KiB to 64 MiB in powers of 4 (ADR 0091).
pub const PROBE_SIZES: [u64; 8] = [
    4 << 10,
    16 << 10,
    64 << 10,
    256 << 10,
    1 << 20,
    4 << 20,
    16 << 20,
    64 << 20,
];

/// One call to a node's daemon.
pub fn request(
    t: &dyn Transport,
    peer: &Peer,
    req: &NodeRequest,
    payload: &[u8],
) -> Result<(NodeReply, Vec<u8>), String> {
    let mut c = t
        .connect(peer, Purpose::Probe { session: 0 })
        .map_err(|e| format!("cannot reach node `{}` at {}: {e}", peer.id, peer.addr))?;
    let r = proto::call(&mut c, req, payload);
    (c.close)();
    match r.map_err(|e| format!("node `{}`: {e}", peer.id))? {
        (NodeReply::Error { msg }, _) => Err(format!("node `{}`: {msg}", peer.id)),
        ok => Ok(ok),
    }
}

fn peer_of(cfg: &ClusterConfig, id: &NodeId) -> Result<Peer, String> {
    cfg.peer(id).ok_or_else(|| format!("`{id}` is not a node"))
}

/// Most of anything a node's hardware graph may hold, and the largest size or rate it may claim
/// (1 PiB, 1 EB/s): a graph comes from another machine, and the planner trusts its numbers.
const MAX_ITEMS: usize = 256;
const MAX_QUANTITY: f64 = 1.0e18;

fn plausible(what: &str, v: f64) -> Result<(), String> {
    if v.is_finite() && (0.0..=MAX_QUANTITY).contains(&v) {
        Ok(())
    } else {
        Err(format!("{what} is {v}"))
    }
}

fn plausible_m(what: &str, m: &Measurement) -> Result<(), String> {
    plausible(what, m.value)?;
    plausible(&format!("{what} spread"), m.spread)
}

/// Refuses a hardware graph from a node that holds NaN, negative or absurd numbers or too many
/// items, before the planner prices anything with it. `validate` checks the structure only.
pub fn check_profile(g: &HardwareGraph) -> Result<(), String> {
    if g.devices.len() > MAX_ITEMS
        || g.links.len() > MAX_ITEMS
        || g.lane_groups.len() > MAX_ITEMS
        || g.collectives.len() > MAX_ITEMS
        || g.notes.len() > MAX_ITEMS
    {
        return Err(format!("more than {MAX_ITEMS} devices, links or curves"));
    }
    for d in &g.devices {
        let id = &d.id;
        if d.capacity_bytes as f64 > MAX_QUANTITY {
            return Err(format!("device {id}: capacity {} bytes", d.capacity_bytes));
        }
        if let Some(b) = &d.bandwidth_bytes_per_s {
            plausible_m(&format!("device {id}: bandwidth"), b)?;
        }
        if let Some(c) = &d.compute {
            for (n, v) in [
                ("flops_per_s", c.flops_per_s),
                ("c0_graph_s", c.c0_graph_s),
                ("c0_eager_s", c.c0_eager_s),
                ("expert_flops_per_s", c.expert_flops_per_s.unwrap_or(0.0)),
            ] {
                plausible(&format!("device {id}: {n}"), v)?;
            }
        }
        if let Some(v) = d.copy_bytes_per_s {
            plausible(&format!("device {id}: copy_bytes_per_s"), v)?;
        }
        if d.thread_curve.len() > MAX_ITEMS || d.queue_curve.len() > MAX_ITEMS {
            return Err(format!("device {id}: too many curve points"));
        }
        for t in &d.thread_curve {
            plausible(&format!("device {id}: thread flops_per_s"), t.flops_per_s)?;
            plausible(&format!("device {id}: thread bytes_per_s"), t.bytes_per_s)?;
        }
        for q in &d.queue_curve {
            plausible(&format!("device {id}: queue bytes_per_s"), q.bytes_per_s)?;
        }
    }
    for l in &g.links {
        plausible_m(&format!("link {}: alpha_s", l.id), &l.alpha_s)?;
        plausible_m(
            &format!("link {}: beta_bytes_per_s", l.id),
            &l.beta_bytes_per_s,
        )?;
    }
    for lg in &g.lane_groups {
        plausible(
            &format!("lane group {}: capacity", lg.id),
            lg.capacity_bytes_per_s,
        )?;
    }
    for c in &g.collectives {
        if c.points.len() > MAX_ITEMS || c.ranks.len() > MAX_ITEMS {
            return Err("a collective curve has too many points".into());
        }
        for (b, t) in &c.points {
            plausible("collective bytes", *b)?;
            plausible("collective seconds", *t)?;
        }
    }
    Ok(())
}

/// Asks every node for its hardware graph; the result is the merged graph (ids `<node>/<id>`,
/// `node` set), with the links and curves of `probe` when given.
pub fn profile_cluster(
    t: &dyn Transport,
    cfg: &ClusterConfig,
    probe: Option<&ProbeResult>,
) -> Result<HardwareGraph, String> {
    let mut graphs = Vec::new();
    for n in &cfg.nodes {
        let (r, payload) = request(t, &peer_of(cfg, &n.id)?, &NodeRequest::Profile, &[])?;
        if !matches!(r, NodeReply::Profile) {
            return Err(format!("node `{}`: unexpected reply {r:?}", n.id));
        }
        let text = String::from_utf8(payload)
            .map_err(|e| format!("node `{}`: graph is not UTF-8: {e}", n.id))?;
        let g = HardwareGraph::from_json(&text)
            .map_err(|e| format!("node `{}`: bad hardware graph: {e}", n.id))?;
        g.validate().map_err(|e| format!("node `{}`: {e}", n.id))?;
        if let Err(e) = check_profile(&g) {
            tetraeder_transport::metrics::rejected("bad_profile");
            return Err(format!("node `{}`: implausible hardware graph: {e}", n.id));
        }
        if g.device("cpu0").is_none() {
            return Err(format!("node `{}` reports no cpu0 device", n.id));
        }
        graphs.push((n.id.0.clone(), g));
    }
    let (links, curves) = probe.map_or((Vec::new(), Vec::new()), |p| {
        (p.links.clone(), p.curves.clone())
    });
    let hw = HardwareGraph::merge_cluster(&graphs, links, curves);
    hw.validate()?;
    Ok(hw)
}

/// What `cluster probe` writes: the curves and the links derived from them, and the socket
/// setting it ran with.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeResult {
    /// `auto`, or `send,recv` bytes asked for.
    pub buffers: String,
    pub links: Vec<Link>,
    pub curves: Vec<CollectiveCurve>,
    /// Round trip of an empty frame per ordered pair, `(from, to, seconds)`.
    pub empty_rtt_s: Vec<(String, String, f64)>,
}

/// Measures every ordered pair of nodes through the daemons' own transports: node `a` times
/// frames to node `b` (so each direction is the sender's view), median of `repeats`.
pub fn probe_cluster(
    t: &dyn Transport,
    cfg: &ClusterConfig,
    sizes: &[u64],
    repeats: u32,
    buffers: Option<(usize, usize)>,
) -> Result<ProbeResult, String> {
    let mut out = ProbeResult {
        buffers: buffers.map_or_else(|| "auto".to_owned(), |(s, r)| format!("{s},{r}")),
        ..ProbeResult::default()
    };
    for a in &cfg.nodes {
        for b in &cfg.nodes {
            if a.id == b.id {
                continue;
            }
            let (r, _) = request(
                t,
                &peer_of(cfg, &a.id)?,
                &NodeRequest::ProbePeer {
                    peer: b.id.clone(),
                    sizes: sizes.to_vec(),
                    repeats,
                    buffers,
                },
                &[],
            )?;
            let NodeReply::Curve {
                points,
                empty_rtt_s,
            } = r
            else {
                return Err(format!("node `{}`: unexpected reply {r:?}", a.id));
            };
            out.curves.push(CollectiveCurve {
                op: CollectiveOp::Copy,
                ranks: vec![format!("{}/cpu0", a.id), format!("{}/cpu0", b.id)],
                comm: "tls".into(),
                points,
            });
            out.empty_rtt_s
                .push((a.id.0.clone(), b.id.0.clone(), empty_rtt_s));
        }
    }
    out.links = links_from(&out.curves, repeats);
    Ok(out)
}

/// One `Network` link per node pair from its two directions: alpha is the mean small-frame
/// one-way time, beta the slower direction's slope over the two largest frames. The curves
/// stay the source of truth; the link is the prior used when a curve is absent.
fn links_from(curves: &[CollectiveCurve], samples: u32) -> Vec<Link> {
    let mut links = Vec::new();
    for c in curves {
        let (a, b) = (&c.ranks[0], &c.ranks[1]);
        if a >= b {
            continue;
        }
        let Some(back) = curves.iter().find(|o| o.ranks[0] == *b && o.ranks[1] == *a) else {
            continue;
        };
        let alpha = |c: &CollectiveCurve| c.points.first().map_or(0.0, |p| p.1);
        let beta = |c: &CollectiveCurve| match c.points.as_slice() {
            [.., (b0, t0), (b1, t1)] if t1 > t0 => (b1 - b0) / (t1 - t0),
            [.., (b1, t1)] if *t1 > 0.0 => b1 / t1,
            _ => 0.0,
        };
        let na = a.split('/').next().unwrap_or(a);
        let nb = b.split('/').next().unwrap_or(b);
        let m = |value: f64| Measurement {
            value,
            spread: 0.0,
            samples,
            source: Source::Measured,
        };
        links.push(Link {
            id: format!("net-{na}-{nb}"),
            a: a.clone(),
            b: b.clone(),
            kind: LinkKind::Network,
            alpha_s: m((alpha(c) + alpha(back)) / 2.0),
            beta_bytes_per_s: m(beta(c).min(beta(back))),
            lane_group: None,
        });
    }
    links
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(a: &str, b: &str, pts: &[(f64, f64)]) -> CollectiveCurve {
        CollectiveCurve {
            op: CollectiveOp::Copy,
            ranks: vec![format!("{a}/cpu0"), format!("{b}/cpu0")],
            comm: "tls".into(),
            points: pts.to_vec(),
        }
    }

    #[test]
    fn a_link_per_pair_from_both_directions() {
        let c = [
            curve("a", "b", &[(4096.0, 0.001), (1e6, 0.011), (2e6, 0.021)]),
            curve("b", "a", &[(4096.0, 0.003), (1e6, 0.011), (2e6, 0.041)]),
        ];
        let l = links_from(&c, 5);
        assert_eq!(l.len(), 1);
        assert_eq!((l[0].a.as_str(), l[0].b.as_str()), ("a/cpu0", "b/cpu0"));
        assert_eq!(l[0].kind, LinkKind::Network);
        assert!((l[0].alpha_s.value - 0.002).abs() < 1e-12);
        // The slower direction: 1e6 bytes in 0.030 s.
        assert!((l[0].beta_bytes_per_s.value - 1e6 / 0.030).abs() < 1.0);
    }
}
