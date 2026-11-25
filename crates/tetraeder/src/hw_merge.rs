//! What a judged Case S session adds to the cluster graph that `cluster profile` writes (ADR 0091,
//! 0092): each node's measured RAM budget, the node graphs fitted from the anchors, and the end
//! probe's link curves. It replaces `scripts/phase6_exit_s_merge.py`; the result is parsed back
//! as a [`HardwareGraph`], so a schema change fails here and in CI, not on the box.
//!
//! Rule of [`merge_fitted`]: a fitted device `X` replaces the fields of the cluster device
//! `<node>/X` except its identity (`id`, `name`, `kind`, `tier`, `node`, `index`, `numa_nodes`,
//! `cores`) and, for a GPU, its capacity (the discovered one stays); a fitted disk goes to the
//! node's first disk device when no `<node>/<id>` exists. A fitted link replaces alpha and beta
//! of the cluster link of the same kind between the same two devices. A device or link without
//! a counterpart is reported and left out. Everything the fit did not touch is kept.

use serde_json::Value;
use std::collections::BTreeMap;
use tetraeder_profiler::HardwareGraph;

const KEEP: [&str; 8] = [
    "id",
    "name",
    "kind",
    "tier",
    "node",
    "index",
    "numa_nodes",
    "cores",
];

fn array<'a>(hw: &'a mut Value, key: &str) -> Result<&'a mut Vec<Value>, String> {
    match hw.get_mut(key) {
        Some(Value::Array(a)) => Ok(a),
        _ => Err(format!("the graph has no `{key}` list")),
    }
}

fn str_of<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}

/// Sets each node's CPU capacity to its measured RAM budget (ADR 0092 section 4). A node named
/// in `ram` without a CPU is an error; a node without a budget keeps its discovered capacity.
pub fn set_ram_budgets(hw: &mut Value, ram: &BTreeMap<String, u64>) -> Result<(), String> {
    let devices = array(hw, "devices")?;
    for (node, bytes) in ram {
        let mut hit = false;
        for d in devices.iter_mut() {
            if str_of(d, "kind") == Some("cpu") && str_of(d, "node") == Some(node) {
                d["capacity_bytes"] = Value::from(*bytes);
                hit = true;
            }
        }
        if !hit {
            return Err(format!("no CPU device on node `{node}` for its RAM budget"));
        }
    }
    Ok(())
}

/// Puts each fitted node graph into the cluster graph (the rule in the module docs); returns
/// what had no counterpart.
pub fn merge_fitted(
    hw: &mut Value,
    fitted: &BTreeMap<String, Value>,
) -> Result<Vec<String>, String> {
    let mut notes = Vec::new();
    for (node, f) in fitted {
        let disks: Vec<String> = array(hw, "devices")?
            .iter()
            .filter(|d| str_of(d, "node") == Some(node) && str_of(d, "kind") == Some("disk"))
            .filter_map(|d| str_of(d, "id").map(str::to_owned))
            .collect();
        let mut idmap: BTreeMap<String, Option<String>> = BTreeMap::new();
        let fitted_devices = f
            .get("devices")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for d in &fitted_devices {
            let fid = str_of(d, "id")
                .ok_or("a fitted device has no id")?
                .to_owned();
            let mut target = format!("{node}/{fid}");
            let devices = array(hw, "devices")?;
            let exists =
                |t: &str, devices: &Vec<Value>| devices.iter().any(|x| str_of(x, "id") == Some(t));
            if !exists(&target, devices) && str_of(d, "kind") == Some("disk") && !disks.is_empty() {
                target = disks[0].clone();
            }
            if !exists(&target, devices) {
                notes.push(format!("{node}: no device for fitted {fid}"));
                idmap.insert(fid, None);
                continue;
            }
            let dst = devices
                .iter_mut()
                .find(|x| str_of(x, "id") == Some(&target))
                .ok_or("device vanished")?;
            let is_gpu = str_of(dst, "kind") == Some("gpu");
            if let Some(obj) = d.as_object() {
                for (k, v) in obj {
                    if KEEP.contains(&k.as_str()) || (k == "capacity_bytes" && is_gpu) {
                        continue;
                    }
                    dst[k] = v.clone();
                }
            }
            idmap.insert(fid, Some(target));
        }
        let fitted_links = f
            .get("links")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for lk in &fitted_links {
            let side = |k: &str| {
                str_of(lk, k)
                    .and_then(|id| idmap.get(id))
                    .cloned()
                    .flatten()
            };
            let (a, b) = (side("a"), side("b"));
            let hit = match (&a, &b) {
                (Some(a), Some(b)) => array(hw, "links")?.iter_mut().find(|x| {
                    str_of(x, "kind") == str_of(lk, "kind") && {
                        let (xa, xb) = (str_of(x, "a"), str_of(x, "b"));
                        (xa == Some(a) && xb == Some(b)) || (xa == Some(b) && xb == Some(a))
                    }
                }),
                _ => None,
            };
            let Some(hit) = hit else {
                notes.push(format!(
                    "{node}: no link for fitted {}",
                    str_of(lk, "id").unwrap_or("?")
                ));
                continue;
            };
            for k in ["alpha_s", "beta_bytes_per_s"] {
                if let Some(v) = lk.get(k) {
                    hit[k] = v.clone();
                }
            }
        }
    }
    Ok(notes)
}

/// The graph with the probe's link curves and network links in place of its own; nothing else
/// changes (the end pricing of Case S).
pub fn swap_link(hw: &mut Value, probe: &Value) -> Result<(), String> {
    let curves = probe
        .get("curves")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let links = probe
        .get("links")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let node_of = |r: &Value| {
        r.as_str()
            .map(|s| s.split('/').next().unwrap_or(s).to_owned())
    };
    let nodes: Vec<String> = curves
        .iter()
        .filter_map(|c| c.get("ranks").and_then(Value::as_array))
        .flatten()
        .filter_map(node_of)
        .collect();
    let is_probe_curve = |c: &Value| {
        let ranks = c
            .get("ranks")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        str_of(c, "op") == Some("copy")
            && ranks.len() == 2
            && ranks.iter().filter_map(node_of).all(|n| nodes.contains(&n))
    };
    if let Ok(cs) = array(hw, "collectives") {
        cs.retain(|c| !is_probe_curve(c));
        cs.extend(curves);
    } else {
        hw["collectives"] = Value::Array(curves);
    }
    let ids: Vec<String> = links
        .iter()
        .filter_map(|x| str_of(x, "id").map(str::to_owned))
        .collect();
    let ls = array(hw, "links")?;
    ls.retain(|x| !str_of(x, "id").is_some_and(|i| ids.iter().any(|j| j == i)));
    ls.extend(links);
    Ok(())
}

/// The graph as the planner reads it: parses `hw` back as a [`HardwareGraph`].
pub fn check_reads(hw: &Value) -> Result<HardwareGraph, String> {
    HardwareGraph::from_json(&serde_json::to_string(hw).map_err(|e| e.to_string())?)
        .map_err(|e| format!("the merged graph is not a hardware graph: {e}"))
}

/// A fitted node graph file (TOML, as `tetraeder-plan segment-predict --hw-out` writes it) as JSON.
pub fn read_fitted(text: &str) -> Result<Value, String> {
    let t: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
    serde_json::to_value(t).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cluster() -> Value {
        json!({
            "devices": [
                {"id": "a/gpu0", "kind": "gpu", "node": "a", "name": "g", "capacity_bytes": 24, "bandwidth_bytes_per_s": 1},
                {"id": "a/cpu0", "kind": "cpu", "node": "a", "capacity_bytes": 100, "cores": 8},
                {"id": "a/nvme0", "kind": "disk", "node": "a", "capacity_bytes": 7, "bandwidth_bytes_per_s": 2},
                {"id": "b/cpu0", "kind": "cpu", "node": "b", "capacity_bytes": 100},
            ],
            "links": [
                {"id": "a/nvme-cpu0", "kind": "storage", "a": "a/nvme0", "b": "a/cpu0", "alpha_s": 1, "beta_bytes_per_s": 2},
                {"id": "net", "kind": "network", "a": "a/cpu0", "b": "b/cpu0", "alpha_s": 1, "beta_bytes_per_s": 2},
            ],
            "collectives": [{"op": "copy", "ranks": ["a/cpu0", "b/cpu0"], "points": [[1, 2]]}],
        })
    }

    #[test]
    fn the_budget_goes_to_the_nodes_cpu() {
        let mut hw = cluster();
        set_ram_budgets(&mut hw, &BTreeMap::from([("b".to_owned(), 59u64)])).unwrap();
        assert_eq!(hw["devices"][3]["capacity_bytes"], 59);
        assert_eq!(
            hw["devices"][1]["capacity_bytes"], 100,
            "a node without a budget keeps its capacity"
        );
        assert!(set_ram_budgets(&mut hw, &BTreeMap::from([("z".to_owned(), 1u64)])).is_err());
    }

    #[test]
    fn a_fitted_device_keeps_identity_and_a_gpus_capacity() {
        let mut hw = cluster();
        let fitted = read_fitted(
            "[[devices]]\nid = \"gpu0\"\nkind = \"gpu\"\ncapacity_bytes = 1\nbandwidth_bytes_per_s = 9\n\
             [[devices]]\nid = \"nvme0\"\nkind = \"disk\"\nbandwidth_bytes_per_s = 5\n\
             [[devices]]\nid = \"ghost\"\nkind = \"cpu\"\n\
             [[links]]\nid = \"x\"\nkind = \"storage\"\na = \"nvme0\"\nb = \"gpu0\"\nalpha_s = 3\n",
        )
        .unwrap();
        let notes = merge_fitted(&mut hw, &BTreeMap::from([("a".to_owned(), fitted)])).unwrap();
        assert_eq!(hw["devices"][0]["bandwidth_bytes_per_s"], 9);
        assert_eq!(hw["devices"][0]["capacity_bytes"], 24);
        assert_eq!(hw["devices"][0]["id"], "a/gpu0");
        assert_eq!(hw["devices"][2]["bandwidth_bytes_per_s"], 5);
        assert!(
            notes
                .iter()
                .any(|n| n.contains("no device for fitted ghost")),
            "{notes:?}"
        );
        assert!(
            notes.iter().any(|n| n.contains("no link for fitted x")),
            "{notes:?}"
        );
    }

    /// The ported case of the Python overlay: a fitted disk lands on the node's first disk when
    /// no device of its own id exists, a fitted CPU brings its measured RAM budget, and the
    /// fitted PCIe link replaces alpha and beta of the cluster's.
    #[test]
    fn a_fitted_disk_lands_on_the_first_disk_and_a_cpu_takes_its_budget() {
        let mut hw = json!({
            "devices": [
                {"id": "a/gpu0", "kind": "gpu", "node": "a", "capacity_bytes": 25, "bandwidth_bytes_per_s": 9},
                {"id": "a/cpu0", "kind": "cpu", "node": "a", "capacity_bytes": 64, "threads": 16},
                {"id": "a/disk0", "kind": "disk", "node": "a", "capacity_bytes": 1},
            ],
            "links": [
                {"id": "a/pcie", "a": "a/gpu0", "b": "a/cpu0", "kind": "pcie", "alpha_s": 1, "beta_bytes_per_s": 1},
                {"id": "net", "a": "a/cpu0", "b": "b/cpu0", "kind": "network", "alpha_s": 0.03},
            ],
        });
        let fitted = read_fitted(
            "[[devices]]\nid = \"gpu0\"\nkind = \"gpu\"\n\
             [[devices]]\nid = \"cpu0\"\nkind = \"cpu\"\ncapacity_bytes = 59\nthreads = 12\n\
             [[devices]]\nid = \"nvme0\"\nkind = \"disk\"\nqueue_curve = [{ depth = 1, bytes_per_s = 1.0 }]\n\
             [[links]]\nid = \"p\"\nkind = \"pcie\"\na = \"gpu0\"\nb = \"cpu0\"\nalpha_s = 2.0\nbeta_bytes_per_s = 3.0\n",
        )
        .unwrap();
        let notes = merge_fitted(&mut hw, &BTreeMap::from([("a".to_owned(), fitted)])).unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(hw["devices"][1]["capacity_bytes"], 59);
        assert_eq!(hw["devices"][1]["threads"], 12);
        assert_eq!(hw["devices"][2]["queue_curve"][0]["depth"], 1);
        assert_eq!(hw["links"][0]["alpha_s"], 2.0);
        assert_eq!(
            hw["links"][1]["alpha_s"], 0.03,
            "the network link is untouched"
        );
    }

    #[test]
    fn the_end_probe_replaces_only_the_link() {
        let mut hw = cluster();
        let probe = json!({
            "curves": [{"op": "copy", "ranks": ["a/cpu0", "b/cpu0"], "points": [[1, 9]]}],
            "links": [{"id": "net", "kind": "network", "a": "a/cpu0", "b": "b/cpu0", "alpha_s": 5, "beta_bytes_per_s": 6}],
        });
        swap_link(&mut hw, &probe).unwrap();
        assert_eq!(hw["collectives"].as_array().unwrap().len(), 1);
        assert_eq!(hw["collectives"][0]["points"][0][1], 9);
        assert_eq!(hw["links"].as_array().unwrap().len(), 2);
        assert_eq!(hw["links"][1]["alpha_s"], 5);
        assert_eq!(hw["devices"].as_array().unwrap().len(), 4);
    }

    /// hardware: none. The real rehearsal graph with a real probe swapped in reads as the planner
    /// reads it, with the probe's own curve (the feasibility test of the planner does the same
    /// swap by hand, `cluster_discovered_ram_experts.rs`).
    #[test]
    fn a_rehearsal_probe_swaps_into_the_rehearsal_graph() {
        let mut hw: Value = serde_json::from_str(include_str!(
            "../../planner/fixtures/phase6-rehearsal/hw.json"
        ))
        .unwrap();
        let probe: Value = serde_json::from_str(include_str!(
            "../../planner/fixtures/phase6-rehearsal/link-probes/probe-start.json"
        ))
        .unwrap();
        swap_link(&mut hw, &probe).unwrap();
        let g = check_reads(&hw).unwrap();
        let c = g.network_curve("a", "b").unwrap();
        assert_eq!(c.points.len(), 6);
        assert_eq!(c.points[0].0, 4096.0);
        assert!((c.points[5].1 - 1.981).abs() < 0.01, "{:?}", c.points);
    }
}
