//! Calibration of the cost model from anchor runs (ADR 0040, "Calibration from anchor runs").
//!
//! The fit sets, in log space: the GPU's F, B, `c0_graph`, `c0_eager`; the CPU's F and B at each
//! distinct thread count of the anchors that use the CPU (the thread curve) and its `c0`; the
//! PCIe `alpha` and `beta` of one link. The objective is
//!
//! ```text
//! sum over anchors and rates (ln predicted - ln measured)^2  +  1e-3 * sum_i (x_i - x0_i)^2
//! ```
//!
//! where `x0` is the prior. The weak pull keeps parameters no anchor exercises at their prior.
//! The minimizer is deterministic: Levenberg-Marquardt on the residual vector (numerical
//! Jacobian), then a coordinate search with a shrinking step to polish across the roofline kinks.
//! No randomness, no solver crate.
//! A parameter is written back (as `Source::Measured` where it has a source) only if it moved.

use crate::predict::{PredictError, RunFeatures, predict};
use serde::{Deserialize, Serialize};
use std::fmt;
use tetraeder_profiler::{HardwareGraph, Measurement, Source, ThreadPoint};

/// One measured run: its features and the llama-bench rates (either may be absent).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    pub id: String,
    pub features: RunFeatures,
    pub prompt: u32,
    pub r#gen: u32,
    pub pp_tok_s: Option<f64>,
    pub tg_tok_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    /// The prior with fitted values written back and the thread curve filled.
    pub hw: HardwareGraph,
    /// `(anchor id + ":pp" or ":tg", ln predicted - ln measured)`.
    pub residuals: Vec<(String, f64)>,
    /// `exp(max |residual|) - 1`, at least 0.05.
    pub band: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalibError {
    NoAnchors,
    MissingSpec(String),
    BadAnchor(String),
    Predict { anchor: String, err: PredictError },
}

impl fmt::Display for CalibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAnchors => write!(f, "no anchor has a measured rate"),
            Self::MissingSpec(s) => write!(f, "prior lacks {s}"),
            Self::BadAnchor(s) => write!(f, "anchor {s} has a non-positive rate"),
            Self::Predict { anchor, err } => write!(f, "anchor {anchor}: {err}"),
        }
    }
}

impl std::error::Error for CalibError {}

const LAMBDA: f64 = 1e-3;
const MIN_BAND: f64 = 0.05;

const GPU_F: usize = 0;
const GPU_B: usize = 1;
const GPU_C0_GRAPH: usize = 2;
const GPU_C0_EAGER: usize = 3;
const CPU_C0: usize = 4;
const PCIE_ALPHA: usize = 5;
const PCIE_BETA: usize = 6;
const CURVE: usize = 7;

struct Problem<'a> {
    prior: &'a HardwareGraph,
    gpu: &'a str,
    cpu: &'a str,
    link: &'a str,
    /// Distinct thread counts of anchors with a CPU part; parameters `CURVE + 2k`, `CURVE + 2k + 1`.
    threads: Vec<u32>,
    x0: Vec<f64>,
}

impl Problem<'_> {
    /// The prior with parameters `x` written in. Unchanged parameters keep their prior value.
    fn apply(&self, x: &[f64]) -> HardwareGraph {
        let mut hw = self.prior.clone();
        let moved = |i: usize| x[i] != self.x0[i];
        let measured = |m: Measurement, i: usize| {
            if moved(i) {
                Measurement {
                    value: x[i].exp(),
                    source: Source::Measured,
                    ..m
                }
            } else {
                m
            }
        };
        let val = |v: f64, i: usize| if moved(i) { x[i].exp() } else { v };
        for d in &mut hw.devices {
            if d.id == self.gpu {
                if let Some(c) = &mut d.compute {
                    c.flops_per_s = val(c.flops_per_s, GPU_F);
                    c.c0_graph_s = val(c.c0_graph_s, GPU_C0_GRAPH);
                    c.c0_eager_s = val(c.c0_eager_s, GPU_C0_EAGER);
                }
                if let Some(b) = &mut d.bandwidth_bytes_per_s {
                    *b = measured(*b, GPU_B);
                }
            } else if d.id == self.cpu {
                if let Some(c) = &mut d.compute {
                    c.c0_graph_s = val(c.c0_graph_s, CPU_C0);
                    c.c0_eager_s = val(c.c0_eager_s, CPU_C0);
                }
                if !self.threads.is_empty() {
                    d.thread_curve
                        .retain(|p| !self.threads.contains(&p.threads));
                    for (k, &t) in self.threads.iter().enumerate() {
                        d.thread_curve.push(ThreadPoint {
                            threads: t,
                            flops_per_s: x[CURVE + 2 * k].exp(),
                            bytes_per_s: x[CURVE + 2 * k + 1].exp(),
                        });
                    }
                    d.thread_curve.sort_by_key(|p| p.threads);
                }
            }
        }
        for l in &mut hw.links {
            if l.id == self.link {
                l.alpha_s = measured(l.alpha_s, PCIE_ALPHA);
                l.beta_bytes_per_s = measured(l.beta_bytes_per_s, PCIE_BETA);
            }
        }
        hw
    }
}

fn residuals(hw: &HardwareGraph, anchors: &[Anchor]) -> Result<Vec<(String, f64)>, CalibError> {
    let mut out = Vec::new();
    for a in anchors {
        let p = predict(hw, &a.features, a.prompt, a.r#gen).map_err(|err| CalibError::Predict {
            anchor: a.id.clone(),
            err,
        })?;
        if let Some(m) = a.pp_tok_s {
            out.push((format!("{}:pp", a.id), (p.pp_tok_s / m).ln()));
        }
        if let Some(m) = a.tg_tok_s {
            out.push((format!("{}:tg", a.id), (p.tg_tok_s / m).ln()));
        }
    }
    Ok(out)
}

/// Solves `a x = b` by Gaussian elimination with partial pivoting; `None` when singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-300 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let k = a[r][c] / a[c][c];
            for j in c..n {
                a[r][j] -= k * a[c][j];
            }
            b[r] -= k * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|j| a[r][j] * x[j]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

/// Levenberg-Marquardt on the residual vector with a central-difference Jacobian. The data
/// residuals are nearly linear in log space but strongly correlated (c0 against B), which
/// stalls a pure coordinate search.
fn levenberg_marquardt(res: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64]) -> Vec<f64> {
    let n = x0.len();
    let cost = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();
    let mut x = x0.to_vec();
    let mut r = res(&x);
    let mut c = cost(&r);
    let mut mu = 1e-2;
    for _ in 0..300 {
        let h = 1e-5;
        let cols: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let (mut hi, mut lo) = (x.clone(), x.clone());
                hi[i] += h;
                lo[i] -= h;
                let (rh, rl) = (res(&hi), res(&lo));
                rh.iter()
                    .zip(&rl)
                    .map(|(a, b)| (a - b) / (2.0 * h))
                    .collect()
            })
            .collect();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
        let g: Vec<f64> = cols.iter().map(|j| dot(j, &r)).collect();
        let mut accepted = false;
        while mu < 1e12 {
            let a: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let v = dot(&cols[i], &cols[j]);
                            if i == j { v + mu * v.max(1e-12) } else { v }
                        })
                        .collect()
                })
                .collect();
            let Some(d) = solve(a, g.iter().map(|v| -v).collect()) else {
                mu *= 10.0;
                continue;
            };
            let xn: Vec<f64> = x.iter().zip(&d).map(|(a, b)| a + b).collect();
            let rn = res(&xn);
            let cn = cost(&rn);
            if cn < c {
                let tiny = c - cn < 1e-18 * c.max(1e-30);
                (x, r, c) = (xn, rn, cn);
                mu = (mu * 0.3).max(1e-9);
                accepted = !tiny;
                break;
            }
            mu *= 10.0;
        }
        if !accepted {
            break;
        }
    }
    x
}

/// Deterministic coordinate search with a shrinking step and line extension; polishes the result
/// of [`levenberg_marquardt`] against the kinks of the roofline `max`.
fn coordinate_search(f: &dyn Fn(&[f64]) -> f64, x0: &[f64]) -> Vec<f64> {
    let mut x = x0.to_vec();
    let mut best = f(&x);
    let mut step = 0.05;
    for _ in 0..10_000 {
        if step < 1e-7 {
            break;
        }
        let mut improved = false;
        for i in 0..x.len() {
            for dir in [1.0, -1.0] {
                let mut moved = false;
                loop {
                    let old = x[i];
                    x[i] = old + dir * step;
                    let v = f(&x);
                    if v < best - 1e-15 {
                        best = v;
                        moved = true;
                    } else {
                        x[i] = old;
                        break;
                    }
                }
                if moved {
                    improved = true;
                    break;
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    x
}

/// Fits `prior` to `anchors`; `gpu`, `cpu` and `link` name the device ids and the link id (in the
/// prior) whose parameters are fitted.
pub fn fit(
    prior: &HardwareGraph,
    anchors: &[Anchor],
    gpu: &str,
    cpu: &str,
    link: &str,
) -> Result<Fit, CalibError> {
    if !anchors
        .iter()
        .any(|a| a.pp_tok_s.is_some() || a.tg_tok_s.is_some())
    {
        return Err(CalibError::NoAnchors);
    }
    for a in anchors {
        if [a.pp_tok_s, a.tg_tok_s]
            .iter()
            .flatten()
            .any(|&v| !(v > 0.0 && v.is_finite()))
        {
            return Err(CalibError::BadAnchor(a.id.clone()));
        }
    }
    let spec = |what: &str| CalibError::MissingSpec(what.to_string());
    let g = prior.device(gpu).ok_or_else(|| spec(gpu))?;
    let c = prior.device(cpu).ok_or_else(|| spec(cpu))?;
    let l = prior
        .links
        .iter()
        .find(|l| l.id == link)
        .ok_or_else(|| spec(link))?;
    let gc = g.compute.ok_or_else(|| spec("gpu compute"))?;
    let gb = g
        .bandwidth_bytes_per_s
        .ok_or_else(|| spec("gpu bandwidth"))?;
    let cc = c.compute.ok_or_else(|| spec("cpu compute"))?;

    let mut threads: Vec<u32> = anchors
        .iter()
        .filter(|a| a.features.parts.iter().any(|p| p.device == cpu))
        .map(|a| a.features.threads)
        .collect();
    threads.sort_unstable();
    threads.dedup();

    let ln = |v: f64| v.max(1e-30).ln();
    let mut x0 = vec![
        ln(gc.flops_per_s),
        ln(gb.value),
        ln(gc.c0_graph_s),
        ln(gc.c0_eager_s),
        ln(cc.c0_eager_s),
        ln(l.alpha_s.value),
        ln(l.beta_bytes_per_s.value),
    ];
    for &t in &threads {
        let (f, b) = c
            .cpu_rates(t)
            .ok_or_else(|| spec("cpu flops or bandwidth"))?;
        x0.push(ln(f));
        x0.push(ln(b));
    }
    let prob = Problem {
        prior,
        gpu,
        cpu,
        link,
        threads,
        x0: x0.clone(),
    };
    residuals(&prob.apply(&x0), anchors)?;

    let res = |x: &[f64]| -> Vec<f64> {
        let mut v = match residuals(&prob.apply(x), anchors) {
            Ok(r) => r.into_iter().map(|e| e.1).collect(),
            Err(_) => vec![1e3; anchors.len() * 2],
        };
        v.extend(x.iter().zip(&x0).map(|(a, b)| LAMBDA.sqrt() * (a - b)));
        v
    };
    let objective = |x: &[f64]| res(x).iter().map(|v| v * v).sum::<f64>();
    let x = levenberg_marquardt(&res, &x0);
    let x = coordinate_search(&objective, &x);
    let hw = prob.apply(&x);
    let residuals = residuals(&hw, anchors)?;
    let worst = residuals.iter().map(|e| e.1.abs()).fold(0.0, f64::max);
    Ok(Fit {
        hw,
        residuals,
        band: (worst.exp() - 1.0).max(MIN_BAND),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predict::fixture::{features, hw};
    use tetraeder_profiler::DeviceKind;

    fn truth() -> HardwareGraph {
        let mut h = hw();
        h.links[0].lane_group = None;
        h.lane_groups.clear();
        h
    }

    /// The prior is the truth with every fitted rate off by 2x to 5x and no thread curve.
    fn perturbed(t: &HardwareGraph) -> HardwareGraph {
        let mut h = t.clone();
        for d in &mut h.devices {
            let c = d.compute.as_mut().unwrap();
            if d.kind == DeviceKind::Gpu {
                c.flops_per_s *= 3.0;
                c.c0_graph_s *= 5.0;
                c.c0_eager_s *= 0.3;
                d.bandwidth_bytes_per_s.as_mut().unwrap().value *= 0.4;
            } else {
                c.flops_per_s *= 0.3;
                c.c0_eager_s *= 4.0;
                c.c0_graph_s *= 4.0;
                d.bandwidth_bytes_per_s.as_mut().unwrap().value *= 2.5;
                d.thread_curve.clear();
            }
        }
        h.links[0].alpha_s.value *= 4.0;
        h.links[0].beta_bytes_per_s.value *= 0.3;
        h
    }

    struct F {
        gpu_gb: f64,
        cpu_gb: f64,
        gpu_ops: f64,
        cpu_ops: f64,
        threads: u32,
        ubatch: u32,
        offload: bool,
        graphs: bool,
    }

    fn feat(s: F) -> RunFeatures {
        let mut f = features(s.gpu_gb, s.cpu_gb);
        for p in &mut f.parts {
            p.ops_per_token = if p.device == "gpu0" {
                s.gpu_ops
            } else {
                s.cpu_ops
            };
        }
        f.threads = s.threads;
        f.ubatch = s.ubatch;
        f.op_offload = s.offload;
        f.graphs = s.graphs;
        f
    }

    fn anchor(hw: &HardwareGraph, id: &str, f: RunFeatures) -> Anchor {
        let p = predict(hw, &f, 2048, 128).unwrap();
        Anchor {
            id: id.into(),
            features: f,
            prompt: 2048,
            r#gen: 128,
            pp_tok_s: Some(p.pp_tok_s),
            tg_tok_s: Some(p.tg_tok_s),
        }
    }

    fn gpu_anchors(hw: &HardwareGraph, with_eager: bool) -> Vec<Anchor> {
        let g = |gb, ops, graphs| F {
            gpu_gb: gb,
            cpu_gb: 0.0,
            gpu_ops: ops,
            cpu_ops: 0.0,
            threads: 8,
            ubatch: 512,
            offload: true,
            graphs,
        };
        let mut v = vec![
            anchor(hw, "g10", feat(g(10.0, 1200.0, true))),
            anchor(hw, "g6", feat(g(6.0, 1500.0, true))),
        ];
        if with_eager {
            v.push(anchor(hw, "g10e", feat(g(10.0, 1200.0, false))));
        }
        v
    }

    fn cpu_anchors(hw: &HardwareGraph) -> Vec<Anchor> {
        let c = |ops, threads| F {
            gpu_gb: 0.0,
            cpu_gb: 4.0,
            gpu_ops: 0.0,
            cpu_ops: ops,
            threads,
            ubatch: 512,
            offload: true,
            graphs: true,
        };
        vec![
            anchor(hw, "c4", feat(c(480.0, 4))),
            anchor(hw, "c8", feat(c(480.0, 8))),
            anchor(hw, "c8x", feat(c(3000.0, 8))),
            anchor(hw, "c16", feat(c(480.0, 16))),
        ]
    }

    fn mixed_anchors(hw: &HardwareGraph) -> Vec<Anchor> {
        let m = |threads, offload| F {
            gpu_gb: 6.0,
            cpu_gb: 4.0,
            gpu_ops: 720.0,
            cpu_ops: 480.0,
            threads,
            ubatch: 512,
            offload,
            graphs: true,
        };
        vec![
            anchor(hw, "m8on", feat(m(8, true))),
            anchor(hw, "m8off", feat(m(8, false))),
            anchor(hw, "m16on", feat(m(16, true))),
        ]
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a / b - 1.0).abs()
    }

    #[test]
    fn recovers_rates_and_predicts_held_out_runs() {
        let truth = truth();
        let mut anchors = gpu_anchors(&truth, true);
        anchors.extend(cpu_anchors(&truth));
        anchors.extend(mixed_anchors(&truth));
        assert_eq!(anchors.len(), 10);
        let prior = perturbed(&truth);
        let fit = fit(&prior, &anchors, "gpu0", "cpu0", "pcie").unwrap();

        eprintln!("fit gpu {:?}", fit.hw.device("gpu0"));
        eprintln!("fit cpu {:?}", fit.hw.device("cpu0"));
        eprintln!("fit link {:?}", fit.hw.links);
        eprintln!("residuals {:?}", fit.residuals);
        let (tg, fg) = (
            truth.device("gpu0").unwrap(),
            fit.hw.device("gpu0").unwrap(),
        );
        let (tc, fc) = (
            truth.device("gpu0").unwrap().compute.unwrap(),
            fg.compute.unwrap(),
        );
        assert!(rel(fc.flops_per_s, tc.flops_per_s) < 0.05, "gpu F");
        assert!(rel(fc.c0_graph_s, tc.c0_graph_s) < 0.05, "c0 graph");
        assert!(rel(fc.c0_eager_s, tc.c0_eager_s) < 0.05, "c0 eager");
        let (tb, fb) = (
            tg.bandwidth_bytes_per_s.unwrap(),
            fg.bandwidth_bytes_per_s.unwrap(),
        );
        assert!(rel(fb.value, tb.value) < 0.05, "gpu B");
        assert_eq!(fb.source, Source::Measured);
        let (tcpu, fcpu) = (
            truth.device("cpu0").unwrap(),
            fit.hw.device("cpu0").unwrap(),
        );
        assert_eq!(fcpu.thread_curve.len(), 3);
        for t in [4, 8, 16] {
            let (tf, tb) = tcpu.cpu_rates(t).unwrap();
            let (ff, fb) = fcpu.cpu_rates(t).unwrap();
            assert!(rel(ff, tf) < 0.05, "cpu F at {t}: {ff} vs {tf}");
            assert!(rel(fb, tb) < 0.05, "cpu B at {t}: {fb} vs {tb}");
        }
        let c0 = fcpu.compute.unwrap().c0_eager_s;
        assert!(rel(c0, tcpu.compute.unwrap().c0_eager_s) < 0.05, "cpu c0");
        let (tl, fl) = (&truth.links[0], &fit.hw.links[0]);
        assert!(
            rel(fl.beta_bytes_per_s.value, tl.beta_bytes_per_s.value) < 0.05,
            "beta"
        );
        assert!(
            fit.residuals.iter().all(|r| r.1.abs() < 0.02),
            "{:?}",
            fit.residuals
        );
        assert!(fit.band >= 0.05 && fit.band < 0.06);

        // Held-out placements, threads at an anchor count, ubatch and op offload.
        let held = [
            F {
                gpu_gb: 8.0,
                cpu_gb: 2.0,
                gpu_ops: 960.0,
                cpu_ops: 240.0,
                threads: 16,
                ubatch: 256,
                offload: true,
                graphs: true,
            },
            F {
                gpu_gb: 3.0,
                cpu_gb: 7.0,
                gpu_ops: 400.0,
                cpu_ops: 900.0,
                threads: 4,
                ubatch: 1024,
                offload: false,
                graphs: true,
            },
            F {
                gpu_gb: 5.0,
                cpu_gb: 5.0,
                gpu_ops: 600.0,
                cpu_ops: 700.0,
                threads: 8,
                ubatch: 128,
                offload: true,
                graphs: false,
            },
            F {
                gpu_gb: 0.0,
                cpu_gb: 6.0,
                gpu_ops: 0.0,
                cpu_ops: 1200.0,
                threads: 16,
                ubatch: 2048,
                offload: true,
                graphs: true,
            },
            F {
                gpu_gb: 12.0,
                cpu_gb: 0.0,
                gpu_ops: 1800.0,
                cpu_ops: 0.0,
                threads: 8,
                ubatch: 512,
                offload: true,
                graphs: false,
            },
        ];
        for (i, s) in held.into_iter().enumerate() {
            let f = feat(s);
            let want = predict(&truth, &f, 1500, 200).unwrap();
            let got = predict(&fit.hw, &f, 1500, 200).unwrap();
            assert!(rel(got.pp_tok_s, want.pp_tok_s) < 0.05, "held {i} pp");
            assert!(rel(got.tg_tok_s, want.tg_tok_s) < 0.05, "held {i} tg");
        }
    }

    #[test]
    fn unexercised_parameters_stay_at_the_prior() {
        let truth = truth();
        let prior = perturbed(&truth);
        // GPU only, graphs on: no CPU thread, no link and no eager anchor.
        let anchors = gpu_anchors(&truth, false);
        let fit = fit(&prior, &anchors, "gpu0", "cpu0", "pcie").unwrap();
        let (pg, fg) = (
            prior.device("gpu0").unwrap(),
            fit.hw.device("gpu0").unwrap(),
        );
        assert_eq!(
            fg.compute.unwrap().c0_eager_s,
            pg.compute.unwrap().c0_eager_s
        );
        assert_ne!(
            fg.compute.unwrap().flops_per_s,
            pg.compute.unwrap().flops_per_s
        );
        assert_eq!(fit.hw.links, prior.links);
        assert_eq!(fit.hw.device("cpu0"), prior.device("cpu0"));
        assert!(rel(fg.compute.unwrap().flops_per_s, 60e12) < 0.05);
    }

    #[test]
    fn rejects_empty_and_bad_input() {
        let t = truth();
        assert_eq!(
            fit(&t, &[], "gpu0", "cpu0", "pcie").unwrap_err(),
            CalibError::NoAnchors
        );
        let mut a = gpu_anchors(&t, false);
        a[0].tg_tok_s = Some(-1.0);
        assert!(matches!(
            fit(&t, &a, "gpu0", "cpu0", "pcie"),
            Err(CalibError::BadAnchor(_))
        ));
        let a = gpu_anchors(&t, false);
        assert!(matches!(
            fit(&t, &a, "gpuX", "cpu0", "pcie"),
            Err(CalibError::MissingSpec(_))
        ));
    }
}
