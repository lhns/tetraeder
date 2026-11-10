//! A lowered plan as the load options of the two serving engines (ADR 0055).

use crate::engine_args::PlacementArgs;
use crate::serve::{resolve_overlap, resolve_prefetch};
use std::sync::Arc;
use tetraeder_engine::{BufferOverride, ContextConfig, Placement};
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy};
use tetraeder_planner::Lowered;
use tetraeder_telemetry::Telemetry;

/// Placement and context of a libllama plan. `base` supplies what a plan does not say
/// (`--flash-attn`, `--load-mode`); the placement flags are not used with `--plan`.
pub fn llama_options(
    l: &Lowered,
    base: &PlacementArgs,
    ctx: u32,
) -> Result<(Placement, ContextConfig), String> {
    let Lowered::Llama {
        n_gpu_layers,
        cpu_patterns,
        threads,
        batch,
        ubatch,
        op_offload,
        ..
    } = l
    else {
        return Err("not a libllama plan".into());
    };
    let overrides = cpu_patterns
        .iter()
        .map(|p| BufferOverride::parse(p).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let placement = Placement {
        n_gpu_layers: *n_gpu_layers,
        overrides,
        ..base.placement()
    };
    let cfg = ContextConfig {
        n_batch: *batch,
        n_ubatch: *ubatch,
        threads: i32::try_from(*threads).ok(),
        op_offload: *op_offload,
        ..base.context(ctx)
    };
    Ok((placement, cfg))
}

/// Prompt tokens per backend call of a builder plan: its ubatch.
pub fn builder_chunk(l: &Lowered) -> Result<usize, String> {
    match l {
        Lowered::Builder { ubatch, .. } => Ok((*ubatch as usize).max(1)),
        Lowered::Llama { .. } => Err("not a builder plan".into()),
    }
}

/// Load options of a builder plan. A RAM tier means misses run on the CPU, as in the campaign;
/// otherwise `miss` (the `--miss` flag) decides.
pub fn builder_options(
    l: &Lowered,
    ctx: u32,
    miss: MissPolicy,
    prefetch: Option<bool>,
    prefetch_depth: u8,
    io_service: Option<bool>,
    overlap: Option<bool>,
    telemetry: Arc<Telemetry>,
) -> Result<LoadOptions, String> {
    let Lowered::Builder {
        gpu,
        expert_slots,
        ram_slots,
        threads,
        op_offload,
        ..
    } = l
    else {
        return Err("not a builder plan".into());
    };
    let threads = *threads as usize;
    let backend = match gpu {
        Some(i) => BackendChoice::Gpu { index: *i as usize },
        None => BackendChoice::Cpu {
            threads: Some(threads),
        },
    };
    let mut o = LoadOptions::new(backend);
    o.cpu_threads = Some(threads);
    o.n_ctx = ctx as usize;
    o.expert_slots = expert_slots.map(|s| s as usize);
    o.ram_slots = ram_slots.map(|s| s as usize);
    o.op_offload = *op_offload;
    o.miss = if ram_slots.is_some() {
        MissPolicy::Cpu
    } else {
        miss
    };
    o.prefetch = resolve_prefetch(ram_slots.is_some(), prefetch);
    o.prefetch_depth = usize::from(prefetch_depth);
    (o.io.service, o.overlap) = resolve_overlap(ram_slots.is_some(), o.miss, io_service, overlap)?;
    o.telemetry = Some(telemetry);
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tetraeder_engine::FlashAttn;
    use tetraeder_telemetry::TelemetryConfig;

    #[derive(Parser)]
    struct P {
        #[command(flatten)]
        place: PlacementArgs,
    }

    fn base(extra: &[&str]) -> PlacementArgs {
        let mut args = vec!["x"];
        args.extend(extra);
        P::parse_from(args).place
    }

    // hardware: none
    #[test]
    fn llama_options_follow_the_lowered_plan() {
        let l = Lowered::Llama {
            n_gpu_layers: 99,
            cpu_patterns: vec![r"blk\.(2[4-9])\.ffn_(up|down|gate)_exps=CPU".into()],
            threads: 6,
            batch: 2048,
            ubatch: 512,
            op_offload: true,
        };
        let (p, c) = llama_options(&l, &base(&["--flash-attn", "on"]), 4096).unwrap();
        assert_eq!(p.n_gpu_layers, 99);
        assert_eq!(p.overrides.len(), 1);
        assert_eq!(p.overrides[0].buffer, "CPU");
        assert_eq!(
            p.overrides[0].pattern,
            r"blk\.(2[4-9])\.ffn_(up|down|gate)_exps"
        );
        assert_eq!(
            (c.n_ctx, c.n_batch, c.n_ubatch, c.threads),
            (4096, 2048, 512, Some(6))
        );
        assert_eq!(c.flash_attn, FlashAttn::On);
        assert!(c.op_offload);
        let cpu = Lowered::Llama {
            n_gpu_layers: 0,
            cpu_patterns: vec![],
            threads: 2,
            batch: 2048,
            ubatch: 512,
            op_offload: false,
        };
        let (p, c) = llama_options(&cpu, &base(&[]), 256).unwrap();
        assert!(!c.op_offload);
        assert_eq!((p.n_gpu_layers, p.overrides.len()), (0, 0));
        assert!(
            llama_options(
                &Lowered::Builder {
                    gpu: None,
                    expert_slots: None,
                    ram_slots: None,
                    threads: 1,
                    ubatch: 512,
                    op_offload: true
                },
                &base(&[]),
                256
            )
            .is_err()
        );
    }

    // hardware: none
    #[test]
    fn builder_options_follow_the_lowered_plan() {
        let tel = Arc::new(Telemetry::new(&TelemetryConfig::from_env()).unwrap());
        let gpu = Lowered::Builder {
            gpu: Some(1),
            expert_slots: Some(16),
            ram_slots: Some(60),
            threads: 7,
            ubatch: 512,
            op_offload: false,
        };
        let o = builder_options(
            &gpu,
            4096,
            MissPolicy::Upload,
            None,
            2,
            None,
            None,
            Arc::clone(&tel),
        )
        .unwrap();
        assert!(matches!(o.backend, BackendChoice::Gpu { index: 1 }));
        assert_eq!(
            (o.cpu_threads, o.n_ctx, o.expert_slots, o.ram_slots),
            (Some(7), 4096, Some(16), Some(60))
        );
        assert_eq!(
            (o.op_offload, o.miss, o.prefetch),
            (false, MissPolicy::Cpu, true)
        );
        // A RAM tier plans both on, as `serve` does by default.
        assert!(o.io.service && o.overlap);
        assert_eq!(o.prefetch_depth, 2);
        assert_eq!(builder_chunk(&gpu).unwrap(), 512);
        let cpu = Lowered::Builder {
            gpu: None,
            expert_slots: None,
            ram_slots: None,
            threads: 2,
            ubatch: 64,
            op_offload: true,
        };
        assert_eq!(builder_chunk(&cpu).unwrap(), 64);
        let tel2 = Arc::clone(&tel);
        let o = builder_options(&cpu, 256, MissPolicy::Upload, None, 2, None, None, tel).unwrap();
        assert!(matches!(o.backend, BackendChoice::Cpu { threads: Some(2) }));
        assert_eq!(
            (o.expert_slots, o.ram_slots, o.miss),
            (None, None, MissPolicy::Upload)
        );
        assert!(!o.io.service && !o.overlap && !o.prefetch);
        assert!(
            builder_options(
                &cpu,
                256,
                MissPolicy::Upload,
                None,
                2,
                None,
                Some(true),
                tel2
            )
            .is_err()
        );
    }
}
