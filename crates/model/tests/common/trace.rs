//! Writes the builder's recorded routing as a `tetraeder-routing` trace (ADR 0050 (c)).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tetraeder_memory::replay::trace::{TraceHeader, TraceWriter};
use tetraeder_model::Qwen3Moe;

pub struct RoutingRecorder {
    writer: TraceWriter<BufWriter<File>>,
    next_step: u32,
    step: u32,
}

impl RoutingRecorder {
    /// A recorder for `m`, which must be loaded with `record_routing` and stream experts.
    /// `hot_source` is `first_prefill`, `refresh` or `profile`; `profile` the routing profile the
    /// run started from, if any.
    pub fn create(
        path: &Path,
        m: &Qwen3Moe,
        ubatch: usize,
        prompt_hash: &str,
        hot_source: &str,
        profile: Option<Vec<Vec<u64>>>,
    ) -> Self {
        let h = m.hparams();
        let (vram, ram) = m
            .tier_slots()
            .expect("a routing trace needs streamed experts");
        let mut header = TraceHeader::new(h.n_layer, h.n_expert, h.n_expert_used);
        header.model = m
            .file()
            .gguf
            .paths
            .first()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        header.vram_slots = vram as u32;
        header.ram_slots = ram.unwrap_or(0) as u32;
        header.prefetch_depth = m.prefetch_depth() as u32;
        header.hot_source = hot_source.into();
        header.ubatch = ubatch as u32;
        header.prompt_hash = prompt_hash.into();
        header.expert_bytes = m.file().expert_bytes_per_layer().expect("expert sizes");
        header.profile = profile;
        Self {
            writer: TraceWriter::create(path, &header).expect("trace file"),
            next_step: 0,
            step: 0,
        }
    }

    /// Writes the routing recorded since the last call (call it after every `eval`).
    pub fn drain(&mut self, m: &mut Qwen3Moe) {
        for r in m.take_routing_log() {
            if r.refresh {
                self.writer.hot_refresh(self.step).expect("trace write");
                continue;
            }
            if r.layer == 0 {
                self.step = self.next_step;
                self.next_step += 1;
            }
            self.writer
                .layer_step(
                    self.step,
                    r.layer as u16,
                    r.n_tokens as u32,
                    &r.ids,
                    r.predicted.as_deref(),
                    r.predicted2.as_deref(),
                )
                .expect("trace write");
        }
    }

    pub fn finish(self) {
        self.writer.finish().expect("trace flush");
    }
}
