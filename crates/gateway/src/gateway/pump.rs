//! The output pump: engine events to decoded, parsed segments for one generation.

use super::{Gateway, Operation, OutEvent};
use crate::chat::{Finish, Segment, Usage};
use crate::decode::{StopMatcher, Utf8Decoder};
use crate::engine::{EngineStats, FinishReason, GenEvent, Vocab};
use crate::error::GatewayError;
use crate::toolcall::OutputParser;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{OwnedSemaphorePermit, mpsc};
use tokio_util::sync::CancellationToken;

/// State of one generation's output pump.
pub(super) struct Run {
    pub(super) gw: Arc<Gateway>,
    pub(super) id: String,
    pub(super) model: String,
    pub(super) caller: String,
    pub(super) op: Operation,
    pub(super) started: Instant,
    pub(super) prompt_tokens: u32,
    pub(super) vocab: Arc<Vocab>,
    pub(super) parser: Option<OutputParser>,
    /// Parallel calls are off: generation ends after the first complete tool call.
    pub(super) single_call: bool,
    pub(super) stop: StopMatcher,
    pub(super) cancel: CancellationToken,
    pub(super) span: tracing::Span,
    pub(super) _permit: OwnedSemaphorePermit,
}

enum Outcome {
    Done(Finish),
    ClientGone,
    Failed(GatewayError),
}

impl Run {
    fn segments(&mut self, text: &str, out: &mut Vec<Segment>) {
        if text.is_empty() {
            return;
        }
        match &mut self.parser {
            Some(p) => out.extend(p.push(text)),
            None => out.push(Segment::Text(text.to_owned())),
        }
    }

    pub(super) async fn pump(
        mut self,
        mut erx: mpsc::Receiver<GenEvent>,
        tx: mpsc::Sender<OutEvent>,
    ) {
        let mut utf8 = Utf8Decoder::default();
        let mut n_out: u32 = 0;
        let mut first: Option<Instant> = None;
        let mut last = self.started;
        let mut stop_hit: Option<String> = None;
        let mut saw_tool_call = false;
        let mut call_done = false;
        let mut stats = EngineStats::default();
        let mut buf = Vec::new();

        let outcome = loop {
            let Some(ev) = erx.recv().await else {
                break Outcome::Failed(GatewayError::Internal(
                    "engine ended without finishing".into(),
                ));
            };
            buf.clear();
            match ev {
                GenEvent::Token(id) => {
                    if stop_hit.is_some() || call_done {
                        continue;
                    }
                    let now = Instant::now();
                    if first.is_none() {
                        first = Some(now);
                        if let Some(m) = &self.gw.metrics {
                            m.genai.time_to_first_token(
                                now - self.started,
                                &self.model,
                                self.op.as_str(),
                            );
                        }
                    }
                    last = now;
                    n_out += 1;
                    let text = utf8.push(self.vocab.bytes(id));
                    let (emit, hit) = self.stop.push(&text);
                    self.segments(&emit, &mut buf);
                    if hit.is_some() {
                        stop_hit = hit;
                        self.cancel.cancel();
                    }
                }
                GenEvent::Finished { reason, stats: s } => {
                    stats = s;
                    let ended_early = stop_hit.is_some() || call_done;
                    if reason == FinishReason::Cancelled && !ended_early {
                        break Outcome::ClientGone;
                    }
                    if !ended_early {
                        let (emit, _) = self.stop.push(&utf8.finish());
                        self.segments(&emit, &mut buf);
                        let rest = self.stop.finish();
                        self.segments(&rest, &mut buf);
                    }
                    if let Some(p) = self.parser.as_mut().filter(|_| !call_done) {
                        buf.extend(p.finish());
                    }
                    saw_tool_call |= buf.iter().any(|s| matches!(s, Segment::ToolCall(_)));
                    let finish = match (stop_hit.take(), reason) {
                        (Some(s), _) => Finish::StopSequence(s),
                        _ if call_done => Finish::ToolCalls,
                        (None, FinishReason::Length) => Finish::Length,
                        _ if saw_tool_call => Finish::ToolCalls,
                        _ => Finish::Stop,
                    };
                    if !self.send_all(&tx, &mut buf).await {
                        break Outcome::ClientGone;
                    }
                    break Outcome::Done(finish);
                }
                GenEvent::Error(msg) => {
                    break Outcome::Failed(GatewayError::Internal(format!("engine error: {msg}")));
                }
            }
            saw_tool_call |= buf.iter().any(|s| matches!(s, Segment::ToolCall(_)));
            if !self.send_all(&tx, &mut buf).await {
                self.cancel.cancel();
                break Outcome::ClientGone;
            }
            if self.single_call && saw_tool_call && !call_done {
                call_done = true;
                self.cancel.cancel();
            }
        };

        if matches!(outcome, Outcome::ClientGone) {
            // The engine stops on cancellation; wait for its final stats for accounting.
            self.cancel.cancel();
            while let Some(ev) = erx.recv().await {
                if let GenEvent::Finished { stats: s, .. } = ev {
                    stats = s;
                }
            }
        }
        let usage = Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: n_out,
            cached_tokens: stats.cached_tokens,
        };
        // Account before the final event so a caller that saw it also sees the metrics.
        let error_type = match &outcome {
            Outcome::Done(finish) => {
                self.span
                    .record("gen_ai.response.finish_reasons", finish.openai_reason());
                None
            }
            Outcome::ClientGone => Some("cancelled"),
            Outcome::Failed(e) => Some(e.otel_type()),
        };
        self.finish(usage, &stats, first, last, error_type);
        let last_event = match outcome {
            Outcome::Done(finish) => Some(OutEvent::Done { finish, usage }),
            Outcome::ClientGone => None,
            Outcome::Failed(e) => Some(OutEvent::Error(e)),
        };
        if let Some(ev) = last_event {
            let _ = tx.send(ev).await;
        }
    }

    async fn send_all(&self, tx: &mpsc::Sender<OutEvent>, buf: &mut Vec<Segment>) -> bool {
        for s in buf.drain(..) {
            if tx.send(OutEvent::Segment(s)).await.is_err() {
                return false;
            }
        }
        true
    }
}
