//! The all-reduce fallback of ADR 0069: a star through the first rank of the group, built from
//! async copies and one-node in-place add graphs. It serves CPU ranks in CI, several ranks on one
//! GPU and any group the CUDA comm declines.
//!
//! Everything the adds touch is scratch on the root: `s[0]` receives a copy of the root's tensor,
//! `s[i]` the tensor of group member `i`; `s[0] += s[i]` runs as a graph of its own; the sum is
//! copied back to every member. So the add graphs depend on the scratch only, and one scratch
//! set (per group and shape) serves every all-reduce of that shape, in any program.

use crate::{ExecError, RankId, Ranks};
use std::rc::Rc;
use tetraeder_ggml_sys::ggml::{Context, Event, OwnedGraph, OwnedTensor, Tensor, Type};

pub(crate) struct Scratch {
    // Keeps the context (and its buffer) alive next to the handles; never read.
    _ctx: Rc<Context>,
    s: Vec<OwnedTensor>,
    adds: Vec<OwnedGraph>,
    /// Recorded on a GPU root after the last add of an all-reduce: the next all-reduce makes the
    /// members' streams wait for it before they overwrite scratch the adds read. None on a CPU
    /// root, which computes synchronously.
    done: Option<Event>,
    recorded: bool,
}

impl Scratch {
    pub(crate) fn new(
        ranks: &Ranks,
        group: &[RankId],
        like: Tensor<'_>,
    ) -> Result<Scratch, ExecError> {
        if like.ty() != Type::F32 {
            return Err(ExecError::invalid(
                "the fallback all-reduce supports F32 tensors only",
            ));
        }
        let n = group.len();
        let root = ranks.backend(group[0]);
        let ctx = Rc::new(Context::with_graphs(2 * n, &vec![4; n - 1])?);
        let ne = like.ne();
        let s: Vec<OwnedTensor> = (0..n).map(|_| ctx.new_persistent(Type::F32, &ne)).collect();
        let mut adds = Vec::with_capacity(n - 1);
        for member in &s[1..] {
            let sum = ctx.add_inplace(s[0].get(), member.get());
            let mut g = ctx.new_graph(4);
            g.expand(sum);
            adds.push(ctx.keep_graph(g, &[])?);
        }
        // After the ops, so ggml initializes the in-place results as views of `s[0]`.
        ctx.alloc_tensors(root)?;
        let done = if root.is_cpu() {
            None
        } else {
            Event::new(root).ok()
        };
        Ok(Scratch {
            _ctx: ctx,
            s,
            adds,
            done,
            recorded: false,
        })
    }

    /// Sums `tensors` (one per rank of `group`, the root first) in place.
    pub(crate) fn all_reduce(
        &mut self,
        ranks: &Ranks,
        group: &[RankId],
        tensors: &[Tensor<'_>],
    ) -> Result<(), ExecError> {
        let root = ranks.backend(group[0]);
        tensors[0].copy_tensor_async(root, root, self.s[0].get())?;
        for (i, t) in tensors.iter().enumerate().skip(1) {
            let member = ranks.backend(group[i]);
            if let (Some(ev), true, false) = (&self.done, self.recorded, member.is_cpu()) {
                ev.wait(member);
            }
            t.copy_tensor_async(member, root, self.s[i].get())?;
        }
        for add in &self.adds {
            root.compute_async(&add.get())?;
        }
        if let Some(ev) = &self.done {
            ev.record(root);
            self.recorded = true;
        }
        for (i, t) in tensors.iter().enumerate() {
            self.s[0]
                .get()
                .copy_tensor_async(root, ranks.backend(group[i]), *t)?;
        }
        Ok(())
    }
}
