# Session 20261114b

- Status: clean (middle pair flagged by the co-tenant guard); Phase 2 point 3 met under the owner's 2026-11-14 rule: the builder with the free VRAM as expert slots leads KTransformers in every pair (ADR 0045).
- Code and box: commit on main at the time of box 60600766 (see git log of the results commit); host 60400, same launch as `20261114`.
- Raw files: this directory.

## From docs/m2-plan.md, section "Facts from the read-only checks" (moved verbatim)

- **20261114b per-window tok/s are sampler artifacts** (about 15 tokens over whole seconds). Use only the builder's window JSON.
