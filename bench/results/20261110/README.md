# Session 20261110

- Status: several boxes. Detector box exit test passes (ADR 0041); Phase 2 met on the means against KTransformers and the llama.cpp cache, thin (b1-exit, ADR 0045); KV quality run unclean for ADR 0050 (b) (32K rows measured nothing); proxy run with Tetraeder tool errors above llama-server (diagnosis for ADR 0057); ik_llama.cpp quality comparison clean (ADR 0048).
- Code and box: commits of 2026-11-10 (results commits 158e850, 521c516 and others, see git log of this directory); box 60042169 (b1-exit: RTX 3090, Xeon Gold 6330N, quota 26.9, 26 threads, driver 595.91.07), box 59998776 (RTX A6000), boxes 59977867 and 59978288.
- Raw files: this directory.

## From ADR 0041 Status (moved verbatim)

Status: Accepted (2026-11-15: detector passed on a box (single 30/30, no false positives in 10, pairs 9/9) and in the simulated battery). Earlier: Proposed (Phase 3a)
