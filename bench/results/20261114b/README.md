# Session 20261114b

- Status: clean (middle pair flagged by the co-tenant guard); Phase 2 point 3 met under the owner's 2026-11-14 rule: the builder with the free VRAM as expert slots leads KTransformers in every pair (ADR 0045).
- Code and box: commit on main at the time of box 60600766 (see git log of the results commit); host 60400, same launch as `20261114`.
- Raw files: this directory.

## From docs/m2-plan.md, section "Facts from the read-only checks" (moved verbatim)

- **20261114b per-window tok/s are sampler artifacts** (about 15 tokens over whole seconds). Use only the builder's window JSON.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261114b` paragraph, moved verbatim)

`20261114b` (box 60600766, host 60400, same launch as `20261114`; guards clean except the middle pair): **builder with the free VRAM as expert slots ahead in every pair.** Order vramfill, KT, vramfill, KT, builder at the default 10.97 GB, vramfill at poll 50. Decode 7.560, 5.707, 6.882, 5.384, 5.559, 9.187 tok/s: vramfill over KT 1.32 (clean) and 1.28 (both sides flagged by the co-tenant guard, 18.6 and 14.8 cores outside); default builder over the adjacent KT 0.97 and 1.03; vramfill at poll 50 1.22x vramfill (37% of decode periods throttled); KT spread 5.7%, vramfill 9%; prefill vramfill 2.4x KT. Peak VRAM vramfill 40.9 of 48 GB, KT 20.4. Across `20261114` and `20261114b` vramfill leads KT in all four pairs (1.28 to 1.65), three of them clean; its slowest run (6.88) is 1.16x KT's fastest (5.91). At the 10.97 GB default the builder and KT are level (0.85 to 1.15). Results `bench/results/20261114b/`.

## From ADR 0045 (addendum "owner decision on point 3", 2026-11-14, its numbers, moved verbatim)

Decision (owner, 2026-11-14, after `20261114` and `20261114b`): against KTransformers the builder runs with the free VRAM as expert slots (`main-overlap-vramfill`). KTransformers 0.7.2 on a GGUF setup places no experts on the GPU (its GPU experts need HF weights, which do not fit the 230 GB disk next to the GGUF), so this pairs each engine at its best runnable configuration on the same box, the rule already used for the llama.cpp cache (mmap at poll 0). The 10.97 GB default stays the pairing against llama.cpp, which can place experts on the GPU (`-ot`). Point 3 is met: the builder leads in all four pairs (1.28 to 1.65, three clean, two sessions on host 60400). At the 10.97 GB default the two are level (0.85 to 1.15), recorded as such.
