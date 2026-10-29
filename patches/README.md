# ggml patch queue (private)

Ordered patches against the vendored llama.cpp pinned in `vendor/PIN` (ADR 0011).
Never edit `vendor/llama.cpp` in place; the working tree there is disposable.

- `series` lists patch file names, one per line, applied in order (`#` comments allowed).
- Each patch has an ADR in `docs/adr/` and tests on every backend it touches.
- Name patches `NNNN-short-title.patch`; keep them small and upstream-clean.
- Apply with `patches/apply.sh` or `patches/apply.ps1`: both reset the submodule to the pin and
  apply the series with `git -C vendor/llama.cpp apply`. Safe to rerun.
- Create a patch: edit in `vendor/llama.cpp`, `git -C vendor/llama.cpp diff > patches/NNNN-title.patch`,
  add it to `series`, then rerun the apply script to check it applies from a clean pin.
- `PIN_REF=<ref>` overrides the pin (used by the weekly rebase job to try upstream master).
