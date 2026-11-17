#!/usr/bin/env bash
# S2b diagnostics on the prebuilt test binaries (no rebuild): the two new gpu-parity failures with fusion off and on
# one GPU, then gpu2_ MoE with the 30B-A3B file.
exec 8> /root/diag2.lock; flock -n 8 || exit 0
root=/root/tetraeder
out=$root/bench/results/s2b-20261118-step0/phase5
md="$root/.hf_cache"
d=$root/target/release/deps
cd "$root/crates/model"
run() {
  local name=$1; shift
  echo "=== $name $(date +%T)" >> /root/diag2.progress
  env TETRAEDER_REQUIRE_GPU=1 TETRAEDER_MODELS_DIR="$md" "$@" > "$out/diag-$name.log.full" 2>&1 8>&-
  echo "rc=$?" >> "$out/diag-$name.log.full"
  cut -c1-300 "$out/diag-$name.log.full" > "$out/diag-$name.log"
  rm -f "$out/diag-$name.log.full"
  echo "=== $name done $(date +%T)" >> /root/diag2.progress
}
A="--nocapture --test-threads 1"
P=$d/parity-533b210982ba47a9
S=$d/slots-8df054c608e68d0d
M=$d/gpu_sharded_moe-dfd4f4fde4c50551
run parity-nofusion GGML_CUDA_DISABLE_FUSION=1 $P $A gpu_matches_libllama
run slots-nofusion GGML_CUDA_DISABLE_FUSION=1 $S $A gpu_slot_counts_match_libllama
run parity-gpu0 CUDA_VISIBLE_DEVICES=0 $P $A gpu_matches_libllama
run slots-gpu0 CUDA_VISIBLE_DEVICES=0 $S $A gpu_slot_counts_match_libllama
run moe-gpu2 TETRAEDER_REQUIRE_GPUS=2 TETRAEDER_MOE_GGUF="$md/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf" $M $A gpu2_
echo DIAG_DONE >> /root/diag2.progress
