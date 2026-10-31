"""Writes a synthetic random-weight qwen3moe GGUF (ADR 0030) with the vendored gguf-py.

Usage (cargo xtask tiny-moe runs this through uv): tiny_moe.py OUT.gguf
2 layers, 8 experts, top-2, 64-wide, byte-level vocab. Matrices are F16, norms and router F32,
like llama.cpp's converter. Deterministic: fixed seed.
"""

import sys

import numpy as np
import gguf

N_LAYER, N_EMBD, N_HEAD, N_KV, D_HEAD = 2, 64, 4, 2, 16
N_EXPERT, N_USED, N_FF_EXP, N_CTX = 8, 2, 32, 256
SEED = 20261031


def byte_to_unicode():
    """GPT-2 byte to printable unicode map, so the vocab is a valid byte-level BPE vocab."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def main(out):
    rng = np.random.default_rng(SEED)

    def mat(*shape, std=0.08):
        return (rng.standard_normal(shape) * std).astype(np.float16)

    def norm(*shape):
        return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)

    tokens = byte_to_unicode() + ["th", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    n_vocab = len(tokens)
    types = [gguf.TokenType.NORMAL] * 257 + [gguf.TokenType.CONTROL] * 3
    eos = tokens.index("<|im_end|>")

    w = gguf.GGUFWriter(out, "qwen3moe")
    w.add_name("tiny-qwen3moe")
    w.add_block_count(N_LAYER)
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF_EXP * N_USED)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_key_length(D_HEAD)
    w.add_value_length(D_HEAD)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(["t h"])
    w.add_eos_token_id(eos)
    w.add_pad_token_id(tokens.index("<|endoftext|>"))
    w.add_add_bos_token(False)

    w.add_tensor("token_embd.weight", mat(n_vocab, N_EMBD, std=0.5))
    w.add_tensor("output_norm.weight", norm(N_EMBD))
    w.add_tensor("output.weight", mat(n_vocab, N_EMBD, std=0.5))
    for i in range(N_LAYER):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "attn_q.weight", mat(N_HEAD * D_HEAD, N_EMBD, std=0.2))
        w.add_tensor(p + "attn_k.weight", mat(N_KV * D_HEAD, N_EMBD, std=0.2))
        w.add_tensor(p + "attn_v.weight", mat(N_KV * D_HEAD, N_EMBD, std=0.2))
        w.add_tensor(p + "attn_output.weight", mat(N_EMBD, N_HEAD * D_HEAD, std=0.2))
        w.add_tensor(p + "attn_q_norm.weight", norm(D_HEAD))
        w.add_tensor(p + "attn_k_norm.weight", norm(D_HEAD))
        w.add_tensor(p + "ffn_norm.weight", norm(N_EMBD))
        w.add_tensor(
            p + "ffn_gate_inp.weight",
            (rng.standard_normal((N_EXPERT, N_EMBD)) * 0.5).astype(np.float32),
        )
        w.add_tensor(p + "ffn_gate_exps.weight", mat(N_EXPERT, N_FF_EXP, N_EMBD, std=0.2))
        w.add_tensor(p + "ffn_up_exps.weight", mat(N_EXPERT, N_FF_EXP, N_EMBD, std=0.2))
        w.add_tensor(p + "ffn_down_exps.weight", mat(N_EXPERT, N_EMBD, N_FF_EXP, std=0.2))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main(sys.argv[1])
