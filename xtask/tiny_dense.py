"""Writes a synthetic random-weight qwen3 (dense) GGUF (ADR 0066) with the vendored gguf-py.

Usage (cargo xtask tiny-dense runs this through uv): tiny_dense.py OUT.gguf [n_layer]
2 layers by default (`--layers 4` gives the four-layer variant for the split-point tests), 128-wide, 4 heads over 2 KV heads of 64 (so CUDA flash attention, quantized KV and
the Hadamard rotation run on it), FFN 256, byte-level vocab as tiny_moe.py, untied output.
Matrices are F16, norms F32, like llama.cpp's converter. Deterministic: fixed seed.
"""

import sys

import numpy as np
import gguf

N_EMBD, N_HEAD, N_KV, D_HEAD = 128, 4, 2, 64
N_FF, N_CTX = 256, 512
SEED = 20261114


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


def main(out, n_layer=2):
    rng = np.random.default_rng(SEED)

    def mat(*shape, std=0.08):
        return (rng.standard_normal(shape) * std).astype(np.float16)

    def norm(*shape):
        return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)

    tokens = byte_to_unicode() + ["th", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    n_vocab = len(tokens)
    types = [gguf.TokenType.NORMAL] * 257 + [gguf.TokenType.CONTROL] * 3
    eos = tokens.index("<|im_end|>")

    w = gguf.GGUFWriter(out, "qwen3")
    w.add_name("tiny-qwen3")
    w.add_block_count(n_layer)
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_key_length(D_HEAD)
    w.add_value_length(D_HEAD)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
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
    for i in range(n_layer):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "attn_q.weight", mat(N_HEAD * D_HEAD, N_EMBD, std=0.2))
        w.add_tensor(p + "attn_k.weight", mat(N_KV * D_HEAD, N_EMBD, std=0.2))
        w.add_tensor(p + "attn_v.weight", mat(N_KV * D_HEAD, N_EMBD, std=0.2))
        w.add_tensor(p + "attn_output.weight", mat(N_EMBD, N_HEAD * D_HEAD, std=0.2))
        w.add_tensor(p + "attn_q_norm.weight", norm(D_HEAD))
        w.add_tensor(p + "attn_k_norm.weight", norm(D_HEAD))
        w.add_tensor(p + "ffn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", mat(N_FF, N_EMBD, std=0.2))
        w.add_tensor(p + "ffn_up.weight", mat(N_FF, N_EMBD, std=0.2))
        w.add_tensor(p + "ffn_down.weight", mat(N_EMBD, N_FF, std=0.2))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
