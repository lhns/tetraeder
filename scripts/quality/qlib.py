"""Shared helpers of the quality scripts (ADR 0042): GGUF headers, profile groups, llama-perplexity runs.

Stdlib only, so the scripts run on a box without a virtualenv.
"""

import hashlib
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path

# ggml type id -> (name, elements per block, bytes per block), ggml.h and ggml-common.h at the pin.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
BYTES = {name: (blk, size) for name, blk, size in GGML_TYPES.values()}

# GGUF metadata value types: id -> struct format (strings and arrays handled apart).
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def format_bytes(fmt, n_elements):
    blk, size = BYTES[fmt]
    return -(-n_elements // blk) * size


class Tensor:
    def __init__(self, name, ty, ne, offset, path):
        self.name, self.ne, self.offset, self.path = name, ne, offset, path
        self.type = GGML_TYPES[ty][0]
        self.n = 1
        for d in ne:
            self.n *= d
        self.nbytes = format_bytes(self.type, self.n)


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, t):
    if t == 8:
        return _read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et == 8:
            for _ in range(n):
                _read_str(f)
        else:
            f.seek(n * struct.calcsize(_SCALAR[et]), 1)
        return None
    fmt = _SCALAR[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def read_header(path):
    """(metadata dict without arrays, [Tensor]) of one GGUF file; offsets are absolute."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        _version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        kv = {}
        for _ in range(n_kv):
            key = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            kv[key] = _read_value(f, t)
        infos = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            ne = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            ty, off = struct.unpack("<IQ", f.read(12))
            infos.append((name, ty, ne, off))
        align = kv.get("general.alignment", 32)
        data = -(-f.tell() // align) * align
    return kv, [Tensor(n, t, ne, data + off, path) for n, t, ne, off in infos]


def split_paths(first):
    """All shard paths of a model given its first shard (llama.cpp naming)."""
    first = Path(first)
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    n = int(m.group(2))
    return [first.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def read_model(first):
    """(metadata of the first shard, every tensor of every shard)."""
    kv, tensors = None, []
    for p in split_paths(first):
        k, ts = read_header(p)
        kv = kv if kv is not None else k
        tensors += ts
    return kv, tensors


def tensor_bytes(first):
    return sum(t.nbytes for t in read_model(first)[1])


def data_digest(first):
    """SHA-256 per tensor name over its data bytes (to compare two conversions byte for byte)."""
    out = {}
    for t in read_model(first)[1]:
        h = hashlib.sha256()
        with open(t.path, "rb") as f:
            f.seek(t.offset)
            left = t.nbytes
            while left:
                b = f.read(min(left, 64 << 20))
                h.update(b)
                left -= len(b)
        out[t.name] = (t.type, h.hexdigest())
    return out


# Role classes of ADR 0042: tensor suffix -> (class, roles, unit kind).
_CLASSES = {
    "attn_q": ("attn.q+k", ["attn.q", "attn.k"], "attn"),
    "attn_k": ("attn.q+k", ["attn.q", "attn.k"], "attn"),
    "attn_v": ("attn.v", ["attn.v"], "attn"),
    "attn_output": ("attn.o", ["attn.o"], "attn"),
    "ffn_gate": ("mlp.gate+up", ["mlp.gate", "mlp.up"], "mlp"),
    "ffn_up": ("mlp.gate+up", ["mlp.gate", "mlp.up"], "mlp"),
    "ffn_down": ("mlp.down", ["mlp.down"], "mlp"),
    "ffn_gate_exps": ("expert.gate+up", ["expert.gate", "expert.up"], "expert"),
    "ffn_up_exps": ("expert.gate+up", ["expert.gate", "expert.up"], "expert"),
    "ffn_down_exps": ("expert.down", ["expert.down"], "expert"),
}


def build_groups(tensors, n_layer, span):
    """Profile groups (ADR 0042 v1): embed, head, and per role class per `span` layers. Each group gets its
    `units` selector, roles, params and tensor names; tensors outside every group (norms, router) are left out."""
    groups = {}

    def add(gid, units, roles, t):
        g = groups.setdefault(gid, {"id": gid, "units": units, "roles": roles, "params": 0, "tensors": []})
        g["params"] += t.n
        g["tensors"].append(t.name)

    for t in tensors:
        if t.name == "token_embd.weight":
            add("embed", {"kinds": ["embed"]}, [], t)
        elif t.name == "output.weight":
            add("head", {"kinds": ["head"]}, [], t)
        else:
            m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", t.name)
            if not m or m.group(2) not in _CLASSES:
                continue
            il = int(m.group(1))
            cls, roles, kind = _CLASSES[m.group(2)]
            a = il // span * span
            b = min(a + span, n_layer)
            add(f"{cls}@{a}-{b}", {"kinds": [kind], "layers": [a, b]}, roles, t)
    order = lambda g: (g["id"] not in ("embed",), g["id"] == "head", g["units"].get("layers", [0])[0], g["id"])
    return sorted(groups.values(), key=order)


def arch_layers(kv):
    arch = kv["general.architecture"]
    return arch, int(kv[f"{arch}.block_count"]), int(kv.get(f"{arch}.expert_count") or 0)


def shard_name(prefix, no, count):
    return f"{prefix}-{no + 1:05d}-of-{count:05d}.gguf"


def assemble(mix_dir, shard_dirs, choice, count, name="mix"):
    """A split model in `mix_dir` made of symbolic links: shard 0 (metadata, untouched tensors) from
    `shard_dirs[choice[0]]`, shard i from `shard_dirs[choice[i]]`. Returns the first shard's path."""
    mix_dir = Path(mix_dir)
    mix_dir.mkdir(parents=True, exist_ok=True)
    for no in range(count):
        link = mix_dir / shard_name(name, no, count)
        if link.is_symlink() or link.exists():
            link.unlink()
        target = Path(shard_dirs[choice[no]]) / shard_name("m", no, count)
        os.symlink(target.resolve(), link)
    return mix_dir / shard_name(name, 0, count)


_NUM = r"([-+0-9.eE]+|nan|inf)"


def parse_kld(text):
    """Numbers of a `llama-perplexity --kl-divergence` log."""
    out = {}
    pats = {
        "kl": rf"Mean\s+KLD:\s+{_NUM}\s+±\s+{_NUM}",
        "ppl": rf"Mean PPL\(Q\)\s+:\s+{_NUM}\s+±\s+{_NUM}",
        "ppl_base": rf"Mean PPL\(base\)\s+:\s+{_NUM}\s+±\s+{_NUM}",
        "same_top_p": rf"Same top p:\s+{_NUM}\s+±\s+{_NUM}",
        "kl_p99": rf"99\.0%\s+KLD:\s+{_NUM}",
    }
    for k, p in pats.items():
        m = re.search(p, text)
        if m:
            out[k] = float(m.group(1))
            if m.lastindex and m.lastindex >= 2:
                out[k + "_err"] = float(m.group(2))
    if "kl" not in out:
        raise RuntimeError("no KLD in llama-perplexity output:\n" + text[-3000:])
    return out


def perplexity(bin_dir, model, args, log_path, timeout=3600):
    """Runs llama-perplexity; returns (combined output, seconds). Raises on failure."""
    cmd = [str(Path(bin_dir) / "llama-perplexity"), "-m", str(model)] + args
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t
    text = p.stdout + p.stderr
    Path(log_path).write_text(" ".join(cmd) + "\n" + text)
    if p.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed ({p.returncode}), see {log_path}")
    return text, dt


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


# Precision order (bits per weight), ADR 0027.
def bpw(fmt):
    blk, size = BYTES[fmt]
    return size * 8 / blk


def estimate(profile, choice):
    """ADR 0042 estimate for a whole-group choice {group id: format}."""
    base = profile["base"]
    q = base["kl"]
    for g in profile["groups"]:
        f = choice.get(g["id"], base["format"])
        if bpw(f) >= bpw(base["format"]):
            continue
        e = next((e for e in g["entries"] if e["format"] == f), None)
        if e is None:
            raise KeyError(f"no entry for {g['id']} at {f}")
        q += e["kl"] - base["kl"]
    return q
