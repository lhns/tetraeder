"""Holds GPU memory so the engines see a smaller card (ADR 0040 fixtures, ADR 0041 VRAM pressure fault).

Usage: python3 vram_hold.py --leave-free-mib N [--gpu 0]   (or --hold-mib N)
Allocates with cudaMalloc through ctypes (no build step), touches it with cudaMemset, prints one line
`vram_hold: holding <MiB> MiB, free now <MiB> MiB` and sleeps until killed. Exits 1 if CUDA fails.
"""

import argparse
import ctypes
import glob
import subprocess
import sys
import time


def free_mib(gpu):
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def load_cudart():
    names = ["libcudart.so"] + sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"), reverse=True)
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    sys.exit("vram_hold: libcudart not found")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--leave-free-mib", type=int)
    g.add_argument("--hold-mib", type=int)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    rt = load_cudart()
    if rt.cudaSetDevice(a.gpu) != 0:
        sys.exit("vram_hold: cudaSetDevice failed")
    # the CUDA context itself takes some memory; measure after creating it
    rt.cudaFree(ctypes.c_void_p(0))
    hold = a.hold_mib if a.hold_mib is not None else free_mib(a.gpu) - a.leave_free_mib
    held = 0
    chunk = 256
    ptrs = []
    while held < hold:
        n = min(chunk, hold - held)
        p = ctypes.c_void_p()
        if rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n << 20)) != 0:
            sys.exit(f"vram_hold: cudaMalloc failed after {held} MiB")
        rt.cudaMemset(p, 1, ctypes.c_size_t(n << 20))
        ptrs.append(p)
        held += n
    rt.cudaDeviceSynchronize()
    print(f"vram_hold: holding {held} MiB, free now {free_mib(a.gpu)} MiB", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
