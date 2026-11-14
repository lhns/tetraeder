"""Reading a process's smaps_rollup stalls its munmap (ADR 0060 addendum 2026-11-14), the suspected cause of the
one-window decode stall: the host sampler read it every tick while the builder held an 80 GB RAM tier on 4 KiB pages.

Linux only. A 1 GiB anonymous mapping on 4 KiB pages stands in for the tier; a second process reads smaps_rollup in
a loop, as the sampler did, while this one times mmap plus munmap of 1 MiB. Hardware label: none (any Linux runner).
"""

import mmap
import os
import statistics
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("/proc/self/smaps_rollup"), reason="needs Linux smaps_rollup")

TIER = 1 << 30
SECONDS = 1.5


def map_unmap_ms(seconds):
    """Wall time in ms of each mmap, touch and munmap of 1 MiB over `seconds`."""
    out = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t = time.perf_counter()
        m = mmap.mmap(-1, 1 << 20)
        m[0] = 1
        m.close()
        out.append((time.perf_counter() - t) * 1e3)
        time.sleep(0.001)
    return out


def test_smaps_rollup_reader_stalls_munmap_of_the_read_process():
    tier = mmap.mmap(-1, TIER)
    if hasattr(mmap, "MADV_NOHUGEPAGE"):
        tier.madvise(mmap.MADV_NOHUGEPAGE)
    for off in range(0, TIER, mmap.PAGESIZE):
        tier[off] = 1
    base = map_unmap_ms(SECONDS)
    reader = subprocess.Popen([sys.executable, "-c",
                               "import sys, time\n"
                               "end = time.time() + float(sys.argv[2])\n"
                               "n = 0\n"
                               "while time.time() < end:\n"
                               "    open(f'/proc/{sys.argv[1]}/smaps_rollup').read(); n += 1\n"
                               "print(n)\n",
                               str(os.getpid()), str(SECONDS + 0.5)], stdout=subprocess.PIPE, text=True)
    time.sleep(0.2)
    loaded = map_unmap_ms(SECONDS)
    reads = int(reader.communicate()[0])
    tier.close()
    b, l = statistics.median(base), statistics.median(loaded)
    print(f"mmap+munmap 1 MiB: median {b:.3f} ms alone, {l:.3f} ms beside {reads} smaps_rollup reads; "
          f"max {max(base):.2f} vs {max(loaded):.2f} ms")
    # Each read walks the whole tier under the mmap lock, so a munmap waits for the rest of the walk.
    assert l > 3 * b, (b, l)
