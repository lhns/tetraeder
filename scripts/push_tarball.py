"""Write the tarball `taxameter run/provision` would push (committed HEAD, [push] paths of taxameter.toml).

Calls taxameter's own git_archive, so the content is exactly what the box receives.
Usage: uv run python scripts/push_tarball.py OUT.tgz [MAX_MB]   (exit 1 when over MAX_MB, default 5)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from taxameter import config
from taxameter.core import git_archive

out = Path(sys.argv[1])
max_mb = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
cfg = config.load(Path(__file__).resolve().parent.parent / "taxameter.toml")
data = git_archive(SimpleNamespace(cfg=cfg, root=cfg.root))
out.write_bytes(data)
mb = len(data) / 1048576
print(f"push tarball: {len(data)} bytes ({mb:.2f} MB), limit {max_mb} MB")
if mb > max_mb:
    print("ERROR: the push is over the limit (the box push goes through a slow ssh proxy, 55 min for 50 MB)")
    sys.exit(1)
