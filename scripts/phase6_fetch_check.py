#!/usr/bin/env python3
"""The check before a Case S box is destroyed (docs/box-driver.md "Stage by stage").

`taxameter destroy` refuses while its run list holds unfetched runs, and that list carries the committed bench/results dirs, so
every destroy of this search needs `--force`, which skips taxameter's own fetch check entirely. This is the check in its place:
after `taxameter fetch --id <id>` of both boxes, before `destroy --force`.

  python3 scripts/phase6_fetch_check.py NAME [--stage-reached predict|summary] [--root bench/results]

Passes (exit 0) when, under ROOT:
  NAME-na/DONE and NAME-nb/DONE exist (each box's sc_box.sh results came back; sc_box.sh writes DONE after every run and abort)
  NAME/session.json exists (the driver's own record)
  from predict on: NAME/predictions/SHA256SUMS lists files that all exist and match (paths relative to NAME), and
                   NAME/predictions/FROZEN.sha256 matches SHA256SUMS
  from summary on: NAME/verdict-s.json exists
Before any sc_box.sh run (a session stopped at the gates or the bulk screen) there is nothing on the boxes to fetch: pass
--stage-reached gates and only session.json and the gates/ files are checked.
"""

import argparse
import hashlib
import pathlib
import sys

ORDER = ("gates", "setup", "predict", "summary")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_sums(base, sums):
    """[problems] of a sha256sum file whose paths are relative to `base`."""
    bad = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        f = base / name.lstrip("*")
        if not f.exists():
            bad.append(f"missing {f}")
        elif sha(f) != digest:
            bad.append(f"hash differs {f}")
    return bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("name")
    p.add_argument("--stage-reached", choices=ORDER, default="summary")
    p.add_argument("--root", default="bench/results")
    a = p.parse_args(argv)
    root, n = pathlib.Path(a.root), a.name
    out = root / n
    reach = ORDER.index(a.stage_reached)
    bad = []
    if not (out / "session.json").exists():
        bad.append(f"missing {out / 'session.json'}")
    if reach == 0:
        for b in ("a", "b"):
            if not (out / "gates" / b / "gates.json").exists():
                bad.append(f"missing {out / 'gates' / b / 'gates.json'}")
    if reach >= 1:
        for b in ("a", "b"):
            if not (root / f"{n}-n{b}" / "DONE").exists():
                bad.append(f"missing {root / f'{n}-n{b}' / 'DONE'} (fetch box {b} first)")
    if reach >= 2:
        sums = out / "predictions" / "SHA256SUMS"
        if not sums.exists():
            bad.append(f"missing {sums}")
        else:
            bad += check_sums(out, sums)
            bad += check_sums(out / "predictions", out / "predictions" / "FROZEN.sha256")
    if reach >= 3 and not (out / "verdict-s.json").exists():
        bad.append(f"missing {out / 'verdict-s.json'}")
    for b in bad:
        print("FETCH CHECK:", b)
    print("fetch check ok: destroy --force may follow" if not bad else "fetch check FAILED: do not destroy yet")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
