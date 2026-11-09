#!/usr/bin/env python3
"""Helper of the exit_run.sh stage `anchors-builder-short` (ADR 0056).

  ledger_stages.py <results dir> <stage>... [--copy-to DIR]
      One `--stage` argument for `tetraeder-plan ledger-dir` per builder stage that has a result
      `exit-<stage>-<stamp>.json` in the results dir (the newest), as
      `<stage>=<vram bytes>,<ram bytes>,<threads>,<ubatch>,<op offload>,<prompt>,<gen>[,KEY=VALUE...]`,
      the extras being the read path and prefetch the run used. With
      --copy-to the result is copied to `DIR/cand-<stage>.r1.json`, where the ledger finds it.
  ledger_stages.py --anchors <candidates.json>
      `<variant id> TAB <KEY=VALUE ...>` per anchor of a ledger directory's candidates.json.
"""
import glob
import json
import os
import re
import shutil
import sys


def stages(results, names, copy_to):
    for name in names:
        files = [
            f
            for f in glob.glob(os.path.join(results, f"exit-{name}-*.json"))
            if re.fullmatch(rf"exit-{re.escape(name)}-\d+\.json", os.path.basename(f))
        ]
        if not files:
            continue
        path = max(files, key=os.path.getmtime)
        try:
            j = json.load(open(path))
        except ValueError:
            continue
        o = j.get("options") or {}
        runs = [
            r
            for r in (j.get("runs") or [])
            if r.get("decode_tok_s") and not str(r.get("engine", "")).startswith("lib")
        ]
        if not runs or not o.get("expert_slot_bytes") or not o.get("ram_slot_bytes"):
            continue
        r = runs[-1]
        fields = [
            o["expert_slot_bytes"],
            o["ram_slot_bytes"],
            o.get("threads") or 1,
            r.get("ubatch") or 512,
            int(bool(o.get("op_offload", True))),
            r.get("prefill_tokens") or 0,
            r.get("decode_tokens") or 0,
        ]
        # the read path and the prefetch of the run, which the cost model reads from the variant's env
        extra = []
        if o.get("io_service"):
            extra.append("TETRAEDER_EXIT_IO_SERVICE=1")
            if o.get("io_workers"):
                extra.append(f"TETRAEDER_EXIT_IO_WORKERS={int(o['io_workers'])}")
        if o.get("overlap"):
            extra.append("TETRAEDER_EXIT_OVERLAP=1")
        if o.get("prefetch") is not None:
            extra.append(f"TETRAEDER_EXIT_PREFETCH={int(bool(o['prefetch']))}")
        print(f"{name}=" + ",".join([str(int(x)) for x in fields] + extra))
        if copy_to:
            os.makedirs(copy_to, exist_ok=True)
            shutil.copyfile(path, os.path.join(copy_to, f"cand-{name}.r1.json"))


def anchors(candidates):
    f = json.load(open(candidates))
    for v in f["anchors"]:
        env = " ".join(f"{k}={x}" for k, x in sorted(v["env"].items()))
        print(v["id"] + "\t" + env)


def main(argv):
    if argv[:1] == ["--anchors"] and len(argv) == 2:
        anchors(argv[1])
        return 0
    copy_to = None
    if "--copy-to" in argv:
        i = argv.index("--copy-to")
        copy_to = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    stages(argv[0], argv[1:], copy_to)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
