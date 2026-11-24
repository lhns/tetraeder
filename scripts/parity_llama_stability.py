#!/usr/bin/env python3
"""Is llama-server run-to-run stable on this GPU? (tree node serving-cuda-parity, diagnostic.)

For each parity case id, two fresh llama-server processes get the case (setup requests, then the compared request) and
each server then gets the same request a second time. The case's own comparison (parity_lib.compare) is applied between
the two fresh servers (`across_servers`) and between the first and second request on one server (`same_server`). An
empty list is a stable answer; the answers are stored for reading.

    python scripts/parity_llama_stability.py --llama target/llama-server/bin/llama-server --out stability.json ID...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PARITY = HERE.parent / "tests" / "conformance" / "parity"
sys.path.insert(0, str(PARITY))

import test_parity as tp  # noqa: E402
from parity_lib import compare  # noqa: E402
from servers import CTX, FAMILY, start_llama  # noqa: E402


def answer_dict(a) -> dict:
    return {"status": a.status, "body": a.body, "events": a.events}


def one_server(binary, case, engine, log_dir) -> tuple:
    ctx = case.get("ctx", CTX)
    srv = start_llama(binary, FAMILY[engine], log_dir, case.get("llama_args", ()), "", ctx)
    try:
        tp.run_setup(case, srv)
        first = tp.send(srv.base_url, case, None)
        second = tp.send(srv.base_url, case, None)
        return first, second
    finally:
        srv.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("ids", nargs="+")
    args = ap.parse_args()
    cases = {c["id"]: c for c in tp.CASES}
    result = {}
    for cid in args.ids:
        case = cases[cid]
        engine = "auto"
        with tempfile.TemporaryDirectory() as td:
            logs = pathlib.Path(td)
            s1a, s1b = one_server(pathlib.Path(args.llama), case, engine, logs)
            s2a, s2b = one_server(pathlib.Path(args.llama), case, engine, logs)
        entry = {
            "across_servers": compare(case, s1a, s2a),
            "same_server_1": compare(case, s1a, s1b),
            "same_server_2": compare(case, s2a, s2b),
            "answers": {"server1_first": answer_dict(s1a), "server1_second": answer_dict(s1b),
                        "server2_first": answer_dict(s2a), "server2_second": answer_dict(s2b)},
        }
        result[cid] = entry
        print(f"{cid}: across servers {'stable' if not entry['across_servers'] else 'DIFFERS'}, "
              f"same server {'stable' if not (entry['same_server_1'] or entry['same_server_2']) else 'DIFFERS'}", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
