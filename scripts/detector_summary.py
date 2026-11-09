"""Markdown summary of a detector score.json (tetraeder detector-score), for the nightly job summary."""

import json
import sys

s = json.load(open(sys.argv[1]))
print("## Fault testbed")
print(f"bars met: {s['pass']} (reported only; runner noise)\n")
for k in ("single_top1", "no_fault", "pair_top3"):
    print(f"- {k}: {s[k]['ok']} / {s[k]['runs']}")
print("\n| run | label | found | ok |\n|---|---|---|---|")
for r in s["runs"]:
    print(f"| {r['id']} | {r['label']} | {', '.join(r['found']) or '-'} | {'yes' if r['ok'] else 'MISS'} |")
