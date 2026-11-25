#!/usr/bin/env python3
"""Verdict of Case F, the Phase 6 failover exit test (ADR 0091 "Case F"), every pass rule as code.

    phase6_exit_f_summary.py RESULTS_DIR                  writes RESULTS_DIR/verdict-f.json, prints the tables
    phase6_exit_f_summary.py admits NODE_LOG [--out F]    epochs of the `admit` lines of a node log (plain text)

Layout of RESULTS_DIR (written by scripts/phase6_exit_f.sh; the tests build it from the rehearsal data and from
synthetic trials):

    session.json          boxes, model sha256, load shape, bound, clock offset, creation epochs, replica names
    gates/<box>/          lscpu.txt, cotenant.json, gates.json {power_ok, idle_ok}  (box a and box b)
    head-requests.jsonl   the head's request log (whole session; .gz also read)
    trials/<mode>-<n>/    meta.json, load.jsonl, b-admits.txt
    manual.jsonl          operator actions, one {"t": epoch, "what": text} per line

Pass rules per trial (ADR 0091 Case F, "Pass, per trial"; no speed is judged):
  clean_error_in_bound  every request in flight on R_B at the fault ends with one clean error within bound
                        (node_down_after + 1 s = 3 s); at least one was in flight (else the fault hit nothing)
  no_failure_on_r_a     every failure names R_B (the router's error says "replica <id> ..."), and none fell outside the
                        in-flight set or the first bound seconds
  ok_after_bound        every request submitted from bound after the fault succeeded, and there were some
  queued_move           every request submitted in the first bound seconds, i.e. waiting at the router for R_B or
                        racing the detection, succeeded or ended with a clean error naming R_B within the bound
  exactly_once          each request has one outcome on the client (no token after an error, no missing [DONE]) and the
                        head's request log has one entry per client request with the same outcome, none besides
  rejoin                B serves a request within rejoin_limit (60 s) of its restore, searched only up to
                        restore + limit; the epoch is the first `admit` on B at or after the restore
  trial1_no_manual      trial 1 of a mode needs no operator action between the fault and the rejoin deadline
  b_served_before       B admitted a request in the 60 s before the fault (both replicas were in service)
Session rules: gates on both boxes (one NUMA node, co-tenant by engine_sample.py, power, idle), nine complete trials,
the same model file on both boxes, certificates pinned and cluster.toml written after the boxes were created, the load
shape, and the clock offset between the boxes known within 0.5 s. A rule whose evidence is missing fails.
Pass only if every rule holds in all trials of all three modes.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_gates as sg  # noqa: E402

MODES = ["worker", "daemon", "stop"]  # ADR 0091 fault (1), (2), (3)
TRIALS = 3
BOUND_S = 3.0  # node_down_after (2 s) + 1 s
REJOIN_S = 60.0
LOAD_S = 180.0
CONCURRENCY = 8
PROMPT_TOKENS = 512
PROMPT_TOL = 0.05
MAX_TOKENS = 256
CLOCK_UNCERTAINTY_MAX_S = 0.5
FINISHES = ("stop", "length", "tool_calls")

# the reading of a failed rule, per trial rule
LABELS = {
    "clean_error_in_bound": "the clean error within 3 s",
    "no_failure_on_r_a": "failures off R_B",
    "ok_after_bound": "requests after the bound",
    "queued_move": "queued requests",
    "exactly_once": "exactly-once",
    "rejoin": "rejoin",
    "trial1_no_manual": "manual action in trial 1",
    "b_served_before": "B serving before the fault",
}
RULES = list(LABELS)


def opener(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def read_jsonl(path: str) -> list[dict]:
    with opener(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def first_existing(*paths: str) -> str | None:
    return next((p for p in paths if os.path.isfile(p)), None)


# ---- request outcomes (as scripts/phase6_casef.py classifies them) -----------------------------------------------------


def classify(r: dict) -> str:
    err = r.get("error")
    if err is None and r.get("done") and r.get("finish") in FINISHES:
        return "ok"
    if err is not None and not (r.get("done") and r.get("finish") in ("stop", "length")):
        return "error"
    return "ambiguous"


def clean_error(r: dict) -> bool:
    """One error and nothing after it: an HTTP error before the stream, or one error event that ended it."""
    if r.get("status") not in (None, 200):
        return True
    return r.get("error") is not None and r.get("after_error", 0) == 0 and not r.get("finish")


def names_b(r: dict, replica_b: str) -> bool:
    return replica_b in (r.get("error") or "")


# ---- the head's request log against the client --------------------------------------------------------------------------


def nonce_of(entry: dict) -> tuple[str, int] | None:
    """(label, n) of a log entry from the `[label-n] ` the load puts in front of its prompt."""
    req = entry.get("request") or {}
    msgs = req.get("messages")
    text = req.get("prompt") if isinstance(req.get("prompt"), str) else ""
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and isinstance(msgs[0].get("content"), str):
        text = msgs[0]["content"]
    m = re.match(r"\[(.+)-(\d+)\] ", text[:80])
    return (m.group(1), int(m.group(2))) if m else None


def gateway_agreement(rows: list[dict], gw: list[dict], label: str) -> tuple[bool, str]:
    """One head log entry per client request of this load, with the same outcome; no entry without a client request."""
    by: dict[int, list[dict]] = {}
    for e in gw:
        k = nonce_of(e)
        if k and k[0] == label and (e.get("path") or "").endswith("/chat/completions"):
            by.setdefault(k[1], []).append(e)
    bad = []
    for r in rows:
        es = by.pop(r["n"], [])
        if len(es) != 1:
            bad.append(f"request {r['n']} has {len(es)} gateway entries")
            continue
        g_err = es[0].get("error_type") is not None or es[0].get("status") != 200
        if g_err != (classify(r) != "ok"):
            bad.append(f"request {r['n']}: client {classify(r)}, gateway {'error' if g_err else 'ok'}")
    bad += [f"gateway entry {n} has no client request" for n in by]
    return (not bad), (f"{len(rows)} client requests, {len(rows)} gateway entries, same outcomes" if not bad else "; ".join(bad[:5]))


# ---- one trial ----------------------------------------------------------------------------------------------------------


def judge_trial(rows: list[dict], meta: dict, gw: list[dict] | None, b_admits: list[float] | None, manual: list[dict],
                bound: float, rejoin_limit: float, replica_b: str) -> dict:
    """Applies the per-trial rules; `rows` are the client's, epochs on the client's (box A's) clock."""
    for r in rows:
        r["class"] = classify(r)
    tf = meta.get("t_fault")
    tr = meta.get("t_restore")
    rules: dict[str, dict] = {}

    def rule(name: str, ok: bool, detail: str) -> None:
        rules[name] = {"ok": bool(ok), "detail": detail}

    if not rows or tf is None:
        for n in RULES:
            rule(n, False, "no client requests or no fault epoch")
        return {"rules": rules, "stats": {}}
    infl = [r for r in rows if r["t_submit"] < tf <= (r["t_end"] or 1e18)]
    infl_err = [r for r in infl if r["class"] != "ok"]
    early = [r for r in rows if tf <= r["t_submit"] < tf + bound]
    after = [r for r in rows if r["t_submit"] >= tf + bound]
    delays = sorted(round(r["t_end"] - tf, 3) for r in infl_err)
    stats = {"requests": len(rows), "in_flight_at_fault": len(infl), "in_flight_errors": len(infl_err),
             "in_flight_error_delay_s": delays, "submitted_in_bound": len(early), "submitted_after_bound": len(after)}

    rule("clean_error_in_bound", bool(infl_err) and all(d <= bound for d in delays) and all(clean_error(r) for r in infl_err),
         f"{len(infl_err)} of {len(infl)} in flight ended with an error, delays {delays} s against {bound}"
         + ("" if infl_err else " (none in flight on B: the fault hit nothing)"))
    off_b = [r for r in rows if r["class"] != "ok" and not names_b(r, replica_b)]
    stray = [r for r in rows if r["class"] != "ok" and r not in infl and r not in early]
    rule("no_failure_on_r_a", not off_b and not stray,
         f"{len(off_b)} failures do not name {replica_b!r}, {len(stray)} outside the in-flight set and the first {bound} s")
    bad_after = [r for r in after if r["class"] != "ok"]
    rule("ok_after_bound", bool(after) and not bad_after, f"{len(bad_after)} of {len(after)} failed")
    bad_early = [r for r in early if r["class"] != "ok"
                 and not (names_b(r, replica_b) and clean_error(r) and r["t_end"] <= tf + bound)]
    rule("queued_move", not bad_early, f"{len(early)} submitted in the first {bound} s, {len(bad_early)} neither ok nor a clean error")
    after_err = sum(r.get("after_error", 0) for r in rows)
    amb = sum(r["class"] == "ambiguous" for r in rows)
    if gw is None:
        rule("exactly_once", False, "no request log of the head")
    else:
        agree, why = gateway_agreement(rows, gw, meta.get("load_label", ""))
        rule("exactly_once", amb == 0 and after_err == 0 and agree,
             f"{amb} ambiguous, {after_err} chunks after an error; {why}")
    if tr is None or b_admits is None:
        rule("rejoin", False, "no restore epoch or no admit lines of B")
        rule("b_served_before", False, "no admit lines of B")
    else:
        # only the window restore .. restore + limit counts: an admit after a manual head restart is none
        hit = [a for a in b_admits if tr <= a <= tr + rejoin_limit]
        until = meta.get("load_end")
        rule("rejoin", bool(hit),
             (f"B admitted at +{min(hit) - tr:.2f} s after its restore" if hit else
              f"no admit on B within {rejoin_limit} s of the restore"
              + (", and the load ended before the window closed" if until is not None and until < tr + rejoin_limit else "")))
        stats["rejoin_s"] = round(min(hit) - tr, 2) if hit else None
        pre = [a for a in b_admits if tf - 60 <= a < tf]
        rule("b_served_before", bool(pre), f"{len(pre)} admits on B in the 60 s before the fault")
    if meta.get("trial") == 1 and tr is not None:
        acts = [m for m in manual if tf <= m.get("t", 0) <= tr + rejoin_limit]
        rule("trial1_no_manual", not acts, "; ".join(m.get("what", "?") for m in acts) or "no operator action")
    elif meta.get("trial") == 1:
        rule("trial1_no_manual", False, "no restore epoch")
    else:
        rule("trial1_no_manual", True, "not trial 1")
    return {"rules": rules, "stats": stats}


# ---- the session --------------------------------------------------------------------------------------------------------


def ordinal_list(nums: list[int]) -> str:
    s = [str(n) for n in nums]
    return s[0] if len(s) == 1 else ", ".join(s[:-1]) + " and " + s[-1]


def reading(per_mode_fail: dict[str, set[int]]) -> str:
    """'modes 1 and 2 fail on rejoin' for each failing rule, in rule order."""
    parts = []
    for r in RULES:
        modes = sorted(per_mode_fail.get(r, ()))
        if modes:
            parts.append(f"mode{'s' if len(modes) > 1 else ''} {ordinal_list(modes)} fail{'' if len(modes) > 1 else 's'} on {LABELS[r]}")
    return "; ".join(parts) if parts else "every rule holds in every mode"


def session_rules(res: str, sess: dict) -> list[dict]:
    out: list[dict] = []

    def rule(name: str, ok: bool, detail: str) -> None:
        out.append({"name": name, "ok": bool(ok), "detail": detail})

    for box in ("a", "b"):
        g = os.path.join(res, "gates", box)
        why: list[str] = []
        try:
            why += sg.check_numa(open(os.path.join(g, "lscpu.txt"), encoding="utf-8").read())
        except OSError:
            why.append("no lscpu.txt")
        why += sg.check_cotenant(os.path.join(g, "cotenant.json"))
        try:
            gj = load_json(os.path.join(g, "gates.json"))
            for k in ("power_ok", "idle_ok"):
                if gj.get(k) is not True:
                    why.append(f"{k} is {gj.get(k)!r}")
        except (OSError, ValueError):
            why.append("no gates.json")
        rule(f"gates_{box}", not why, "; ".join(why) or "one NUMA node, co-tenant under the limit, power and idle gates passed")
    shas = {b: (sess.get("boxes", {}).get(b) or {}).get("model_sha256") for b in ("a", "b")}
    rule("same_model", bool(shas["a"]) and shas["a"] == shas["b"] and shas["a"] == sess.get("model_sha256"),
         f"a {shas['a']}, b {shas['b']}, expected {sess.get('model_sha256')}")
    certs = sess.get("certs") or {}
    rule("certificates_pinned", bool(certs.get("a")) and bool(certs.get("b")) and certs.get("pinned_on") == ["a", "b"],
         f"certificate sha256 a {certs.get('a')}, b {certs.get('b')}, pinned on {certs.get('pinned_on')}")
    ta, tb, tc = (sess.get("t_created") or {}).get("a"), (sess.get("t_created") or {}).get("b"), sess.get("t_cluster_toml")
    rule("cluster_toml_after_create", None not in (ta, tb, tc) and tc > max(ta, tb),
         f"created at {ta} and {tb}, cluster.toml written at {tc}")
    ld = sess.get("load") or {}
    pt = ld.get("prompt_tokens")
    shape_ok = (ld.get("concurrency") == CONCURRENCY and ld.get("max_tokens") == MAX_TOKENS and (ld.get("duration_s") or 0) >= LOAD_S
                and pt is not None and abs(pt / PROMPT_TOKENS - 1) <= PROMPT_TOL and ld.get("stream") is True)
    rule("load_shape", shape_ok, f"concurrency {ld.get('concurrency')}, {ld.get('duration_s')} s, prompt {pt} tokens against "
         f"{PROMPT_TOKENS} +-{PROMPT_TOL:.0%}, {ld.get('max_tokens')} output tokens, stream {ld.get('stream')}")
    unc = (sess.get("clock") or {}).get("uncertainty_s")
    rule("clock_offset", unc is not None and unc <= CLOCK_UNCERTAINTY_MAX_S,
         f"offset {(sess.get('clock') or {}).get('b_minus_a_s')} s, uncertainty {unc} s against {CLOCK_UNCERTAINTY_MAX_S}")
    return out


def trial_dirs(res: str) -> dict[tuple[str, int], str]:
    found = {}
    for d in glob.glob(os.path.join(res, "trials", "*-*")):
        m = re.fullmatch(r"(worker|daemon|stop)-(\d+)", os.path.basename(d))
        if m and os.path.isdir(d):
            found[(m.group(1), int(m.group(2)))] = d
    return found


def summarize(res: str) -> dict:
    sess = load_json(os.path.join(res, "session.json"))
    bound = sess.get("bound_s", BOUND_S)
    limit = sess.get("rejoin_limit_s", REJOIN_S)
    replica_b = sess.get("replica_b", "replica 1")
    trials_per_mode = sess.get("trials_per_mode", TRIALS)
    gwp = first_existing(os.path.join(res, "head-requests.jsonl"), os.path.join(res, "head-requests.jsonl.gz"))
    gw = read_jsonl(gwp) if gwp else None
    mp = os.path.join(res, "manual.jsonl")
    manual = read_jsonl(mp) if os.path.isfile(mp) else []
    dirs = trial_dirs(res)
    out: dict = {"case": "F", "bound_s": bound, "rejoin_limit_s": limit, "trials": []}
    fails: dict[str, set[int]] = {}
    missing = []
    for mi, mode in enumerate(MODES, 1):
        for n in range(1, trials_per_mode + 1):
            d = dirs.get((mode, n))
            if d is None:
                missing.append(f"{mode}-{n}")
                continue
            try:
                meta = load_json(os.path.join(d, "meta.json"))
                rows = read_jsonl(os.path.join(d, "load.jsonl"))
            except (OSError, ValueError) as e:
                missing.append(f"{mode}-{n} ({e})")
                continue
            meta.setdefault("trial", n)
            meta.setdefault("mode", mode)
            ap = os.path.join(d, "b-admits.txt")
            off = (sess.get("clock") or {}).get("b_minus_a_s") or 0.0  # B's log is on B's clock, the trial on A's
            admits = [float(x) - off for x in open(ap, encoding="utf-8").read().split()] if os.path.isfile(ap) else None
            t = judge_trial(rows, meta, gw, admits, manual, bound, limit, replica_b)
            t.update(mode=mode, mode_number=mi, trial=n, ok=all(r["ok"] for r in t["rules"].values()))
            out["trials"].append(t)
            for name, r in t["rules"].items():
                if not r["ok"]:
                    fails.setdefault(name, set()).add(mi)
    rules = session_rules(res, sess)
    out["session_rules"] = rules
    out["missing_trials"] = missing
    out["reading"] = reading(fails)
    out["manual_actions"] = manual
    complete = not missing
    trials_ok = bool(out["trials"]) and all(t["ok"] for t in out["trials"])
    session_ok = all(r["ok"] for r in rules)
    out["pass"] = bool(complete and trials_ok and session_ok)
    out["verdict"] = "pass" if out["pass"] else ("incomplete" if missing else "fail")
    bad = [f"{t['mode']}-{t['trial']}.{k}" for t in out["trials"] for k, r in t["rules"].items() if not r["ok"]]
    bad += [r["name"] for r in rules if not r["ok"]]
    out["status"] = (f"{out['verdict']}: {out['reading']}" + (f"; missing {', '.join(missing)}" if missing else "")
                     + (f"; failing {', '.join(bad[:12])}" + (" ..." if len(bad) > 12 else "") if bad else ""))
    return out


def print_tables(o: dict) -> None:
    print("status:", o["status"])
    for t in o["trials"]:
        s = t["stats"]
        print(f"  mode {t['mode_number']} {t['mode']:6} trial {t['trial']}: in flight {s.get('in_flight_at_fault')} errors "
              f"{s.get('in_flight_errors')} delays {s.get('in_flight_error_delay_s')} rejoin_s {s.get('rejoin_s')} "
              f"{'ok' if t['ok'] else 'FAIL ' + ', '.join(k for k, r in t['rules'].items() if not r['ok'])}")
    for r in o["session_rules"]:
        print(f"  rule {r['name']:26} {'ok  ' if r['ok'] else 'FAIL'} {r['detail']}")


# ---- admit lines of a node log -------------------------------------------------------------------------------------------

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ADMIT = re.compile(r"^(\d{4}-\d\d-\d\dT[\d:.]+)Z\s+DEBUG tetraeder::step: admit ")


def admit_epochs(text: str) -> list[float]:
    out = []
    for line in ANSI.sub("", text).splitlines():
        m = ADMIT.match(line)
        if m:
            d = datetime.datetime.fromisoformat(m.group(1)).replace(tzinfo=datetime.timezone.utc)
            out.append(round(d.timestamp(), 6))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.target == "admits":
        text = open(a.arg, encoding="utf-8", errors="replace").read()
        eps = admit_epochs(text)
        body = "".join(f"{e:.6f}\n" for e in eps)
        if a.out:
            open(a.out, "w", encoding="utf-8").write(body)
        else:
            sys.stdout.write(body)
        return 0
    o = summarize(a.target)
    with open(os.path.join(a.target, "verdict-f.json"), "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)
    print_tables(o)
    return 0 if o["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
