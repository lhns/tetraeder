"""deploy/alerts/*.yml: structure, and that every expr uses only series the code emits or the rules record."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
RULE_FILES = sorted((ROOT / "deploy" / "alerts").glob("*.yml"))
# Metric names are declared in the telemetry crate and in some crates of their own (the gateway).
CRATES = ROOT / "crates"
# Name parts that -opentelemetry.usePrometheusNaming appends (unit, type, histogram series).
SUFFIX = re.compile(r"^(_(seconds|ratio|bytes|watts|celsius|segment|token|request|expert|layer))*(_total|_bucket|_sum|_count)?$")
TOKEN = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")
FUNCS = {
    "sum", "rate", "increase", "by", "without", "on", "ignoring", "group_left", "group_right", "and", "or",
    "unless", "vector", "avg", "max", "min", "count", "histogram_quantile", "abs", "clamp_min", "clamp_max",
    "avg_over_time", "max_over_time", "bool", "le", "offset",
}


def _emitted_bases() -> set[str]:
    text = "\n".join(f.read_text(encoding="utf-8") for f in CRATES.glob("*/src/**/*.rs"))
    return {m.replace(".", "_") for m in re.findall(r'"((?:tetraeder|gen_ai)\.[a-z_.]+)"', text)}


def _rules(path: Path) -> list[dict]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict) and isinstance(doc.get("groups"), list) and doc["groups"], path.name
    out = []
    for g in doc["groups"]:
        assert g.get("name") and isinstance(g.get("rules"), list) and g["rules"], g
        out.extend(g["rules"])
    return out


def _idents(expr: str) -> set[str]:
    # Drop string literals (label values) and label matchers before tokenizing.
    expr = re.sub(r'"[^"]*"|`[^`]*`', "", expr)
    expr = re.sub(r"\{[^}]*\}", "", expr)
    expr = re.sub(r"\[[^\]]*\]", "", expr)
    # Drop grouping label lists: by (a, b), on(sli).
    expr = re.sub(r"\b(?:by|without|on|ignoring)\s*\([^)]*\)", "", expr)
    return {t for t in TOKEN.findall(expr) if t not in FUNCS}


def test_rule_files_exist():
    assert {p.name for p in RULE_FILES} >= {"engine.yml", "gateway.yml", "slo.yml"}


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda p: p.name)
def test_structure(path):
    for r in _rules(path):
        assert ("alert" in r) != ("record" in r), r
        assert isinstance(r.get("expr"), str) and r["expr"].strip(), r
        if "alert" in r:
            assert r.get("labels", {}).get("severity"), r["alert"]
            assert r.get("annotations", {}).get("summary"), r["alert"]


def test_expr_series_are_emitted_or_recorded():
    bases = _emitted_bases()
    assert "tetraeder_plan_residual" in bases, "crate sources not parsed"
    recorded = {r["record"] for f in RULE_FILES for r in _rules(f) if "record" in r}
    for f in RULE_FILES:
        for r in _rules(f):
            for ident in _idents(r["expr"]):
                if ":" in ident:
                    assert ident in recorded, f"{f.name}: {r.get('alert') or r['record']} uses undefined {ident}"
                    continue
                ok = any(ident.startswith(b) and SUFFIX.match(ident[len(b):]) for b in bases)
                assert ok, f"{f.name}: {r.get('alert') or r['record']} uses unknown series or label {ident}"


def test_slo_burn_windows():
    rules = {r["alert"]: r["expr"] for r in _rules(ROOT / "deploy" / "alerts" / "slo.yml") if "alert" in r}
    for name, rate, w1, w2 in [
        ("TetraederSloBurnFast", "14.4", "1h", "5m"),
        ("TetraederSloBurnSlow", "6", "6h", "30m"),
        ("TetraederSloBurnTicket", "1", "3d", "6h"),
    ]:
        e = rules[name]
        assert f"rate{w1} " in e and f"rate{w2} " in e and e.count(f"> {rate}") == 2, name


def test_dashboards_are_valid_json_with_unique_panel_ids():
    import json

    for p in (ROOT / "deploy" / "grafana" / "dashboards").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        ids = [x["id"] for x in d["panels"]]
        assert len(ids) == len(set(ids)), p.name
