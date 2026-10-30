"""taxameter.toml: the project's own settings checked against the repo (offline, no taxameter commands run)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from taxameter import config

ROOT = Path(__file__).resolve().parent.parent
SPEC = "bench/spec/phase0.toml"


@pytest.fixture(scope="module")
def cfg() -> config.Config:
    return config.load(ROOT / "taxameter.toml")


def _pin(key: str) -> str:
    return re.search(rf"^{key}=(\S+)", (ROOT / "vendor" / "PIN").read_text(encoding="utf-8"), re.M).group(1)


def test_push_paths_cover_workspace_members_and_build_rs_inputs(cfg):
    def covered(path: str) -> bool:
        return any(path == a or path.startswith(a + "/") for a in cfg.push.paths)

    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    for member in cargo["workspace"]["members"]:
        assert covered(member.removesuffix("/*")), member
    build_rs = (ROOT / "crates" / "ggml-sys" / "build.rs").read_text(encoding="utf-8")
    paths = re.findall(r'rerun-if-changed=((?:\.\./)+[^"]+)', build_rs)
    assert "../../vendor/PIN" in paths
    for p in paths:
        repo_path = p.replace("../", "")
        if any(repo_path.startswith(s) for s in cfg.push.submodules):
            continue
        assert covered(repo_path), repo_path


def test_image_tag_comes_from_vendor_pin(cfg):
    assert cfg.image_ref() == f"ghcr.io/lhns/tetraeder-box:{_pin('tag')}"


def test_bench_spec_llama_pin_matches_vendor_pin():
    spec = tomllib.loads((ROOT / SPEC).read_text(encoding="utf-8"))
    assert spec["pins"]["llama_cpp"] == _pin("commit")


def test_nightly_spec_yields_models(cfg):
    spec = tomllib.loads((ROOT / SPEC).read_text(encoding="utf-8"))
    expected = [m["hf"] for m in spec["model"].values() if m.get("hf")]
    assert expected
    assert cfg.model_specs() == expected


def test_config_loads_and_files_exist(cfg):
    # load() rejects unknown keys; the scripts it names must exist
    for rel in (cfg.nightly.script, cfg.box.provision_hook, cfg.box.verify_hook):
        assert (ROOT / rel).is_file(), rel
    assert cfg.min_cuda() == 12.8
