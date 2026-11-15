"""scripts/loop_bench_summary.py on fake result files."""

import importlib.util
import json
import pathlib

spec = importlib.util.spec_from_file_location(
    "loop_bench_summary", pathlib.Path(__file__).parent.parent / "scripts" / "loop_bench_summary.py"
)
lbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lbs)


def write(d, engine, test, rep, thr, dec, errors=0):
    s = {"throughput_tps": thr, "decode_tps_median": dec, "errors": errors}
    (d / f"{engine}-{test}-r{rep}.json").write_text(json.dumps({"summary": s}))


def test_median_over_repeats_and_ratio_to_llama_server(tmp_path):
    write(tmp_path, "llama-server", "c4", 1, 40.0, 12.0)
    write(tmp_path, "llama-server", "c4", 2, 60.0, 14.0)
    write(tmp_path, "tetraeder-dense", "c4", 1, 25.0, 6.0)
    write(tmp_path, "tetraeder-dense", "c4", 2, 25.0, 6.0, errors=1)
    write(tmp_path, "tetraeder-dense", "c1", 1, 10.0, 10.0)
    s = lbs.summarize(lbs.load_runs(str(tmp_path)))
    assert s["c4"]["llama-server"]["throughput_tps"] == 50.0
    d = s["c4"]["tetraeder-dense"]
    assert d["errors"] == 1 and d["runs"] == 2
    assert d["ratio"]["throughput_tps"] == 0.5
    assert d["ratio"]["decode_tps_median"] == 6.0 / 13.0
    assert s["c1"]["tetraeder-dense"]["ratio"] == {"throughput_tps": None, "decode_tps_median": None}


def test_main_writes_summary_and_table(tmp_path, capsys):
    write(tmp_path, "llama-server", "c1", 1, 10.0, 10.0)
    write(tmp_path, "tetraeder-llama", "c1", 1, 9.0, 9.0)
    assert lbs.main([str(tmp_path)]) == 0
    assert (tmp_path / "summary.json").exists()
    assert "| c1 | tetraeder-llama |" in capsys.readouterr().out


def test_main_fails_without_results(tmp_path):
    assert lbs.main([str(tmp_path)]) == 1


def test_script_runs_the_baseline_with_flash_attention_and_a_moe_pass():
    text = (pathlib.Path(__file__).parent.parent / "scripts" / "loop_bench.sh").read_text()
    cmd = text.split('"$llama_server"', 1)[1].split("ready=", 1)[0]
    assert "-fa on" in cmd
    assert "run_model llama-server tetraeder-llama tetraeder-dense" in text
    assert "run_model llama-server tetraeder-llama tetraeder-moe" in text
    assert "tiny-qwen3moe-f16.gguf" in text
