import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from ttft_join import admits, parse_admit, parse_step_save

OLD = ("\x1b[2m2026\x1b[0m DEBUG admit seq=1 queued_ms=0.10 place_ms=2.00 save_ms=0.00 prompt=500 common=400 "
       "source=live live=1 pool_spare=2 pool_fallback=0 pool_ready=3")
NEW = ("admit seq=2 queued_ms=0.10 place_ms=2.00 save_ms=0.00 restore_ms=12.50 prompt=500 common=400 "
       "source=ram live=1 pool_spare=2 pool_fallback=0 pool_ready=3 skipped=own_output")


class ParseAdmit(unittest.TestCase):
    def test_old_format(self):
        r = parse_admit(OLD.split("DEBUG ")[1])
        self.assertEqual((r["seq"], r["prompt"], r["common"], r["pool_ready"]), (1, 500, 400, 3))
        self.assertIsNone(r["restore_ms"])
        self.assertIsNone(r["skipped"])

    def test_new_format(self):
        r = parse_admit(NEW)
        self.assertEqual(r["restore_ms"], 12.5)
        self.assertEqual(r["skipped"], "own_output")
        self.assertEqual(r["source"], "ram")

    def test_other_lines(self):
        self.assertIsNone(parse_admit("WARN something else"))


def admit_line(seq, prompt):
    return (f"admit seq={seq} queued_ms=0.10 place_ms=2.00 save_ms=1.00 restore_ms=0.00 prompt={prompt} common=0 "
            "source=none live=1 pool_spare=0 pool_fallback=0 pool_ready=0")


def log_rows(lines):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.log")
        open(p, "w").write("\n".join(lines) + "\n")
        return admits(p)


class StepSave(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_step_save("step save reason=preempt seq=1 save_ms=12.50 prefilling=0,2"), (12.5, [0, 2]))
        self.assertEqual(parse_step_save("step save reason=evict seq=1 save_ms=3.00 prefilling="), (3.0, []))
        self.assertIsNone(parse_step_save("step dec_seqs=1 save_ms=3"))

    def test_counts_against_prefilling_slots_only(self):
        rows, keep = log_rows([admit_line(0, 100), admit_line(1, 200),
                               "step save reason=preempt seq=1 save_ms=40.00 prefilling=1",
                               "step save reason=evict seq=0 save_ms=5.00 prefilling=0,1",
                               "step save reason=evict seq=0 save_ms=9.00 prefilling=",
                               admit_line(0, 300)])
        self.assertEqual([r.get("step_save_ms") for r in rows], [5.0, 45.0, None])
        self.assertEqual(len(keep), 6)

    def test_old_log_has_no_step_field(self):
        rows, keep = log_rows([OLD.split("DEBUG ")[1], "WARN x"])
        self.assertNotIn("step_save_ms", rows[0])
        self.assertEqual(len(keep), 2)

    def test_join_output_unchanged_without_step_lines_and_counts_them_with(self):
        script = os.path.join(os.path.dirname(__file__), "ttft_join.py")
        res = [dict(index=0, status=200, ttft_s=1.0, total_s=2.0, usage=dict(input_tokens=100))]
        with tempfile.TemporaryDirectory() as d:
            rp = os.path.join(d, "r.json")
            json.dump(dict(results=res), open(rp, "w"))

            def run(lines):
                lp = os.path.join(d, "s.log")
                open(lp, "w").write("\n".join(lines) + "\n")
                return subprocess.run([sys.executable, script, rp, lp], capture_output=True, text=True, check=True).stdout

            plain = run([admit_line(0, 100)])
            self.assertIn("sum save_ms 0.00 s;", plain)
            self.assertNotIn("in-step", plain)
            with_step = run([admit_line(0, 100), "step save reason=evict seq=0 save_ms=500.00 prefilling=0"])
            self.assertIn("sum save_ms 0.50 s (of which in-step 0.50 s);", with_step)


if __name__ == "__main__":
    unittest.main()
