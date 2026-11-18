import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from ttft_join import parse_admit

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


if __name__ == "__main__":
    unittest.main()
