import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("batching", Path(__file__).resolve().parents[1] / "37_continuous_batching.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class BatchingTests(unittest.TestCase):
    def test_known_cohort_and_refill(self):
        requests = [dict(arrival=0, tokens=n) for n in (1, 4, 1, 4)]
        self.assertEqual(lab.simulate(requests, 2, "fixed")["makespan_units"], 8)
        continuous = lab.simulate(requests, 2)
        self.assertEqual(continuous["makespan_units"], 6)
        self.assertEqual([r["first_token"] for r in continuous["requests"]], [1, 1, 2, 3])

    def test_arrivals_token_conservation_and_empty(self):
        requests = [dict(arrival=t, tokens=n) for t, n in ((5, 3), (1, 2), (5, 1), (20, 2))]
        for policy in ("fixed", "continuous"):
            for capacity in (1, 2, 8):
                report = lab.simulate(requests, capacity, policy)
                counts = [0]*len(requests)
                for iteration in report["trace"]:
                    self.assertLessEqual(len(iteration["request_ids"]), capacity)
                    for i in iteration["request_ids"]:
                        self.assertGreaterEqual(iteration["time"], requests[i]["arrival"])
                        counts[i] += 1
                self.assertEqual(counts, [r["tokens"] for r in requests])
        self.assertEqual(lab.simulate([])["makespan_units"], 0)


if __name__ == "__main__":
    unittest.main()
