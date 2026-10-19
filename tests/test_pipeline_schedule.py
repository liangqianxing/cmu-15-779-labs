import importlib.util
import itertools
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("pipeline", Path(__file__).resolve().parents[1] / "32_pipeline_schedule.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class PipelineTests(unittest.TestCase):
    def test_dependencies_coverage_and_stage_exclusion(self):
        for p, m, policy in itertools.product(range(1, 6), range(1, 9), ("gpipe", "1f1b")):
            schedule = lab.schedule_pipeline(p, m, policy=policy)
            tasks = {(t["stage"], t["phase"], t["microbatch"]): t for t in schedule["tasks"]}
            self.assertEqual(len(tasks), 2*p*m)
            for task in tasks.values():
                for dependency in task["dependencies"]:
                    self.assertLessEqual(tasks[tuple(dependency)]["finish"], task["start"])
            for stage in range(p):
                ordered = sorted((t for t in tasks.values() if t["stage"] == stage), key=lambda t: t["start"])
                for left, right in zip(ordered, ordered[1:]):
                    self.assertLessEqual(left["finish"], right["start"])
            expected_peaks = [m]*p if policy == "gpipe" else [min(m, p-stage) for stage in range(p)]
            self.assertEqual(schedule["peak_live_activation_units_per_stage"], expected_peaks)
            if policy == "gpipe":
                self.assertEqual(schedule["makespan_model_units"], (p + m - 1)*3)

    def test_heterogeneous_costs(self):
        for policy in ("gpipe", "1f1b"):
            schedule = lab.schedule_pipeline(3, 5, [1, 3, 2], [5, 2, 1], policy)
            self.assertTrue(all(0 < u <= 1 for u in schedule["stage_utilization"]))
            self.assertGreaterEqual(schedule["makespan_model_units"], 5*6)

    def test_partition_search_optimum_and_all_layers_assigned(self):
        report = lab.partition_search([1, 2, 1, 4, 1, 3, 1, 2], 3)
        self.assertEqual(report["candidates_evaluated"], 21)
        self.assertEqual(report["selected"]["max_stage_cost"], 6)
        for candidate in report["all_candidates"]:
            self.assertEqual(sum(candidate["stage_costs"]), 15)
        self.assertEqual(lab.partition_search([1, 2, 3], 1)["selected"]["stage_costs"], [6])
        self.assertEqual(lab.partition_search([1, 2, 3], 3)["selected"]["stage_costs"], [1, 2, 3])

    def test_invalid_parameters(self):
        for args in ((0, 2), (2, 0), (2, 3, [1], [1, 1]), (1, 2, [float("nan")], [1])):
            with self.assertRaises(ValueError):
                lab.schedule_pipeline(*args)
        with self.assertRaises(ValueError):
            lab.schedule_pipeline(policy="pipedream")
        for costs, stages in (([], 1), ([1, 0], 1), ([1, float("inf")], 1), ([1], 2)):
            with self.assertRaises(ValueError):
                lab.partition_search(costs, stages)


if __name__ == "__main__":
    unittest.main()
