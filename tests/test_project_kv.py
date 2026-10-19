import importlib.util
from pathlib import Path
import unittest

import torch

SPEC = importlib.util.spec_from_file_location("project", Path(__file__).resolve().parents[1] / "course/project/run_experiments.py")
project = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(project)
torch.set_num_threads(1)


class ProjectTests(unittest.TestCase):
    def test_all_policies_and_block_boundaries_match_attention_reference(self):
        for common in (0, 1, 3, 4, 7, 8):
            workload = project.make_workload(dict(name="test", common=common, prompt_base=max(1, common)), 4100, requests=4, width=3)
            expected = project.expected_outputs(workload)
            for policy in ("static", "dynamic", "paged", "shared"):
                for block in (2, 4, 8):
                    actual, metrics = project.execute(workload, policy, block)
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    self.assertGreater(metrics["active_slot_bytes"], 0)

    def test_full_page_sharing_and_partial_page_copy_accounting(self):
        for scenario in project.SCENARIOS:
            workload = project.make_workload(scenario, 41)
            for block in (2, 4, 8, 16, 32, 64):
                _, independent = project.execute(workload, "paged", block)
                _, shared = project.execute(workload, "shared", block)
                self.assertEqual(independent["page_count"]-shared["page_count"], 7*(scenario["common"]//block))
                self.assertEqual(shared["copied_old_token_slots"], 7*(scenario["common"] % block))
                self.assertEqual(shared["pool_tensor_bytes"], independent["pool_tensor_bytes"])


if __name__ == "__main__":
    unittest.main()
