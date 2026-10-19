import importlib.util
from pathlib import Path
import unittest

import torch

SPEC = importlib.util.spec_from_file_location("parallelism", Path(__file__).resolve().parents[1] / "31_cpu_parallelism.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class ParallelismTests(unittest.TestCase):
    def test_weighted_gradients_for_every_shard_count(self):
        model, x, y = lab.make_problem()
        reference = lab.reference_gradients(model.state_dict(), x, y)
        for ranks in range(1, len(x) + 1):
            actual, sizes = lab.aggregate_gradients(model.state_dict(), x, y, ranks)
            self.assertEqual(sum(sizes), len(x))
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_equal_rank_weighting_fails_for_unequal_sample_counts(self):
        model, x, y = lab.make_problem()
        reference = lab.reference_gradients(model.state_dict(), x, y)
        actual, sizes = lab.aggregate_gradients(model.state_dict(), x, y, 2, weighted=False)
        self.assertEqual(sizes, [4, 3])
        self.assertGreater(max((a - b).abs().max().item() for a, b in zip(actual, reference)), 0.01)

    def test_tensor_splits_with_empty_shards_and_noncontiguous_inputs(self):
        generator = torch.Generator().manual_seed(3100)
        for m, k, n in ((2, 5, 3), (0, 5, 3), (3, 0, 5), (2, 3, 0)):
            a = torch.randn(m, 2*k, generator=generator, dtype=torch.float64)[:, ::2]
            b = torch.randn(k, 2*n, generator=generator, dtype=torch.float64)[:, ::2]
            for parts in (1, 2, 7):
                for actual in lab.tensor_partitions(a, b, parts):
                    torch.testing.assert_close(actual, a @ b, atol=1e-12, rtol=1e-12)

    def test_zero_model_padding_and_limits(self):
        self.assertEqual(lab.zero_storage_model(1_000_000, 4)["per_rank_bytes"],
                         dict(replicated=16_000_000, zero1=7_000_000, zero2=5_500_000, zero3=4_000_000))
        self.assertEqual(lab.zero_storage_model(5, 1)["per_rank_bytes"], dict.fromkeys(("replicated", "zero1", "zero2", "zero3"), 80))
        self.assertEqual(lab.zero_storage_model(5, 2)["per_rank_bytes"]["zero3"], 48)
        for parameters, ranks in ((0, 2), (5, 0), (2.5, 2)):
            with self.assertRaises(ValueError):
                lab.zero_storage_model(parameters, ranks)


if __name__ == "__main__":
    unittest.main()
