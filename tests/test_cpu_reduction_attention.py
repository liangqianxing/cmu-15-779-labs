import importlib.util
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_lab(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reduction = load_lab("tree_reduction", "27_tree_reduction.py")
attention = load_lab("attention_online", "28_attention_online_softmax.py")


class ReductionTests(unittest.TestCase):
    def test_lecture_example(self):
        result = reduction.tree_reduce([3, 1, 7, 0, 4, 1, 6, 3], keep_levels=True)
        self.assertEqual(result["levels"], [[3, 1, 7, 0, 4, 1, 6, 3], [4, 7, 5, 9], [11, 14], [25]])
        self.assertEqual(result["additions"], 7)
        self.assertEqual(result["dependency_depth"], 3)

    def test_many_lengths_and_block_sizes(self):
        for n in range(258):
            for size in (2, 3, 4, 32):
                with self.subTest(n=n, block_size=size):
                    case = reduction.run_case(n, size)
                    self.assertTrue(case["fixture_outputs_exact"])
                    self.assertEqual(case["staged"]["additions"], max(0, n - 1))

    def test_staged_partials(self):
        case = reduction.run_case(257, 32)
        self.assertEqual([r["output_partials"] for r in case["staged"]["rounds"]], [9, 1])

    def test_random_floats_with_tolerance(self):
        rng = random.Random(27)
        for n in (1, 3, 17, 257):
            values = [rng.uniform(-1, 1) for _ in range(n)]
            expected = math.fsum(values)
            for actual in (reduction.serial_sum(values), reduction.tree_reduce(values)["sum"],
                           reduction.staged_reduce(values, 7)["sum"]):
                self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))

    def test_cancellation_exposes_order(self):
        values = [1e16, 1.0, -1e16, 1.0]
        self.assertEqual(reduction.serial_sum(values), 1.0)
        self.assertEqual(reduction.tree_reduce(values)["sum"], 0.0)
        self.assertEqual(math.fsum(values), 2.0)

    def test_invalid_inputs(self):
        for n, size in ((-1, 4), (65537, 4), (3, 0), (3, 1), (3, 2.5)):
            with self.subTest(n=n, size=size), self.assertRaises(ValueError):
                reduction.run_case(n, size)


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_shapes_masks_dtypes_and_tiles(self):
        for n, d, dv in ((1, 1, 1), (3, 4, 2), (7, 3, 5), (17, 8, 4)):
            for qb, kb in ((1, 1), (3, 4), (8, 5), (32, 32)):
                for causal in (False, True):
                    for dtype in (torch.float32, torch.float64):
                        with self.subTest(n=n, qb=qb, kb=kb, causal=causal, dtype=dtype):
                            case = attention.run_case(n, d, dv, qb, kb, causal, dtype)
                            counts = case["score_storage"]
                            self.assertEqual(counts["largest_score_tile_elements"], min(n, qb) * min(n, kb))
                            self.assertTrue(case["outputs_close"])

    def test_zero_scores_give_uniform_or_prefix_mean(self):
        q = torch.zeros(7, 3, dtype=torch.float64)
        v = torch.arange(14, dtype=torch.float64).reshape(7, 2)
        for causal in (False, True):
            result, _ = attention.tiled_attention(q, q, v, 3, 2, causal)
            expected = torch.stack([v[:i + 1].mean(0) if causal else v.mean(0) for i in range(7)])
            torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_future_keys_and_values_do_not_affect_prefix(self):
        q, k, v = attention.make_inputs(7, 3, 5, torch.float64)
        before, _ = attention.tiled_attention(q, k, v, 3, 2, True)
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[4:] += 100
        changed_v[4:] -= 100
        after, _ = attention.tiled_attention(q, changed_k, changed_v, 3, 2, True)
        torch.testing.assert_close(before[:4], after[:4], rtol=0, atol=0)

    def test_changing_running_max_rescales_previous_blocks(self):
        q = torch.ones(4, 1, dtype=torch.float64)
        k = torch.tensor([[0.0], [1.0], [10.0], [11.0]], dtype=torch.float64)
        v = torch.tensor([[10.0], [-10.0], [2.0], [3.0]], dtype=torch.float64)
        actual, _ = attention.tiled_attention(q, k, v, 2, 1)
        expected, _ = attention.dense_attention(q, k, v)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_large_finite_scores_are_stable(self):
        q, k, v = attention.make_inputs(7, 4, 2, torch.float64)
        q, k = q * 1000, k * 1000
        actual, _ = attention.tiled_attention(q, k, v, 3, 2, True)
        expected = attention.sdpa_math_reference(q, k, v, True)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_noncontiguous_inputs_and_forward_only(self):
        q, k, v = attention.make_inputs(9, 4, 6, torch.float64)
        q, k, v = q[:, ::2], k[:, ::2], v[:, ::2]
        q.requires_grad_(True)
        actual, _ = attention.tiled_attention(q, k, v, 4, 5)
        expected = attention.sdpa_math_reference(q, k, v)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_invalid_shapes_types_and_values(self):
        q, k, v = attention.make_inputs(3, 4, 2, torch.float64)
        cases = [(q[None], k, v), (q[:0], k[:0], v[:0]), (q, k[:2], v),
                 (q, k, v.float()), (q.long(), k.long(), v.long()),
                 (torch.full_like(q, float("nan")), k, v)]
        for inputs in cases:
            with self.assertRaises(ValueError):
                attention.tiled_attention(*inputs)
        for qb, kb in ((0, 1), (1, -1), (1.5, 2)):
            with self.assertRaises(ValueError):
                attention.tiled_attention(q, k, v, qb, kb)

    def test_invalid_cli_does_not_write_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            for script, arguments in (("27_tree_reduction.py", ["--block-size", "1"]),
                                      ("28_attention_online_softmax.py", ["--n", "0"]),
                                      ("28_attention_online_softmax.py", ["--key-block", "0"])):
                result = subprocess.run([sys.executable, str(ROOT / script), *arguments,
                                         "--output", str(output)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
