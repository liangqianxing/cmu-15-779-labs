import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kv_cache", ROOT / "29_kv_cache.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class KVCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_prefix_outputs_counts_and_cache_size(self):
        for prompt, steps in ((1, 0), (1, 5), (3, 0), (3, 5), (17, 2)):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(prompt=prompt, steps=steps, dtype=dtype):
                    case = lab.run_case(prompt, steps, 8, 4, 3, dtype)
                    lengths = list(range(prompt, prompt + steps + 1))
                    self.assertTrue(case["all_outputs_close"])
                    self.assertEqual(case["projection_token_rows"], {"recompute": sum(lengths), "cached": prompt + steps})
                    self.assertEqual(case["attention_score_pairs"],
                                     {"recompute": sum(n * n for n in lengths),
                                      "cached": prompt * prompt + sum(lengths[1:])})
                    self.assertEqual(case["final_kv_bytes"], (prompt + steps) * 7 * torch.empty((), dtype=dtype).element_size())

    def test_single_query_attends_all_cached_keys(self):
        q = torch.zeros(1, 1, dtype=torch.float64)
        k = torch.zeros(3, 1, dtype=torch.float64)
        v = torch.tensor([[1.0], [2.0], [6.0]], dtype=torch.float64)
        actual = lab.latest_query_attention(q, k, v)
        torch.testing.assert_close(actual, torch.tensor([[3.0]], dtype=torch.float64))

    def test_past_outputs_do_not_use_future_tokens(self):
        generator = torch.Generator().manual_seed(2901)
        x = torch.randn(7, 4, generator=generator, dtype=torch.float64)
        weights = tuple(torch.randn(4, 3, generator=generator, dtype=torch.float64) for _ in range(3))
        before = lab.causal_attention(*lab.project(x, weights))
        changed = x.clone()
        changed[4:] += 100
        after = lab.causal_attention(*lab.project(changed, weights))
        torch.testing.assert_close(before[:4], after[:4], rtol=0, atol=0)

    def test_invalid_parameters(self):
        for params in ({"prompt": 0}, {"steps": -1}, {"steps": 129}, {"head_dim": 0},
                       {"value_dim": 129}, {"model_dim": 1.5}, {"dtype": torch.int64}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                lab.run_case(**params)


if __name__ == "__main__":
    unittest.main()
