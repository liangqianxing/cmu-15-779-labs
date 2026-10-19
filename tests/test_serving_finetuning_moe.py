import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(stem):
    spec = importlib.util.spec_from_file_location(stem, ROOT / (stem+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache_lab = load("33_paged_kv_cache")
spec_lab = load("34_speculative_sampling")
lora_lab = load("35_lora_quantization")
moe_lab = load("36_moe_dispatch")
torch.set_num_threads(1)


class PagedCacheTests(unittest.TestCase):
    def test_random_fork_append_release_against_independent_storage(self):
        rng = np.random.default_rng(3300)
        gen = torch.Generator().manual_seed(3300)
        for block_size in (1, 2, 4, 7):
            cache = cache_lab.PagedKVCache(100, block_size, 3, 2)
            reference = {}
            cache.start("0")
            reference["0"] = ([], [])
            next_id = 1
            for _ in range(160):
                name = str(rng.choice(list(reference)))
                choice = rng.random()
                if choice < 0.65:
                    key = torch.randn(3, generator=gen, dtype=torch.float64)
                    value = torch.randn(2, generator=gen, dtype=torch.float64)
                    cache.append(name, key, value)
                    reference[name][0].append(key)
                    reference[name][1].append(value)
                elif choice < 0.85 and len(reference) < 10:
                    child = str(next_id)
                    next_id += 1
                    prefix = int(rng.integers(len(reference[name][0])+1))
                    cache.fork(name, child, prefix)
                    reference[child] = tuple(items[:prefix].copy() for items in reference[name])
                elif len(reference) > 1:
                    cache.release(name)
                    del reference[name]
                cache.check_invariants()
                for sequence, items in reference.items():
                    for actual, expected, width in zip(cache.gather(sequence), items, (3, 2)):
                        expected = torch.stack(expected) if expected else torch.empty(0, width, dtype=torch.float64)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for name in list(reference):
                cache.release(name)
            cache.check_invariants()
            self.assertEqual(cache.stats()["active_pages"], 0)

    def test_oom_is_atomic_for_new_page_and_copy_on_write(self):
        for block_size in (1, 2):
            cache = cache_lab.PagedKVCache(1, block_size, 2, 2)
            cache.start("a")
            key = torch.tensor([1., 2.], dtype=torch.float64)
            cache.append("a", key, key)
            cache.fork("a", "b")
            before = cache.gather("b")
            with self.assertRaises(MemoryError):
                cache.append("b", key+10, key+10)
            cache.check_invariants()
            self.assertEqual(cache.lengths, {"a": 1, "b": 1})
            for old, actual in zip(before, cache.gather("b")):
                torch.testing.assert_close(old, actual, atol=0, rtol=0)

    def test_attention_and_parent_isolation(self):
        report = cache_lab.run()
        self.assertLess(report["max_abs_attention_error"], 1e-12)
        self.assertEqual(report["shared_prefix"]["active_pages"], 2)
        self.assertEqual(report["after_divergence"]["active_pages"], 4)
        self.assertEqual(report["after_release"]["active_pages"], 0)


class SpeculativeTests(unittest.TestCase):
    def test_exact_mass_identity_including_disjoint_support(self):
        rng = np.random.default_rng(3400)
        fixtures = [(np.array([1., 0, 0]), np.array([0., 1, 0])),
                    (np.array([0., .4, .6]), np.array([.4, .6, 0])),
                    (np.array([.3, .7]), np.array([.3, .7]))]
        fixtures += [(rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))) for _ in range(100)]
        for p, q in fixtures:
            np.testing.assert_allclose(spec_lab.corrected_marginal(p, q), p, rtol=0, atol=1e-14)

    def test_always_accept_and_bonus_token(self):
        model = lambda context: np.array([0., 1., 0.])
        tokens, record = spec_lab.speculative_round((), model, model, 4, np.random.default_rng(1))
        self.assertEqual(tokens, [1]*5)
        self.assertEqual(record["accepted"], 4)
        self.assertFalse(record["rejected"])

    def test_disjoint_support_rejects_then_resamples_from_target(self):
        p = lambda context: [1., 0.]
        q = lambda context: [0., 1.]
        tokens, record = spec_lab.speculative_round((), p, q, 3, np.random.default_rng(2))
        self.assertEqual(tokens, [0])
        self.assertTrue(record["rejected"])
        self.assertEqual(record["accepted"], 0)

    def test_invalid_distributions(self):
        for p in ([0., 0.], [-1., 2.], [float("nan"), 1], [0.2, 0.2], [[1.]]):
            with self.assertRaises(ValueError):
                spec_lab.distribution(p)


class LoRATests(unittest.TestCase):
    def test_adapter_gradients_match_merged_expression(self):
        torch.manual_seed(3500)
        module = lora_lab.LoRALinear(torch.randn(5, 7, dtype=torch.float64), 2, 4)
        with torch.no_grad():
            module.b.normal_()
        x = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
        y = module(x).square().sum()
        actual = torch.autograd.grad(y, (x, module.a, module.b))
        expected = torch.autograd.grad((x @ module.merged_weight().T).square().sum(), (x, module.a, module.b))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-12)

    def test_initial_zero_b_and_frozen_base(self):
        record = lora_lab.run()
        self.assertEqual(record["initial_forward_max_abs_error"], 0)
        self.assertEqual(record["initial_gradient_norms"]["a"], 0)
        self.assertGreater(record["initial_gradient_norms"]["b"], 0)
        self.assertTrue(record["base_weight_unchanged"])

    def test_packed_quantization_error_bound_with_partial_blocks(self):
        generator = torch.Generator().manual_seed(3501)
        for size in (1, 7, 64, 65, 191):
            for block in (1, 3, 64):
                weight = torch.randn(size, generator=generator, dtype=torch.float64)
                record = lora_lab.quantize_uniform4(weight, block)
                decoded = lora_lab.dequantize_uniform4(record)
                bound = record["scales"].repeat_interleave(block)[:size]/15 + 1e-6
                self.assertTrue(torch.all((decoded-weight).abs() <= bound))
                self.assertEqual(record["packed"].numel(), (len(record["scales"])*block+1)//2)
        zero = torch.zeros(9, dtype=torch.float64)
        torch.testing.assert_close(lora_lab.dequantize_uniform4(lora_lab.quantize_uniform4(zero, 3)), zero, atol=0, rtol=0)


class MoETests(unittest.TestCase):
    def test_sparse_outputs_and_gradients_match_dense_reference(self):
        torch.manual_seed(3600)
        x = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
        up = torch.randn(3, 4, 6, dtype=torch.float64, requires_grad=True)
        down = torch.randn(3, 6, 4, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
        for k in (1, 2, 3):
            actual, record = moe_lab.moe_dispatch(x, logits, up, down, k)
            expected = moe_lab.dense_reference(x, logits, up, down, k)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            self.assertEqual(sum(record["processed_assignments_per_expert"]), 7*k)
            a_grad = torch.autograd.grad(actual.square().sum(), (x, up, down, logits), retain_graph=True)
            b_grad = torch.autograd.grad(expected.square().sum(), (x, up, down, logits), retain_graph=True)
            for a, b in zip(a_grad, b_grad):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-12)

    def test_capacity_drop_is_visible(self):
        record = moe_lab.run()
        self.assertEqual(record["skewed_routing"]["requested_assignments_per_expert"], [13, 13, 0, 0])
        self.assertEqual(record["capacity_limited"]["dropped_assignments"], 12)
        self.assertGreater(record["capacity_output_max_abs_difference"], 0)


if __name__ == "__main__":
    unittest.main()
