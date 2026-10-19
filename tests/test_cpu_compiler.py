import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_compiler", ROOT / "30_cpu_compiler.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class CompilerTests(unittest.TestCase):
    def test_batchnorm_fold_with_and_without_affine_and_linear_bias(self):
        torch.manual_seed(3003)
        for affine in (False, True):
            for bias in (False, True):
                linear = nn.Linear(7, 5, bias=bias, dtype=torch.float64).eval()
                bn = nn.BatchNorm1d(5, affine=affine, dtype=torch.float64).eval()
                bn.running_mean.copy_(torch.randn(5, dtype=torch.float64))
                bn.running_var.copy_(torch.rand(5, dtype=torch.float64) + 0.3)
                if affine:
                    with torch.no_grad():
                        bn.weight.copy_(torch.randn(5, dtype=torch.float64))
                        bn.bias.copy_(torch.randn(5, dtype=torch.float64))
                folded = lab.fold_linear_batchnorm(linear, bn)
                for batch in (1, 3, 11):
                    x = torch.randn(batch, 7, dtype=torch.float64)
                    torch.testing.assert_close(folded(x), bn(linear(x)), atol=1e-12, rtol=1e-12)

    def test_folding_rejects_dynamic_statistics_or_wrong_shape(self):
        linear = nn.Linear(4, 3)
        for bn in (nn.BatchNorm1d(3), nn.BatchNorm1d(3, track_running_stats=False).eval(),
                   nn.BatchNorm1d(4).eval()):
            with self.assertRaises(ValueError):
                lab.fold_linear_batchnorm(linear, bn)

    def test_blocked_schedules_partial_empty_and_noncontiguous(self):
        generator = torch.Generator().manual_seed(3004)
        for m, k, n in ((0, 3, 5), (3, 0, 5), (3, 5, 0), (7, 5, 9), (1, 1, 1)):
            a = torch.randn(m, k * 2, generator=generator, dtype=torch.float64)[:, ::2]
            b = torch.randn(k, n * 2, generator=generator, dtype=torch.float64)[:, ::2]
            for tile in (1, 3, 16):
                torch.testing.assert_close(lab.blocked_matmul(a, b, tile), a @ b, atol=1e-12, rtol=1e-12)

    def test_row_denominator_rewrite_handles_zero_rows_and_signed_gamma(self):
        generator = torch.Generator().manual_seed(3005)
        x = torch.randn(7, 5, generator=generator, dtype=torch.float64)
        x[0].zero_()
        weight = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        gamma = torch.tensor([-1.0, 0, 1, 2, -2], dtype=torch.float64)
        for eps in (1e-6, 0.2):
            torch.testing.assert_close(lab.rmsnorm_linear(x, weight, gamma, eps),
                                       lab.moved_rmsnorm_linear(x, gamma[:, None] * weight, eps),
                                       atol=1e-12, rtol=1e-12)

    def test_invalid_matmul_and_nonlinear_rewrite(self):
        with self.assertRaises(ValueError):
            lab.blocked_matmul(torch.ones(2, 3), torch.ones(4, 2), 2)
        with self.assertRaises(ValueError):
            lab.blocked_matmul(torch.ones(2, 3), torch.ones(3, 2), 0)
        x = torch.tensor([[-1.0, 1.0]])
        w = torch.tensor([[-1.0], [1.0]])
        self.assertFalse(torch.equal(torch.relu(x @ w), torch.relu(x) @ w))


if __name__ == "__main__":
    unittest.main()
