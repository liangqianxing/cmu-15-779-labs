import importlib.util
from pathlib import Path
import unittest

import torch


def _load_script(name: str):
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PiecewiseReLUTests(unittest.TestCase):
    def test_knots_and_analytical_slopes_match_autograd(self):
        mod = _load_script("20_piecewise_linear_slopes")
        model = mod.PiecewiseReLUNet(
            weights1=[1.0, -1.0, 2.0, -2.0],
            biases1=[0.0, 0.0, -1.0, -1.0],
            weights2=[1.0, 1.0, 0.5, 0.5],
            bias2=0.0,
        )
        knots = mod.compute_knots(model)
        self.assertEqual(knots, [-0.5, 0.0, 0.5])

        for pt in (-0.8, -0.25, 0.25, 0.8):
            x = torch.tensor([[pt]], dtype=torch.float32, requires_grad=True)
            y = model(x)
            y.backward()
            self.assertAlmostEqual(
                mod.analytical_slope(model, pt),
                float(x.grad.item()),
                places=5,
            )

    def test_interpolation_vs_extrapolation_error(self):
        mod = _load_script("21_generalization_split")
        metrics = mod.train_and_evaluate(seed=0, steps=1500, lr=0.05)
        self.assertLess(metrics["train_mse"], 1e-3)
        self.assertLess(metrics["interp_mse"], 5e-3)
        self.assertGreater(metrics["extrap_mse"], metrics["interp_mse"])


if __name__ == "__main__":
    unittest.main()
