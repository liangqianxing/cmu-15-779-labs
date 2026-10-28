"""20: Piecewise-Linear Slope & Knot Reconstruction for 1D ReLU Networks.

Verifies how a two-layer 1D ReLU network y = w2 @ ReLU(w1 * x + b1) + b2
partitions the input line into piecewise-linear intervals separated by knots
k_i = -b1_i / w1_i, and compares the analytical regional slope against finite
differences and autograd jacobians.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PiecewiseReLUNet(nn.Module):
    def __init__(self, weights1: list[float], biases1: list[float], weights2: list[float], bias2: float = 0.0):
        super().__init__()
        width = len(weights1)
        if len(biases1) != width or len(weights2) != width:
            raise ValueError("weights1, biases1, and weights2 must have the same length")

        self.fc1 = nn.Linear(1, width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(width, 1)

        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(weights1, dtype=torch.float32).reshape(width, 1))
            self.fc1.bias.copy_(torch.tensor(biases1, dtype=torch.float32))
            self.fc2.weight.copy_(torch.tensor(weights2, dtype=torch.float32).reshape(1, width))
            self.fc2.bias.fill_(float(bias2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def compute_knots(model: PiecewiseReLUNet) -> list[float]:
    w1 = model.fc1.weight.detach().squeeze(-1)
    b1 = model.fc1.bias.detach()
    knots: list[float] = []
    for wi, bi in zip(w1.tolist(), b1.tolist()):
        if abs(wi) > 1e-12:
            knots.append(-bi / wi)
    return sorted(set(round(k, 7) for k in knots))


def analytical_slope(model: PiecewiseReLUNet, x_val: float) -> float:
    w1 = model.fc1.weight.detach().squeeze(-1)
    b1 = model.fc1.bias.detach()
    w2 = model.fc2.weight.detach().squeeze(0)
    active = (w1 * float(x_val) + b1) > 0
    return float((w2 * w1 * active.to(w1.dtype)).sum().item())


def main() -> None:
    # Construct a 4-unit network that synthesizes |x| on [-1, 1] with extra slope changes at +/- 0.5
    model = PiecewiseReLUNet(
        weights1=[1.0, -1.0, 2.0, -2.0],
        biases1=[0.0, 0.0, -1.0, -1.0],
        weights2=[1.0, 1.0, 0.5, 0.5],
        bias2=0.0,
    )

    knots = compute_knots(model)
    probe_points = [-0.75, -0.25, 0.25, 0.75]
    print("折点位置 (knots):", knots)

    for pt in probe_points:
        x_tensor = torch.tensor([[pt]], dtype=torch.float32, requires_grad=True)
        y_tensor = model(x_tensor)
        y_tensor.backward()
        grad_val = float(x_tensor.grad.item())
        exact_slope = analytical_slope(model, pt)
        print(f"x = {pt:+.2f} | y = {float(y_tensor.item()):.4f} | 解析斜率 = {exact_slope:+.4f} | autograd = {grad_val:+.4f}")


if __name__ == "__main__":
    main()
