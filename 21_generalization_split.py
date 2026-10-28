"""21: Train/Interpolation vs. Extrapolation Generalization Split for 1D ReLU Networks.

Demonstrates why evaluating a model on points included in the training grid
(or inside vs. outside the training support) changes the reported MSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AbsReLUNet(nn.Module):
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def train_and_evaluate(seed: int = 0, steps: int = 2000, lr: float = 0.05) -> dict[str, float]:
    torch.manual_seed(seed)
    model = AbsReLUNet(hidden_dim=8)

    train_x = torch.linspace(-1.0, 1.0, 21).reshape(-1, 1)
    train_y = train_x.abs()

    # Unseen interpolation points strictly inside [-1, 1]
    interp_x = torch.tensor([[-0.73], [-0.37], [0.23], [0.77]], dtype=torch.float32)
    interp_y = interp_x.abs()

    # Extrapolation points outside [-1, 1]
    extrap_x = torch.tensor([[-2.5], [-1.8], [1.8], [2.5]], dtype=torch.float32)
    extrap_y = extrap_x.abs()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(train_x)
        loss = ((pred - train_y) ** 2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_mse = float(((model(train_x) - train_y) ** 2).mean().item())
        interp_mse = float(((model(interp_x) - interp_y) ** 2).mean().item())
        extrap_mse = float(((model(extrap_x) - extrap_y) ** 2).mean().item())

    return {
        "train_mse": train_mse,
        "interp_mse": interp_mse,
        "extrap_mse": extrap_mse,
    }


def main() -> None:
    metrics = train_and_evaluate()
    print(f"训练集 MSE ([-1, 1] 网格点): {metrics['train_mse']:.8f}")
    print(f"未见内插点 MSE ([-1, 1] 内部): {metrics['interp_mse']:.8f}")
    print(f"区间外推点 MSE (|x| > 1.5):   {metrics['extrap_mse']:.8f}")


if __name__ == "__main__":
    main()
