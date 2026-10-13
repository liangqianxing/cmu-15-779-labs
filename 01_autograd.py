import torch

w = torch.tensor(0.0, requires_grad=True)
learn_rate = 0.01

for step in range(10):
    # 每次求导前，清除上一轮的梯度
    w.grad = None

    # 前向计算
    loss = (w - 3) ** 2

    # 反向传播
    loss.backward()

    # 更新参数，并观察更新后的损失
    with torch.no_grad():
        w -= learn_rate * w.grad
        new_loss = (w - 3) ** 2

    print(
        f"第 {step + 1:2d} 次："
        f"w={w.item():.4f}, "
        f"loss={new_loss.item():.4f}"
    )