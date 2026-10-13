import torch

w=torch.tensor(0.0,requires_grad=True)

optimizer=torch.optim.SGD([w],lr=0.1)

for step in range(10):
    optimizer.zero_grad()

    loss=(w-3)**2

    loss.backward()
    optimizer.step()


    with torch.no_grad():
        new_loss=(w-3)**2

    print(
        f"第{step+1:2d}次："
        f"w={w.item():.4f},"
        f"loss={new_loss.item():.4f}"
    )