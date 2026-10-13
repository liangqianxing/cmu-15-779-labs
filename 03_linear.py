import torch

w=torch.tensor(0.0,requires_grad=True)

x=torch.tensor([1.0,2.0,3.0,4.0])

target=torch.tensor([2.0,4.0,6.0,8.0])

optimizer=torch.optim.SGD([w],lr=0.01)

for step in range(100):
    optimizer.zero_grad()
    prediction=w*x
    loss=((prediction-target)**2).mean()
    loss.backward()

    optimizer.step()

    if (step+1)%10==0:
        with torch.no_grad():
            new_loss=((w*x-target)**2).mean()
        print(
            f"第{step+1:3d}次："
            f"w={w.item():.4f},"
            f"loss={new_loss.item():.4f}"
        )

with torch.no_grad():
    test_x = torch.tensor([5.0, 6.0])
    print("新输入：", test_x)
    print("预测值：", w * test_x)