import torch

x=torch.tensor([[1.0],[2.0],[3.0],[4.0]])
target=torch.tensor([[3.0],[5.0],[7.0],[9.0]])

model=torch.nn.Linear(1,1)

optimizer=torch.optim.SGD(model.parameters(),lr=0.05)

for step in range(500):
    optimizer.zero_grad()

    prediction=model(x)

    loss=((prediction-target)**2).mean()

    loss.backward()
    optimizer.step()

    if (step+1)%10==0:
        with torch.no_grad():
            new_loss=((model(x)-target)**2).mean()
        print(
            f"第{step+1:3d}次："
            f"w={model.weight.item():.4f},"
            f"b={model.bias.item():.4f},"
            f"loss={new_loss.item():.4f}"
        )

with torch.no_grad():
    test_x=torch.tensor([[5.0],[6.0]])
    print("新输入：",test_x)
    print("预测值：",model(test_x))