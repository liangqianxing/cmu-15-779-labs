import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


x = torch.tensor([
    [-1.0],
    [-0.5],
    [0.0],
    [0.5],
    [1.0],
])

target = x ** 2

model=MyModel()

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
            f"w={model.linear.weight.tolist()},"
            f"b={model.linear.bias.item():.4f},"
            f"loss={new_loss.item():.4f}"
        )

with torch.no_grad():
    test_x=torch.tensor([[1.0],[0.0],[-1.0]])
    print("新输入：",test_x)
    print("预测值：",model(test_x))
    print("权重：", model.linear.weight)
    print("偏置：", model.linear.bias)
print("权重形状：", model.linear.weight.shape)
print("偏置形状：", model.linear.bias.shape)