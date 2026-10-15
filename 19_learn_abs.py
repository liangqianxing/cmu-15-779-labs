import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)


    def forward(self, x):
        hidden = self.fc1(x)
        activated = self.relu(hidden)
        return self.fc2(activated)

torch.manual_seed(0)  # 设置随机种子以确保结果可复现

model = MyModel()

x=torch.linspace(-1.,1.,21).reshape(-1,1)
target=x.abs()

optimizer=torch.optim.SGD(model.parameters(),lr=0.05)

check_x=torch.tensor([[-0.8],[0.0],[0.8]])

with torch.no_grad():
    print("训练前：")
    print("输入：", check_x)
    print("目标值：", check_x.abs())
    print("预测值：", model(check_x))

for step in range(3000):
    optimizer.zero_grad()

    prediction=model(x)

    loss=((prediction-target)**2).mean()

    loss.backward()
    optimizer.step()

    if (step + 1) % 500 == 0:
        with torch.no_grad():
            new_loss = ((model(x) - target) ** 2).mean()

        print(f"第{step + 1:4d}次：loss={new_loss.item():.8f}")

with torch.no_grad():
    print("训练后：")
    print("输入：", check_x)
    print("目标值：", check_x.abs())
    print("预测值：", model(check_x))
