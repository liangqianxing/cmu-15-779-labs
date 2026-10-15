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
    # 这些输入没有出现在训练数据中
    test_x = torch.tensor([
        [-0.75],
        [0.25],
        [0.75],
    ])

    test_target = test_x ** 2
    test_prediction = model(test_x)
    test_loss = ((test_prediction - test_target) ** 2).mean()

    print("新输入：", test_x)
    print("目标值：", test_target)
    print("预测值：", test_prediction)
    print("测试损失：", test_loss.item())