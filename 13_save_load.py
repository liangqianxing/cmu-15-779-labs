import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


x = torch.linspace(-1.,1.,21).reshape(-1,1)
target = x ** 2

dataset = TensorDataset(x, target)
loader = DataLoader(dataset, batch_size=5, shuffle=False)

model=MyModel()

optimizer=torch.optim.SGD(model.parameters(),lr=0.05)

for epoch in range(600):
    for batch_x, batch_target in loader:
        optimizer.zero_grad()

        prediction=model(batch_x)

        loss=((prediction-batch_target)**2).mean()

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            new_loss = ((model(x) - target) ** 2).mean()

        print(f"第{epoch + 1:4d}次：loss={new_loss.item():.8f}")

with torch.no_grad():
    # 这些输入没有出现在训练数据中
    test_x = torch.tensor([
        [-0.73],
        [0.23],
        [0.77],
    ])

    test_target = test_x ** 2
    test_prediction = model(test_x)
    test_loss = ((test_prediction - test_target) ** 2).mean()

    print("新输入：", test_x)
    print("目标值：", test_target)
    print("预测值：", test_prediction)
    print("测试损失：", test_loss.item())




torch.save(model.state_dict(), "square_model.pth")
loaded_model = MyModel()
parameters = torch.load("square_model.pth")
loaded_model.load_state_dict(parameters)
loaded_model.eval()

with torch.no_grad():
    print("比较使用的输入：", test_x)
    print("原模型预测：", model(test_x))
    print("加载后预测：", loaded_model(test_x))