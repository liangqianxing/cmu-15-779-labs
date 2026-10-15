import torch
from torch import nn

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

model=MyModel()
parameters = torch.load("square_model.pth",weights_only=True)
model.load_state_dict(parameters)
model.eval()

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