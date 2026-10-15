import torch
from torch import nn
from time import perf_counter
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
x = torch.linspace(-1.0, 1.0, 1).reshape(-1, 1)


def predict_one_by_one():
    outputs = []
    for i in range(x.shape[0]):
        # 保留 [1, 1] 的输入形状
        outputs.append(model(x[i:i+1]))
    return torch.cat(outputs, dim=0)


def predict_batch():
    return model(x)


def measure_ms(function):
    start = perf_counter()
    for _ in range(5):
        function()
    # 计算每次执行的平均毫秒数
    return (perf_counter() - start) / 5 * 1000


with torch.no_grad():
    a = predict_one_by_one()
    b = predict_batch()
    print("输出近似一致：", torch.allclose(a, b, atol=1e-6))

    # 先预热，减少首次执行的影响
    for _ in range(3):
        predict_one_by_one()
        predict_batch()

    one_ms = measure_ms(predict_one_by_one)
    batch_ms = measure_ms(predict_batch)

print(f"逐个预测：{one_ms:.4f} 毫秒")
print(f"整批预测：{batch_ms:.4f} 毫秒")
print(f"加速比：{one_ms / batch_ms:.2f} 倍")