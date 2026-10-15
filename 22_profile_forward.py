import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity


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


torch.manual_seed(0)

model = MyModel()
model.eval()

# 一批 1000 个样本，每个样本有 1 个特征
x = torch.linspace(-1.0, 1.0, 1000).reshape(-1, 1)

with torch.no_grad():
    # 先预热；这些调用不计入下面的统计
    for _ in range(10):
        model(x)

    # 记录一次前向计算中的 CPU 操作
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        prediction = model(x)

print("输入形状：", x.shape)
print("输出形状：", prediction.shape)

print(
    prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    )
)
