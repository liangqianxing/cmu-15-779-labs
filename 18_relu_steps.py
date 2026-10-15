import torch
from torch import nn

x = torch.tensor([
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0],
])

fc1 = nn.Linear(1, 2)
relu = nn.ReLU()
fc2 = nn.Linear(2, 1)

with torch.no_grad():
    # 第一层：把 x 变成两个数 [x, -x]
    fc1.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    fc1.bias.zero_()

    # 第二层：将两个数相加
    fc2.weight.copy_(torch.tensor([[1.0, 1.0]]))
    fc2.bias.zero_()

    hidden = fc1(x)
    activated = relu(hidden)
    prediction = fc2(activated)

print("第一层输出：\n", hidden)
print("ReLU 后：\n", activated)
print("最终预测：\n", prediction)