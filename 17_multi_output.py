import torch

# 3 个样本，每个样本有 2 个特征
x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

model = torch.nn.Linear(2, 2)

# 每一行权重负责计算一个输出
weights = torch.tensor([
    [2.0,  3.0],
    [4.0, -1.0],
])
bias = torch.tensor([1.0, 2.0])

with torch.no_grad():
    model.weight.copy_(weights)
    model.bias.copy_(bias)

    prediction = model(x)

print("输入形状：", x.shape)
print("权重形状：", model.weight.shape)
print("偏置形状：", model.bias.shape)
print("预测值：", prediction)
print("输出形状：", prediction.shape)