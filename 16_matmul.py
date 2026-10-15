import torch

x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])


weights=torch.tensor([[2.,3.]])
bias=torch.tensor([1.])

prediction=x@weights.T+bias

print("输入形状：", x.shape)
print("转置后的权重形状：", weights.T.shape)
print("预测值：", prediction)
print("输出形状：", prediction.shape)


model = torch.nn.Linear(2, 1)

with torch.no_grad():
    # 将上面手写的权重和偏置复制到模型中
    model.weight.copy_(weights)
    model.bias.copy_(bias)

    model_prediction = model(x)

print("矩阵乘法的结果：", prediction)
print("Linear 的结果：", model_prediction)

print(
    "结果近似一致：",
    torch.allclose(prediction, model_prediction)
)