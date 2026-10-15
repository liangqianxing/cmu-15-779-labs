import torch

x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])


weights=torch.tensor([[2.,3.]])
bias=torch.tensor([1.])

prediction=x@weights.T+bias

print("输入形状：", x.shape)
print("转置后的权重形状：", weights.T.shape)
print("预测值：", prediction)
print("输出形状：", prediction.shape)
