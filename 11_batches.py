import torch
from torch.utils.data import DataLoader, TensorDataset

x=torch.linspace(-1., 1., 21).reshape(-1, 1)

target=x**2

dataset=TensorDataset(x,target)

loader=DataLoader(dataset,batch_size=5,shuffle=False)

for batch_index, (batch_x, batch_target) in enumerate(loader, start=1):
    print(
        f"第{batch_index}批："
        f"输入形状={batch_x.shape}，"
        f"目标形状={batch_target.shape}"
    )