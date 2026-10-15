from statistics import median
from time import perf_counter

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


def predict_one_by_one():
    outputs = []
    for i in range(x.shape[0]):
        # 保留 [1, 1] 的形状，每次传入一个样本。
        outputs.append(model(x[i:i + 1]))
    return torch.cat(outputs, dim=0)


def predict_batch():
    return model(x)


def measure_ms(function):
    measurements = []
    for _ in range(7):
        start = perf_counter()
        for _ in range(10):
            function()
        measurements.append((perf_counter() - start) / 10 * 1000)
    # 每组先取平均，再取七组的中位数，减小偶发波动的影响。
    return median(measurements)


def collect_counts(function):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        function()
    return {event.key: event.count for event in prof.key_averages()}


torch.manual_seed(0)
sample_count = 1000
model = MyModel()
model.eval()
x = torch.linspace(-1.0, 1.0, sample_count).reshape(-1, 1)

with torch.no_grad():
    single_output = predict_one_by_one()
    batch_output = predict_batch()
    matches = torch.allclose(single_output, batch_output, atol=1e-6)
    print("输出近似一致：", matches)
    if not matches:
        raise RuntimeError("输出不一致，请先检查预测代码。")
    print("最大绝对误差：", (single_output - batch_output).abs().max().item())

    for _ in range(10):
        predict_one_by_one()
        predict_batch()

    # 耗时在 Profiler 外测量，避免把记录开销计入速度比较。
    single_ms = measure_ms(predict_one_by_one)
    batch_ms = measure_ms(predict_batch)

    # 每次记录都完整处理同一组输入，只统计一次处理的调用次数。
    single_counts = collect_counts(predict_one_by_one)
    batch_counts = collect_counts(predict_batch)

print(f"\n样本数：{sample_count}，设备：{x.device}")
print("算子调用次数（one_by_one=逐个，batch=整批）：")
print(f"{'operator':<18} {'one_by_one':>12} {'batch':>10}")
for name in ("aten::linear", "aten::addmm", "aten::relu", "aten::slice", "aten::cat"):
    print(f"{name:<18} {single_counts.get(name, 0):>12} {batch_counts.get(name, 0):>10}")

print(f"\n完整处理 {sample_count} 个样本的耗时（Profiler 外计时）：")
print(f"逐个预测：{single_ms:.4f} 毫秒")
print(f"整批预测：{batch_ms:.4f} 毫秒")
print(f"加速比：{single_ms / batch_ms:.2f} 倍")
