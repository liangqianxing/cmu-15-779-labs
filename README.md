# CMU 15-779 代码实验

CMU 15-779 学习记录中的配套代码。当前练习在 CPU 上运行，包含自动求导、线性回归、小型 ReLU 网络、DataLoader、参数保存、批量推理、绝对值网络与算子性能分析。

## 环境

- Python 3.11
- PyTorch 2.14.1
- NumPy 2.4.6

创建环境并安装依赖：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 学习记录与脚本

| 记录 | 脚本 | 内容 |
| --- | --- | --- |
| [01：自动求导与线性回归](https://liangqianxing.github.io/posts/cmu-15-779-pytorch-autograd-linear-regression/) | `01` 至 `03` | 手动梯度下降、SGD、无偏置线性回归 |
| [02：nn.Linear、张量形状与 ReLU](https://liangqianxing.github.io/posts/cmu-15-779-pytorch-linear-module-relu/) | `04` 至 `10` | 带偏置回归、模型封装、非线性拟合 |
| [03：DataLoader、模型保存与批量推理](https://liangqianxing.github.io/posts/cmu-15-779-pytorch-dataloader-checkpoints-batching/) | `11` 至 `16` | 分批训练、参数恢复、推理计时和矩阵乘法 |
| [04：多输出线性层、ReLU 与绝对值网络](https://liangqianxing.github.io/posts/cmu-15-779-pytorch-linear-relu-absolute-value/) | `16` 至 `19` | Linear 参数对照、多输出、逐层计算和绝对值拟合 |
| [05：Profiler、算子调用与批量推理](https://liangqianxing.github.io/posts/cmu-15-779-profiler-operator-batching/) | `22` 至 `24` | 前向计算分析、调用次数与固定工作量的 batch size 扫描 |

各脚本是逐步练习的独立版本，可以按编号运行。

```bash
.venv/bin/python 01_autograd.py
.venv/bin/python 02_SGD.py
.venv/bin/python 03_linear.py
.venv/bin/python 04_linear_bias.py
.venv/bin/python 05_nn_linear.py
.venv/bin/python 06_two_inputs.py
.venv/bin/python 07_custom_model.py
.venv/bin/python 08_square.py
.venv/bin/python 09_square_relu.py
.venv/bin/python 10_more_data.py
.venv/bin/python 11_batches.py
.venv/bin/python 12_mini_batch.py
.venv/bin/python 13_save_load.py
.venv/bin/python 14_predict.py
.venv/bin/python 15_batch_benchmark.py
.venv/bin/python 16_matmul.py
.venv/bin/python 17_multi_output.py
.venv/bin/python 18_relu_steps.py
.venv/bin/python 19_learn_abs.py
.venv/bin/python 22_profile_forward.py
.venv/bin/python 23_profile_batching.py
.venv/bin/python 24_batch_size_sweep.py --output results/24_batch_size_sweep-rerun.json
```

## 复现说明

- `01_autograd.py` 保留最后一次使用的学习率 `0.01`；改为 `0.1` 可以复现第一篇中的对照。
- `05` 至 `08` 没有设置随机种子，参数和预测的末位数字可能随运行变化。`09` 至 `13` 在创建模型前设置了 `torch.manual_seed(0)`。
- `10_more_data.py` 和 `12_mini_batch.py` 使用 25 个训练点。代码中名为 `test_x` 的 `[-0.75, 0.25, 0.75]` 实际包含在这些训练点中，打印的误差不能作为独立测试误差。
- `13_save_load.py` 使用 21 个训练点，检查点是未参与训练的 `[-0.73, 0.23, 0.77]`。
- 仓库包含本次练习保存的 `square_model.pth`，只有 25 个模型参数。`13_save_load.py` 会重新训练并覆盖当前目录中的这个文件；`14`、`15` 可以直接读取附带的参数文件。
- 请在仓库目录运行涉及参数保存和加载的脚本，其路径按当前工作目录解析。
- `15_batch_benchmark.py` 保留最后一次使用的 1 个样本。将 `torch.linspace(-1.0, 1.0, 1)` 的最后一个参数改为 `1000`，可比较 1000 个样本的逐个预测与整批预测。计时包含 Python 循环、切片及拼接开销，倍数随运行环境波动。
- `16` 至 `18` 手动给定参数，不训练模型，也不依赖参数文件。`16_matmul.py` 的基础矩阵乘法见记录 03；追加的 `nn.Linear` 参数对照见记录 04。
- `18_relu_steps.py` 保留使用 ReLU 的版本，手动构造绝对值函数。记录 04 中跳过 ReLU 得到全零的结果，是针对这组固定参数额外执行的对照。
- `19_learn_abs.py` 使用随机种子 `0`、21 个训练点和 SGD（学习率 `0.05`），训练 3000 步。训练集均方误差约为 `0.00005678`；检查输入 `[-0.8, 0, 0.8]` 同样包含在训练集中，不能作为独立测试集评估。

- `22` 至 `24` 是对照 ML Systems 基础的个人配套实验，不是 CMU 官方作业。`24` 固定一个 CPU 线程，将完整处理全部输入样本作为计时单位；原始测量保存在 `results/24_batch_size_sweep.json`，复跑时用 `--output` 另存。

[课程学习专题](https://liangqianxing.github.io/learning/cmu-15-779/)提供学习资料与上述记录的入口。
