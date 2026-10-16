# CMU 15-779 代码实验

CMU 15-779 学习记录中的配套代码。已运行的 CPU 练习包含自动求导、线性回归、小型 ReLU 网络、DataLoader、参数保存、批量推理、绝对值网络、算子性能分析、CUDA 索引演示、矩阵分块、树形归约与在线 Softmax。`cuda/` 中的 CUDA C++ 源码尚未编译或执行。

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
| [06：CUDA 线程索引与边界检查](https://liangqianxing.github.io/posts/cmu-15-779-cuda-thread-indexing/) | `25` | CPU 索引演示、边界案例及未执行的 CUDA 源码 |

| [07：矩阵分块、数据复用与树形归约](https://liangqianxing.github.io/posts/cmu-15-779-matmul-tiling-tree-reduction/) | `26`、`27` | CPU 逻辑读取、依赖深度与浮点求和顺序；CUDA 矩阵乘法未执行 |
| [08：分块 Attention 与在线 Softmax](https://liangqianxing.github.io/posts/cmu-15-779-attention-online-softmax/) | `28` | 单头前向与 CPU SDPA MATH 对照，分数块大小；无 GPU 性能结论 |

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
.venv/bin/python 25_cuda_indexing.py --output results/25_cuda_indexing-rerun.json
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
- `25_cuda_indexing.py` 只在 CPU 上演示索引映射，原始结果保存在 `results/25_cuda_indexing.json` 和 `results/25_indexing_17_8.json`。它不测 GPU 性能，也不能证明 CUDA 设备内存访问安全。`cuda/25_vector_add.cu` 尚未编译执行，验证方法与当前限制见 [CUDA 运行说明](cuda/README.md)。

[课程学习专题](https://liangqianxing.github.io/learning/cmu-15-779/)提供学习资料与上述记录的入口。

## 记录 07–08 的复现

脚本编号为个人配套练习编号，不对应官方作业。原始结果来自 2026-10-16；CPU 验证不能代替 CUDA 编译、运行或性能测量。分块算法计数不是实际设备流量；Attention 分数块大小不是总内存峰值。

```bash
.venv/bin/python 26_matmul_tiling.py --output results/26_matmul_tiling-rerun.json
.venv/bin/python 26_matmul_tiling.py --m 3 --k 5 --n 7 --tile 4 --output results/26_matmul_tiling_rectangular-rerun.json
.venv/bin/python 27_tree_reduction.py --output results/27_tree_reduction-rerun.json
.venv/bin/python 28_attention_online_softmax.py --output results/28_attention_online_softmax-rerun.json
.venv/bin/python 28_attention_online_softmax.py --n 17 --d 8 --value-dim 4 --query-block 8 --key-block 5 --causal --dtype float32 --output results/28_attention_causal_float32-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

`tests/` 包含 21 项测试，覆盖矩阵乘法边界与计数、归约长度及浮点求和顺序、Attention 掩码、分块、数据类型、运行最大值修正和非法输入。Attention 只验证前向，不包含反向传播、dropout、padding mask 或全遮蔽行，不是完整 Transformer，也不是 FlashAttention GPU kernel。
