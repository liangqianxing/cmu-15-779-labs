# CMU 15-779 代码实验

CMU 15-779 学习记录中的配套代码。已运行的 CPU 练习包含自动求导、线性回归、小型 ReLU 网络、DataLoader、参数保存、批量推理、绝对值网络、算子性能分析、CUDA 索引演示、矩阵分块、树形归约、在线 Softmax、KV cache、CPU 编译优化，以及并行训练、推理服务、LoRA 和 MoE 的 CPU 实验。2026-10-19 已在获授权的单卡环境运行 `cuda/` 中的向量加法和矩阵乘法，并补入 Triton 与显式 CUDA SDPA 后端记录；历史 CPU 报告保留原样。

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
| [04：多输出线性层、ReLU 与绝对值网络](https://liangqianxing.github.io/posts/cmu-15-779-pytorch-linear-relu-absolute-value/) | `16` 至 `21` | Linear 参数对照、多输出、逐层计算、绝对值拟合、分段线性折点斜率与内插/外推泛化对照 |
| [05：Profiler、算子调用与批量推理](https://liangqianxing.github.io/posts/cmu-15-779-profiler-operator-batching/) | `22` 至 `24` | 前向计算分析、调用次数与固定工作量的 batch size 扫描 |
| [06：CUDA 线程索引与边界检查](https://liangqianxing.github.io/posts/cmu-15-779-cuda-thread-indexing/) | `25`、`38` | CPU 索引演示；新增原生 CUDA 编译、memcheck 与 Triton 向量加法 |
| [07：矩阵分块、数据复用与树形归约](https://liangqianxing.github.io/posts/cmu-15-779-matmul-tiling-tree-reduction/) | `26`、`27`、`38` | CPU 逻辑计数；新增 CUDA 矩阵及 Triton 归约/矩阵实测 |
| [08：分块 Attention、在线 Softmax 与 KV cache](https://liangqianxing.github.io/posts/cmu-15-779-attention-online-softmax/) | `28`、`29`、`38` | CPU 在线归一化/KV；新增 CUDA 显式 MATH/FLASH 前向和调用区间计时 |
| [09：CPU 编译优化、合法重写与候选搜索](https://liangqianxing.github.io/posts/cmu-15-779-cpu-compiler-optimizations/) | `30` | BatchNorm 折叠、RMSNorm 重排、分块候选计时与 TorchInductor CPU 编译 |
| [10：加权梯度、状态分片与流水线](https://liangqianxing.github.io/posts/cmu-15-779-cpu-parallelism-pipeline/) | `31`、`32` | 双 CPU 进程 Gloo；张量切分、持久状态及同步流水线成本模型 |
| [11：分页 KV、连续批处理与推测采样](https://liangqianxing.github.io/posts/cmu-15-779-kv-serving-speculative-sampling/) | `33`、`34`、`37` | CPU 写时复制、单位成本调度与小概率模型的分布修正 |
| [12：LoRA 与均匀 4-bit 打包](https://liangqianxing.github.io/posts/cmu-15-779-lora-uniform-quantization/) | `35` | 低秩分支、首步梯度、合并及有损均匀量化；未实现 QLoRA |
| [13：MoE 路由与容量](https://liangqianxing.github.io/posts/cmu-15-779-moe-routing-capacity/) | `36` | CPU 稀疏分发、输出/梯度对照和容量丢弃的数值影响 |
| [14：KV 页大小、共享边界与 CPU 实现成本](https://liangqianxing.github.io/posts/cmu-15-779-kv-cache-granularity-study/) | `course/project/` | 210 组配置、1470 次原始计时；活跃槽位与池容量分列，保留慢于基线的结果 |

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
.venv/bin/python 20_piecewise_linear_slopes.py
.venv/bin/python 21_generalization_split.py
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
- `25_cuda_indexing.py` 只在 CPU 上演示索引映射，原始结果保存在 `results/25_cuda_indexing.json` 和 `results/25_indexing_17_8.json`。它不测 GPU 性能，也不能证明 CUDA 设备内存访问安全。`cuda/25_vector_add.cu` 在首次 CPU 记录中尚未编译执行；2026-10-19 的实际续验见 [CUDA 运行说明](cuda/README.md)。

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

`tests/` 现包含 56 项测试，覆盖分段线性 ReLU 折点斜率与内插/外推、矩阵乘法边界与计数、归约长度及浮点求和顺序、Attention 掩码、分块、数据类型、运行最大值修正和非法输入。Attention 只验证前向，不包含反向传播、dropout、padding mask 或全遮蔽行，不是完整 Transformer，也不是 FlashAttention GPU kernel。

## 2026-10-19：Attention 续篇与 CPU 编译实验

这次历史发布以冻结的 29/30 号脚本和原始结果为界，当时源学习聊天仍在继续。KV cache 合并到记录 08，CPU 编译实验为记录 09。新增内容均为个人配套实验，不是官方作业或整门课程完成证明。当时按学习安排跳过真实 GPU 实验，后续实际运行见文末续篇。

```bash
.venv/bin/python 29_kv_cache.py --output results/29_kv_cache-rerun.json
.venv/bin/python 30_cpu_compiler.py --compile --output results/30_cpu_compiler-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

29 号以每次重算整个前缀的投影和完整因果 Attention 为基线，计数是算法工作量，不是加速倍数或 FLOPs。它只验证固定输入表示的一层单头 Attention，使用 `torch.cat` 增长缓存，不生成文本。

30 号原始报告的 `inductor.status` 为 `executed`；编译后函数通过两组输入对照。首调用包含编译或缓存查询，另列预热后计时。分块候选是 Python 循环调用 PyTorch CPU 矩阵乘法，不能解释为生成了 Triton/TVM GPU kernel。BatchNorm 折叠要求 eval 模式和固定运行统计量；RMSNorm 重排只针对脚本中的无偏置线性映射及固定参数。

`results/29_cpu_tests.txt` 是源聊天保存的 25 项测试日志；`results/30_cpu_tests.txt` 是本次冻结副本复验的 30 项日志，新增 KV cache 和编译测试均通过。源码与原始结果对应的 SHA-256 可在各 JSON 的 `script_sha256` 中核对。计时会随运行环境和编译缓存状态变化。

## 2026-10-19：并行训练、服务、LoRA 与 MoE

发布记录 10 至 13 使用独立冻结的 31 至 37 号脚本与原始结果。源聊天已报告这部分实验完成，课程项目当时仍在推进，本版本不包含项目结果或课程完成声明。原始 JSON 的 `script_sha256` 对应公开脚本；本次复跑保留原始报告，另存验证证据。全部 52 项现有测试通过，新增实验覆盖其中 22 项，日志见 `results/37_cpu_tests.txt`。

```bash
.venv/bin/python 31_cpu_parallelism.py --gloo --output results/31-rerun.json
.venv/bin/python 32_pipeline_schedule.py --output results/32-rerun.json
.venv/bin/python 33_paged_kv_cache.py --output results/33-rerun.json
.venv/bin/python 34_speculative_sampling.py --output results/34-rerun.json
.venv/bin/python 35_lora_quantization.py --output results/35-rerun.json
.venv/bin/python 36_moe_dispatch.py --output results/36-rerun.json
.venv/bin/python 37_continuous_batching.py --output results/37-rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

31 号 `--gloo` 实际启动单机两个 CPU 进程，执行手工 all-reduce SUM 和一次 SGD 对照；需要 PyTorch 构建启用 Gloo。未加该参数会记为 `not_requested`。局部 loss 按样本数加权是当前每样本输出维数相同、元素均值 MSE 的条件；token loss 需要对应实际全局分母。ZeRO 字节数是持久状态模型，没有执行 FSDP/ZeRO。

32 号流水线与 37 号连续批处理只使用输入的模拟成本，不能当成硬件计时。33 号页池预分配，回收逻辑页不缩小底层张量，gather 会额外物化连续 K/V。34 号是条件概率小模型，枚举单步质量并记录首 token 频率，没有 Transformer 并行验证或速度测量。35 号只做一次 LoRA SGD 步；均匀 4-bit 码本不是 NF4/QLoRA。36 号串行执行 CPU 专家；容量截断会改变输出。这批历史实验均在 CPU 上运行。

## 2026-10-19：实际 GPU 验证与 KV 页粒度项目实验

GPU 验证环境报告 Python 3.12.3、PyTorch 2.7.0+cu128、Triton 3.3.0、CUDA Toolkit 12.8，设备名称报告为 RTX 4080 SUPER。CPU 历史报告使用另一套环境，不覆盖其依赖和数值。GPU 环境安装说明见 `requirements-gpu.txt` 与 `GPU_REPRODUCE.md`。38 号、原生 runner、两份 JSON 均从源学习仓库冻结；发布端核对源码哈希、24 组逐次 event 数据与中位数、原生命令退出码，没有在 Mac 上假称复跑 GPU。

```bash
python cuda/run_native_checks.py --output results/cuda-native-rerun.json
python 38_gpu_kernels.py --output results/38-gpu-rerun.json
```

原生 CUDA 向量加法和矩阵乘法已编译运行，指定的 Compute Sanitizer 检查未报告错误。Triton 向量加法、两阶段归约、矩阵乘法及显式 MATH/FLASH SDPA 前向通过数值容差检查。CUDA event 时间是调用提交区间，可能包含分配或 Python 提交造成的 GPU 空闲，不是孤立 kernel 或端到端服务延迟。未测量多 GPU、训练收敛或完整 LLM；手写 Triton 归约/矩阵在本次计时中慢于 PyTorch。

记录 14 复用 33 号缓存管理，比较静态预留、动态拼接、独立分页和共享分页。源实验 210 组配置及1470次计时已完成；发布端从冻结代码重跑完整扫描并核对所有存储/复制计数与输出。两项项目测试及此前测试共54项通过，日志在 `results/38_cpu_tests.txt`。课程项目提案、最终报告 PDF、视频和正式提交状态不在这个代码实验发布中作完成声明。

## 2026-10-19：项目报告与中文展示归档

记录 14 补入[独立文档快照](archives/kv-cache-deliverables-20261019/README.md)：一页项目规格、正文 5 页另加参考文献的英文报告、9 分钟中文合成语音展示及讲稿/制作源码。文件来自源聊天已完成的输出，逐页/逐章检查并完整解码视频；这是材料归档，不是正式课程提交、学习者本人录制或掌握程度的证明。原实验脚本、原始数据和先前发布的图表版本保留。
