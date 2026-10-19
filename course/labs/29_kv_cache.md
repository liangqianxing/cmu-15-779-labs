# 第 5 讲后半：KV cache 省下哪些重复工作？

对应 [原课第 5 讲](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/05-transformers-attention.pdf)第 23-29 页。CPU 对照已经运行，GPU prefill/decode 吞吐、Flash-Decoding 的并行实现和速度实验按学习者要求跳过。

## Prefill 和 decode 的形状不同

Prefill 一次处理整段 prompt，许多 query 可以一起计算。Decode 每步加入一个新位置，只有一个新 query，但它要与所有合法历史 K/V 计算注意力。

之前的因果位置不应受未来 token 影响，所以可保存每层历史 K/V。新 token 只生成自己的 Q/K/V，追加 K/V，当前 Q 读取已有的全部缓存。旧 K/V 的投影无需在每步重新计算。

```text
prefill：prompt 的 Q/K/V -> 因果注意力 -> 保存历史 K/V
decode：新 token -> 新 Q/K/V -> 追加 K/V -> 新 Q 对全部缓存求注意力
```

这里不能把 `1 x prefix_length` 的分数矩阵套上左上对齐的三角掩码：它可能只留下第 0 个 key。当前 query 位于最后一个位置，缓存又只包含过去与当前位置，因此全部缓存 key 均合法。本实验的单 query 路径直接计算这一行，不额外构造三角掩码。

## CPU 对照的精确范围

[29_kv_cache.py](../../29_kv_cache.py)使用固定输入表示和固定投影权重，比较一个注意力层的两条路径：

1. 每个前缀都重新计算全部 Q/K/V、全部因果注意力，再取最后一行。
2. prompt 做一次 prefill，之后只投影新增一行并读取 KV cache。

默认 prompt=8，后续加入 4 个位置，比较长度 8、9、10、11、12 的五次结果：

| 观察项 | 每次完整前缀重算 | KV cache |
| --- | ---: | ---: |
| Q/K/V 每项投影所处理的 token 行数 | 50 | 12 |
| 计算的 query/key 分数对 | 510 | 106 |
| 相对完整前缀结果的最大绝对误差 | 基准 | 2.220e-16 |

缓存路径的分数对是 `8² + 9 + 10 + 11 + 12 = 106`。它仍然在每个 decode 步骤读取历史 K/V，不会让新 query 的注意力变成与上下文长度无关。

这些比值只针对本实验明确选择的完整前缀基线。若一个基线保留了别的中间状态、仅计算最后一行 attention，计算量也会不同；不能把本表比值当作所有推理实现通用的加速倍数。

## 时间换空间也有代价

本例一个注意力层、一个头、K 宽度 8、V 宽度 8、float64，到 12 个位置的缓存占 `12*(8+8)*8 = 1536` 字节。一般情况下每层还要乘 batch/请求数和 KV 头数，整个模型还要乘层数。GQA/MQA 的 KV 头数可能小于 query 头数，不能直接套用普通多头注意力的头数。

本脚本用 `torch.cat` 增长缓存，追加时会复制旧内容。这是便于核对数值的教学实现；后续服务系统需要预分配或分页管理来减少移动与浪费。后续 PagedAttention 课程会继续处理这个问题。

原课第 27 页以单 query 说明原先按 Q 划分并行工作的局限，不应理解成所有 FlashAttention 相关方法都不能用于任何 decode。第 29 页的 Flash-Decoding 按 K/V 拆分、最后合并部分结果；它仍要使用正确的 softmax 归一化统计，不能直接平均各块结果。

## 运行和结果

```bash
.venv/bin/python 29_kv_cache.py
.venv/bin/python 29_kv_cache.py --prompt 3 --steps 5 --dtype float32 --output results/29_my_run.json
```

默认运行更新 [原始结果](../../results/29_kv_cache.json)。实验使用一层注意力和固定输入表示；不生成文字，不是已训练语言模型，也没有测量 GPU 性能。完整结果包含每个前缀的误差、缓存元素和字节数、两种路径的工作量与源码哈希。

先解释为什么历史 K/V 可复用、为什么当前 query 仍要看完整历史，以及为什么这种缓存会随上下文和请求数增长。然后进入原课的高级 GPU 执行概念和编译章节。
