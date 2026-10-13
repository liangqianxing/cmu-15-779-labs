# CMU 15-779 代码实验

CMU 15-779 学习记录中的配套代码。第一课用 CPU 理解 PyTorch 的自动求导、手动梯度下降、SGD 和线性回归。

## 环境

- Python 3.11
- PyTorch 2.14.1
- NumPy 2.4.6

创建环境并安装依赖：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 运行

```bash
.venv/bin/python 01_autograd.py
.venv/bin/python 02_SGD.py
.venv/bin/python 03_linear.py
```

三个脚本分别对应：

- `01_autograd.py`：手动梯度下降，连续更新 10 次。
- `02_SGD.py`：使用 PyTorch SGD 更新标量参数。
- `03_linear.py`：用四个样本训练无偏置线性回归，并预测新输入。

配套学习记录：
[CMU 15-779 学习记录 01：PyTorch 自动求导与线性回归](https://liangqianxing.github.io/posts/cmu-15-779-pytorch-autograd-linear-regression/)
