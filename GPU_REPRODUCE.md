# 实际 GPU 记录与复现

源聊天在获授权的单卡 CUDA 环境运行，发布端核对源码哈希及原始结果。没有附带登录地址或凭据。

已运行环境：Python 3.12.3、PyTorch 2.7.0+cu128、Triton 3.3.0、CUDA Toolkit 12.8，设备报告 RTX 4080 SUPER、compute capability 8.9。CPU 根 requirements.txt 用于此前的 Mac 实验，不能与 GPU 环境混装。

```bash
python3.12 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install -r requirements-gpu.txt
.venv-gpu/bin/python cuda/run_native_checks.py --output results/cuda-native-rerun.json
.venv-gpu/bin/python 38_gpu_kernels.py --output results/38-gpu-rerun.json
```

原生 runner 需要 NVIDIA 驱动、nvcc、Compute Sanitizer；工具不在 PATH 时尝试 /usr/local/cuda/bin。先结束 sanitizer 再计时。CUDA 不可用会报错，不记作通过。GPU包不需要CPU或项目依赖。

原生结果在 results/cuda_native_checks.json，包含两份CUDA源码哈希、编译/功能/指定sanitizer输出和10条命令退出码。38号结果在 results/38_gpu_kernels.json，包含脚本哈希、环境、数值误差及24组逐次event时间。所有event中位数由15个样本计算，每样本含5或10次调用，另有5次warmup。初次JIT和主机/设备传输在稳定计时外；区间可能包含Python提交/分配造成的GPU空闲，不是单kernel精确耗时或服务延迟。

单卡合成张量，前向验证；没有完整LLM、训练收敛、多GPU通信、手写FlashAttention或megakernel。本次朴素Triton归约和矩阵乘法慢于PyTorch，原始负例保留。
