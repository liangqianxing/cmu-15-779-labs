# CUDA 实验运行说明

本目录目前准备了向量加法源码 `25_vector_add.cu`。截至 2026-10-15，本机只有 CPU/MPS 可用，没有 NVIDIA CUDA 设备或 `nvcc`，因此该文件未编译、未执行。Python 索引演示的通过结果不能代替本目录的 GPU 验证。

## 在具备 NVIDIA GPU 和 CUDA Toolkit 的环境中验证

以下命令在仓库根目录运行；代码不依赖 PyTorch：

```bash
nvidia-smi
nvcc --version
mkdir -p build/cuda
nvcc -O2 -std=c++17 -lineinfo cuda/25_vector_add.cu -o build/cuda/25_vector_add
./build/cuda/25_vector_add
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/cuda/25_vector_add
```

`build/` 下是生成的可执行文件。首次运行前，需要实际可访问的 CUDA 设备、兼容驱动和 Toolkit。没有设备时程序返回失败，不会将跳过执行报告为通过。

程序检查六组输入：空向量、单元素、恰好整除、不能整除、小于一个线程块、跨越多个线程块。每个非空例子都分配数据、启动 kernel、等待完成、复制结果、逐元素核对并释放内存。空输入直接返回，不启动零大小 grid。

输出全对还不代表无越界访问；`compute-sanitizer` 用于额外检查设备内存错误。这里显式设置 `--error-exitcode 1`，避免应用本身正常退出而检查工具发现错误时仍返回成功。依据：[NVIDIA Compute Sanitizer 官方文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

## 需要保存的执行证据

- [ ] GPU 型号、驱动、CUDA Toolkit 和编译命令。
- [ ] 源码版本或 SHA-256。
- [ ] 可执行文件完整输出，所有非空案例为 PASS，命令退出码为 0。
- [ ] Memcheck 完整输出与退出码，未检出设备内存错误。

当前没有这些 GPU 执行证据。后续有可用环境时再实际运行，不以预期输出填充结果文件。

这个实验只检查功能，不给出速度结论。后续计时需要区分 kernel 时间、数据传输及主机端开销，并明确同步方式。
