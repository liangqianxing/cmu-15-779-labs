#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static bool run_case(int n, int threads) {
    if (n == 0) {
        std::printf("n=0: no launch required\n");
        return true;
    }

    std::vector<float> a(n), b(n, 10.0f), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes), "allocate a");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes), "allocate b");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_c), bytes), "allocate c");
    check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy a");
    check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy b");
    // Unwritten outputs become NaNs, so skipped elements cannot pass validation.
    check_cuda(cudaMemset(device_c, 0xff, bytes), "initialize c");

    const int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(device_a, device_b, device_c, n);
    check_cuda(cudaGetLastError(), "launch vector_add");
    check_cuda(cudaDeviceSynchronize(), "execute vector_add");
    check_cuda(cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost), "copy c");

    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(c[i]) || c[i] != a[i] + b[i]) {
            std::fprintf(stderr, "n=%d, i=%d: expected %g, got %g\n", n, i, a[i] + b[i], c[i]);
            correct = false;
            break;
        }
    }
    check_cuda(cudaFree(device_a), "free a");
    check_cuda(cudaFree(device_b), "free b");
    check_cuda(cudaFree(device_c), "free c");
    std::printf("n=%d, threads=%d, blocks=%d: %s\n", n, threads, blocks, correct ? "PASS" : "FAIL");
    return correct;
}

int main() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "query CUDA devices");
    if (count == 0) {
        std::fprintf(stderr, "No CUDA device available.\n");
        return EXIT_FAILURE;
    }
    check_cuda(cudaSetDevice(0), "select CUDA device");
    cudaDeviceProp properties;
    check_cuda(cudaGetDeviceProperties(&properties, 0), "query device properties");
    std::printf("Device: %s, compute capability: %d.%d\n", properties.name, properties.major, properties.minor);

    const int cases[][2] = {{0, 4}, {1, 4}, {8, 4}, {10, 4}, {17, 32}, {257, 128}};
    bool all_correct = true;
    for (const auto& item : cases) {
        const int threads = item[1];
        if (threads > properties.maxThreadsPerBlock || threads > properties.maxThreadsDim[0]) {
            std::fprintf(stderr, "Requested block size exceeds device limits.\n");
            return EXIT_FAILURE;
        }
        all_correct = run_case(item[0], threads) && all_correct;
    }
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
