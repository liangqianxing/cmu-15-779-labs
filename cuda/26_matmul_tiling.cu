#include <cuda_runtime.h>

#include <algorithm>
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

__global__ void matmul_naive(const float* a, const float* b, float* c,
                             int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + col];
    }
    c[row * n + col] = sum;
}

template <int Tile>
__global__ void matmul_tiled(const float* a, const float* b, float* c,
                             int m, int k, int n) {
    __shared__ float shared_a[Tile][Tile];
    __shared__ float shared_b[Tile][Tile];
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * Tile + ty;
    const int col = blockIdx.x * Tile + tx;
    float sum = 0.0f;

    // Even out-of-range output threads load tiles and reach both barriers.
    for (int base_k = 0; base_k < k; base_k += Tile) {
        shared_a[ty][tx] = row < m && base_k + tx < k
            ? a[row * k + base_k + tx] : 0.0f;
        shared_b[ty][tx] = base_k + ty < k && col < n
            ? b[(base_k + ty) * n + col] : 0.0f;
        __syncthreads();

        for (int inner = 0; inner < Tile; ++inner) {
            sum += shared_a[ty][inner] * shared_b[inner][tx];
        }
        // No thread may overwrite a shared tile while another still reads it.
        __syncthreads();
    }
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

template <int Tile>
static bool run_case(int m, int k, int n, const cudaDeviceProp& properties) {
    if (Tile * Tile > properties.maxThreadsPerBlock ||
        Tile > properties.maxThreadsDim[0] || Tile > properties.maxThreadsDim[1] ||
        2 * Tile * Tile * sizeof(float) > properties.sharedMemPerBlock) {
        std::fprintf(stderr, "Tile exceeds device limits.\n");
        return false;
    }
    if (m == 0 || n == 0) {
        std::printf("m=%d k=%d n=%d tile=%d: empty output, no launch\n", m, k, n, Tile);
        return true;
    }
    std::vector<float> a(m * k), b(k * n), output(m * n);
    std::vector<double> expected(m * n, 0.0);
    for (int i = 0; i < m * k; ++i) {
        a[i] = static_cast<float>(i % 13 - 6) / 4.0f;
    }
    for (int i = 0; i < k * n; ++i) {
        b[i] = static_cast<float>(i % 11 - 5) / 8.0f;
    }
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            for (int inner = 0; inner < k; ++inner) {
                expected[row * n + col] +=
                    static_cast<double>(a[row * k + inner]) * b[inner * n + col];
            }
        }
    }

    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    // For K=0 the kernels write zeros without dereferencing either input.
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a),
                          std::max(size_t{1}, a.size()) * sizeof(float)), "allocate A");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b),
                          std::max(size_t{1}, b.size()) * sizeof(float)), "allocate B");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_c), output.size() * sizeof(float)), "allocate C");
    if (!a.empty()) {
        check_cuda(cudaMemcpy(device_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice), "copy A");
        check_cuda(cudaMemcpy(device_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    }
    const dim3 threads(Tile, Tile);
    const dim3 blocks((n + Tile - 1) / Tile, (m + Tile - 1) / Tile);
    bool all_correct = true;
    for (int version = 0; version < 2; ++version) {
        const char* name = version == 0 ? "naive" : "tiled";
        check_cuda(cudaMemset(device_c, 0xff, output.size() * sizeof(float)), "initialize C with NaNs");
        if (version == 0) {
            matmul_naive<<<blocks, threads>>>(device_a, device_b, device_c, m, k, n);
        } else {
            matmul_tiled<Tile><<<blocks, threads>>>(device_a, device_b, device_c, m, k, n);
        }
        check_cuda(cudaGetLastError(), "launch matmul");
        check_cuda(cudaDeviceSynchronize(), "execute matmul");
        check_cuda(cudaMemcpy(output.data(), device_c, output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy C");
        bool correct = true;
        double max_error = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            const double error = std::abs(output[i] - expected[i]);
            if (!std::isfinite(output[i]) || error > 1e-5 + 1e-5 * std::abs(expected[i])) {
                std::fprintf(stderr, "%s C[%zu]: expected %.9g, got %.9g\n", name, i, expected[i], output[i]);
                correct = false;
                break;
            }
            max_error = std::max(max_error, error);
        }
        std::printf("m=%d k=%d n=%d tile=%d %s: %s max_abs_error=%g\n",
                    m, k, n, Tile, name, correct ? "PASS" : "FAIL", max_error);
        all_correct = correct && all_correct;
    }
    check_cuda(cudaFree(device_a), "free A");
    check_cuda(cudaFree(device_b), "free B");
    check_cuda(cudaFree(device_c), "free C");
    return all_correct;
}

int main() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "query CUDA devices");
    if (count == 0) {
        std::fprintf(stderr, "No CUDA device available.\n");
        return EXIT_FAILURE;
    }
    check_cuda(cudaSetDevice(0), "select device");
    cudaDeviceProp properties;
    check_cuda(cudaGetDeviceProperties(&properties, 0), "query device properties");
    int driver = 0, runtime = 0;
    check_cuda(cudaDriverGetVersion(&driver), "query driver version");
    check_cuda(cudaRuntimeGetVersion(&runtime), "query runtime version");
    std::printf("Device: %s, compute capability: %d.%d, driver API: %d, runtime: %d\n",
                properties.name, properties.major, properties.minor, driver, runtime);
    std::printf("Correctness checks only; no performance measurements.\n");

    bool ok = true;
    ok = run_case<4>(0, 5, 3, properties) && ok;
    ok = run_case<4>(3, 5, 0, properties) && ok;
    ok = run_case<4>(3, 0, 5, properties) && ok;
    ok = run_case<4>(1, 1, 1, properties) && ok;
    ok = run_case<2>(4, 4, 4, properties) && ok;
    ok = run_case<4>(3, 5, 7, properties) && ok;
    ok = run_case<2>(7, 3, 5, properties) && ok;
    ok = run_case<8>(17, 9, 13, properties) && ok;
    ok = run_case<1>(8, 8, 8, properties) && ok;
    ok = run_case<2>(8, 8, 8, properties) && ok;
    ok = run_case<4>(8, 8, 8, properties) && ok;
    ok = run_case<8>(8, 8, 8, properties) && ok;
    ok = run_case<16>(17, 9, 13, properties) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
