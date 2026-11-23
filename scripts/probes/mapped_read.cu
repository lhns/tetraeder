// Host-memory read rates seen from the GPU (ADR 0084, step 1): a kernel that sums a mapped pinned buffer (the read the
// position-split KV route would do) and cudaMemcpy host to device of the same buffer (the copy route).
// Build: nvcc -O2 -o mapped_read scripts/probes/mapped_read.cu   Run: ./mapped_read [GB 1 to 4, default 2] [repeats, default 5]
// Prints one line per repeat and a median line each for "mapped_kernel" and "memcpy_h2d", in GB/s (1e9 bytes).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// grid-stride sum of 16-byte loads; the sum is stored so the loads are not removed
__global__ void sum_kernel(const uint4 *p, size_t n, unsigned long long *out) {
  unsigned long long acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc += (unsigned long long)v.x + v.y + v.z + v.w;
  }
  atomicAdd(out, acc);
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char **argv) {
  double gb = argc > 1 ? atof(argv[1]) : 2.0;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  if (gb < 1.0 || gb > 4.0 || reps < 1) { fprintf(stderr, "usage: mapped_read [GB 1..4] [repeats]\n"); return 2; }
  size_t bytes = (size_t)(gb * (1ull << 30));
  size_t n16 = bytes / 16;
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("device %s, %.2f GB buffer, %d repeats\n", prop.name, gb, reps);

  void *host = nullptr;
  CK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
  for (size_t i = 0; i < bytes / 8; i++) ((uint64_t *)host)[i] = i * 2654435761u;
  void *dptr = nullptr;
  CK(cudaHostGetDevicePointer(&dptr, host, 0));
  void *dev = nullptr;
  CK(cudaMalloc(&dev, bytes));
  unsigned long long *dsum;
  CK(cudaMalloc(&dsum, sizeof(*dsum)));
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  int blocks = prop.multiProcessorCount * 8, threads = 256;

  std::vector<double> rk, rc;
  for (int r = -1; r < reps; r++) { // r = -1 is a warm-up
    float ms;
    CK(cudaMemset(dsum, 0, sizeof(*dsum)));
    CK(cudaEventRecord(t0));
    sum_kernel<<<blocks, threads>>>((const uint4 *)dptr, n16, dsum);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    CK(cudaGetLastError());
    CK(cudaEventElapsedTime(&ms, t0, t1));
    double gk = bytes / 1e9 / (ms / 1e3);
    CK(cudaEventRecord(t0));
    CK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1));
    double gc = bytes / 1e9 / (ms / 1e3);
    if (r >= 0) {
      printf("rep %d mapped_kernel GB/s=%.2f memcpy_h2d GB/s=%.2f\n", r, gk, gc);
      rk.push_back(gk);
      rc.push_back(gc);
    }
  }
  printf("median mapped_kernel GB/s=%.2f\nmedian memcpy_h2d GB/s=%.2f\n", median(rk), median(rc));
  CK(cudaFreeHost(host));
  CK(cudaFree(dev));
  CK(cudaFree(dsum));
  return 0;
}
