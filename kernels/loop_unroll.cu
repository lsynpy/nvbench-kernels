#include <nvbench/nvbench.cuh>
#include <cinttypes>

__global__ void add_kernel(float *out, float a, int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0;

#pragma unroll
  for (int i = 0; i < iters; ++i) {
    acc += a;
  }

  out[idx] = acc;
}

void add_benchmark(nvbench::state &state) {
  float *d_output;
  int iters = 1024 * 1024;
  int block_size = 256;
  int grid_size = 256;
  int num_elements = block_size * grid_size;
  int flops_per_elem = iters;
  int64_t num_flops = static_cast<int64_t>(num_elements) * static_cast<int64_t>(flops_per_elem);
  cudaMalloc(&d_output, sizeof(float) * num_elements);

  printf("num_flops: %" PRId64 "\n", num_flops);
  state.add_element_count(num_flops, "NumFLOPs"); // use elem/s as FLOPS

  state.exec([grid_size, block_size, iters, d_output](nvbench::launch &launch) {
    add_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(d_output, 1.0f, iters);
  });

  float h_output;
  cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_output);
}

NVBENCH_BENCH(add_benchmark);