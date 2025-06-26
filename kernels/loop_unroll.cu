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
  const auto iters = state.get_int64("Iters");
  const auto block_size = state.get_int64("BlockSize");
  const auto grid_size = state.get_int64("NumBlocks");
  int64_t num_elements = block_size * grid_size;
  int64_t flops_per_elem = iters;
  int64_t num_flops = num_elements * flops_per_elem;
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

NVBENCH_BENCH(add_benchmark)
  .add_int64_power_of_two_axis("Iters", nvbench::range(20, 20, 1))
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(9, 10, 1))
  .add_int64_power_of_two_axis("NumBlocks", nvbench::range(9, 10, 1));