#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

#define FLOAT4(a) *(float4 *)(&(a))
#define CEIL(a, b) ((a + b - 1) / (b))

__global__ void elementwise_add_fp32_kernel(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_add_fp32x4_kernel(float *a, float *b, float *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}

void elementwise_add_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto block_size = state.get_int64("BlockSize");
  const auto variant = state.get_string("Variants");

  thrust::device_vector<float> a(N);
  thrust::device_vector<float> b(N);
  thrust::device_vector<float> c(N);
  thrust::fill(a.begin(), a.end(), 1.0f);
  thrust::fill(b.begin(), b.end(), 2.0f);

  state.add_element_count(N, "NumFLOPs");
  state.add_global_memory_reads<nvbench::int32_t>(N * 2);
  state.add_global_memory_writes<nvbench::int32_t>(N);

  state.exec([&](nvbench::launch &launch) {
    if (variant == "fp32") {
      const auto grid_size = CEIL(N, block_size);
      elementwise_add_fp32_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        N
      );
    } else if (variant == "fp32x4") {
      const auto grid_size = CEIL(CEIL(N, 4), block_size);
      elementwise_add_fp32x4_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        N
      );
    }
  });
}

NVBENCH_BENCH(elementwise_add_bench)
  .add_int64_power_of_two_axis("N", nvbench::range(20, 21, 1))
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(8, 10, 1))
  .add_string_axis("Variants", {"fp32", "fp32x4"});