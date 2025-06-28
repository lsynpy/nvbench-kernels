#include <nvbench/nvbench.cuh>
#include <cinttypes>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

template <typename T>
__device__ __host__ T one_value();

template <>
__device__ __host__ __half one_value<__half>() {
  return __float2half(1.0f);
}

template <>
__device__ __host__ __half2 one_value<__half2>() {
  return __half2half2(__float2half(1.0f));
}

template <>
__device__ __host__ float one_value<float>() {
  return 1.0f;
}

template <typename T>
__device__ __host__ T zero_value();

template <>
__device__ __host__ __half zero_value<__half>() {
  return __float2half(0.0f);
}

template <>
__device__ __host__ __half2 zero_value<__half2>() {
  return __float2half2_rn(0.0f);
}

template <>
__device__ __host__ float zero_value<float>() {
  return 0.0f;
}

template <typename T>
__global__ void add_kernel(T *out, T a, int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T acc = zero_value<T>();

#pragma unroll 64
  for (int i = 0; i < iters; ++i) {
    if constexpr (std::is_same_v<T, __half>) {
      acc = __hadd(acc, a);
    } else if constexpr (std::is_same_v<T, __half2>) {
      acc = __hadd2(acc, a);
    } else {
      acc += a;
    }
  }

  out[idx] = acc;
}

template <typename T>
void half_benchmark(nvbench::state &state, nvbench::type_list<T>) {
  T *d_output;
  const auto iters = state.get_int64("Iters");
  const auto block_size = state.get_int64("BlockSize");
  const auto grid_size = state.get_int64("NumBlocks");

  int64_t num_elements = block_size * grid_size;
  int64_t flops_per_elem = iters;
  if constexpr (std::is_same_v<T, __half2>) {
    flops_per_elem *= 2;
  }
  int64_t num_flops = num_elements * flops_per_elem;
  cudaMalloc(&d_output, sizeof(T) * num_elements);

  state.add_element_count(num_flops, "NumFLOPs");

  state.exec([=](nvbench::launch &launch) {
    add_kernel<T><<<grid_size, block_size>>>(d_output, one_value<T>(), iters);
  });

  T h_output;
  cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_output);
}

using cts_types = nvbench::type_list<float, __half, __half2>;

NVBENCH_BENCH_TYPES(half_benchmark, NVBENCH_TYPE_AXES(cts_types))
  .add_int64_power_of_two_axis("Iters", nvbench::range(20, 20, 1))
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(9, 9, 1))
  .add_int64_power_of_two_axis("NumBlocks", nvbench::range(10, 10, 1));
