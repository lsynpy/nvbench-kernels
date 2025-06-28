#include <nvbench/nvbench.cuh>
#include <cinttypes>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__device__ __host__ T one_value() {
  return static_cast<T>(1);
}

template <>
__device__ __host__ __half one_value<__half>() {
  return __float2half(1.0f);
}

template <>
__device__ __host__ __nv_bfloat16 one_value<__nv_bfloat16>() {
  return __float2bfloat16(1.0f);
}

template <typename T, int Unroll>
__global__ void add_kernel(T *out, T a, int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T acc = static_cast<T>(0);

  if constexpr (Unroll == 0) {
    for (int i = 0; i < iters; ++i) {
      if constexpr (std::is_same_v<T, __half>) {
        acc = __hadd(acc, a);
      } else {
        acc += a;
      }
    }
  } else {
#pragma unroll Unroll
    for (int i = 0; i < iters; ++i) {
      if constexpr (std::is_same_v<T, __half>) {
        acc = __hadd(acc, a);
      } else {
        acc += a;
      }
    }
  }

  out[idx] = acc;
}

// Kernel launcher dispatcher
template <typename T>
void launch_kernel(T *out, T a, int iters, int block_size, int grid_size, int unroll, cudaStream_t stream) {
  switch (unroll) {
  case 1:
    add_kernel<T, 1><<<grid_size, block_size, 0, stream>>>(out, a, iters);
    break;
  case 4:
    add_kernel<T, 4><<<grid_size, block_size, 0, stream>>>(out, a, iters);
    break;
  case 8:
    add_kernel<T, 8><<<grid_size, block_size, 0, stream>>>(out, a, iters);
    break;
  case 1024:
    add_kernel<T, 1024><<<grid_size, block_size, 0, stream>>>(out, a, iters);
    break;
  case 0:
    add_kernel<T, 0><<<grid_size, block_size, 0, stream>>>(out, a, iters);
    break; // plain
  default:
    throw std::runtime_error("Unsupported unroll factor");
  }
}

template <typename T>
void add_benchmark(nvbench::state &state, nvbench::type_list<T>) {
  T *d_output;
  const auto iters = state.get_int64("Iters");
  const auto block_size = state.get_int64("BlockSize");
  const auto grid_size = state.get_int64("NumBlocks");
  const auto variant_str = state.get_string("Variants");

  int unroll = 0;
  if (variant_str == "plain")
    unroll = 0;
  else if (variant_str == "unroll1")
    unroll = 1;
  else if (variant_str == "unroll4")
    unroll = 4;
  else if (variant_str == "unroll8")
    unroll = 8;
  else if (variant_str == "unroll1024")
    unroll = 1024;
  else
    throw std::runtime_error("Unknown variant");

  int64_t num_elements = block_size * grid_size;
  int64_t flops_per_elem = iters;
  int64_t num_flops = num_elements * flops_per_elem;
  cudaMalloc(&d_output, sizeof(T) * num_elements);

  state.add_element_count(num_flops, "NumFLOPs");

  state.exec([=](nvbench::launch &launch) {
    launch_kernel<T>(d_output, one_value<T>(), iters, block_size, grid_size, unroll, launch.get_stream());
  });

  T h_output;
  cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_output);
}

using cts_types = nvbench::type_list<float, __half>;

NVBENCH_BENCH_TYPES(add_benchmark, NVBENCH_TYPE_AXES(cts_types))
  .add_int64_power_of_two_axis("Iters", nvbench::range(20, 20, 1))
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(9, 9, 1))
  .add_int64_power_of_two_axis("NumBlocks", nvbench::range(10, 10, 1))
  .add_string_axis("Variants", {"plain", "unroll1", "unroll4", "unroll8", "unroll1024"});
