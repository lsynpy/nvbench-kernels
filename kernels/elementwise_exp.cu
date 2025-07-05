#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// --- FP32 kernels ---
__global__ void elementwise_exp_fp32_kernel(float *a, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = expf(a[idx]);
}

__global__ void elementwise_exp_fp32x4_kernel(float *a, float *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_c;
    reg_c.x = expf(reg_a.x);
    reg_c.y = expf(reg_a.y);
    reg_c.z = expf(reg_a.z);
    reg_c.w = expf(reg_a.w);
    FLOAT4(c[idx]) = reg_c;
  }
}

// --- FP16 kernels ---
__global__ void elementwise_exp_f16_kernel(half *a, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = hexp(a[idx]);
}

__global__ void elementwise_exp_f16x2_kernel(half *a, half *c, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_c = h2exp(reg_a);
    HALF2(c[idx]) = reg_c;
  }
}

__global__ void elementwise_exp_f16x8_kernel(half *a, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 6 < N) {
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);

    half2 reg_c_0 = h2exp(reg_a_0);
    half2 reg_c_1 = h2exp(reg_a_1);
    half2 reg_c_2 = h2exp(reg_a_2);
    half2 reg_c_3 = h2exp(reg_a_3);

    HALF2(c[idx + 0]) = reg_c_0;
    HALF2(c[idx + 2]) = reg_c_1;
    HALF2(c[idx + 4]) = reg_c_2;
    HALF2(c[idx + 6]) = reg_c_3;
  } else {
    for (int i = 0; i < 8 && (idx + i) < N; ++i) {
      c[idx + i] = hexp(a[idx + i]);
    }
  }
}

__global__ void elementwise_exp_f16x8_pack_kernel(half *a, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_a[8], pack_c[8];

  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);

#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    HALF2(pack_c[i]) = h2exp(HALF2(pack_a[i]));
  }

  if ((idx + 7) < N) {
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  } else {
    for (int i = 0; i < 8 && (idx + i) < N; ++i) {
      c[idx + i] = pack_c[i];
    }
  }
}

void elementwise_exp_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto block_size = state.get_int64("BlockSize");
  const auto variant = state.get_string("Variants");

  const bool is_fp16 = (variant.find("f16") != std::string::npos);

  if (is_fp16) {
    thrust::device_vector<half> a(N);
    thrust::device_vector<half> c(N);

    thrust::fill(a.begin(), a.end(), __float2half(1.0f));

    state.add_element_count(N, "NumFLOPs");
    state.add_global_memory_reads<nvbench::int16_t>(N);
    state.add_global_memory_writes<nvbench::int16_t>(N);

    state.exec([&](nvbench::launch &launch) {
      if (variant == "f16") {
        const auto grid_size = CEIL(N, block_size);
        elementwise_exp_f16_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(c.data()), N
        );
      } else if (variant == "f16x2") {
        const auto grid_size = CEIL(CEIL(N, 2), block_size);
        elementwise_exp_f16x2_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(c.data()), N
        );
      } else if (variant == "f16x8") {
        const auto grid_size = CEIL(CEIL(N, 8), block_size);
        elementwise_exp_f16x8_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(c.data()), N
        );
      } else if (variant == "f16x8_pack") {
        const auto grid_size = CEIL(CEIL(N, 8), block_size);
        elementwise_exp_f16x8_pack_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(c.data()), N
        );
      }
    });
  } else {
    thrust::device_vector<float> a(N);
    thrust::device_vector<float> c(N);

    thrust::fill(a.begin(), a.end(), 1.0f);

    state.add_element_count(N, "NumFLOPs");
    state.add_global_memory_reads<nvbench::int32_t>(N);
    state.add_global_memory_writes<nvbench::int32_t>(N);

    state.exec([&](nvbench::launch &launch) {
      if (variant == "fp32") {
        const auto grid_size = CEIL(N, block_size);
        elementwise_exp_fp32_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(c.data()), N
        );
      } else if (variant == "fp32x4") {
        const auto grid_size = CEIL(CEIL(N, 4), block_size);
        elementwise_exp_fp32x4_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(c.data()), N
        );
      }
    });
  }
}

NVBENCH_BENCH(elementwise_exp_bench)
  .add_int64_power_of_two_axis("N", nvbench::range(21, 21, 1))
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(9, 9, 1))
  .add_string_axis("Variants", {"fp32", "fp32x4", "f16", "f16x2", "f16x8", "f16x8_pack"});

/*
## elementwise_exp_bench

### [0] NVIDIA GeForce RTX 3050

|       N        | BlockSize |  Variants  | Samples | GPU Time  | Noise | Elem/s  | GlobalMem BW | BWUtil |
|----------------|-----------|------------|---------|-----------|-------|---------|--------------|--------|
| 2^21 = 2097152 | 2^9 = 512 |       fp32 |   5872x | 85.359 us | 0.62% | 24.569G | 196.550 GB/s | 87.73% |
| 2^21 = 2097152 | 2^9 = 512 |     fp32x4 |   5904x | 84.729 us | 0.74% | 24.751G | 198.011 GB/s | 88.39% |
| 2^21 = 2097152 | 2^9 = 512 |        f16 |  10000x | 50.070 us | 1.32% | 41.885G | 167.539 GB/s | 74.78% |
| 2^21 = 2097152 | 2^9 = 512 |      f16x2 |  11168x | 44.827 us | 1.28% | 46.783G | 187.133 GB/s | 83.53% |
| 2^21 = 2097152 | 2^9 = 512 |      f16x8 |  11200x | 44.671 us | 1.28% | 46.947G | 187.788 GB/s | 83.82% |
| 2^21 = 2097152 | 2^9 = 512 | f16x8_pack |  11248x | 44.485 us | 1.27% | 47.142G | 188.570 GB/s | 84.17% |
*/

// NOTE: Compare to `elementwise add` benchmark. It's obvious that both `elementwise add` and
//       `elementwise exp` are memory bound. `add` do 3 IO per element, `exp` do 2 IO per element.
//       So the GPU time is 120us for `add` when the GPU time is 80us for `exp`.