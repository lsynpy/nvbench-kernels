#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// --- FP32 kernels ---
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

// --- FP16 kernels ---
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}

__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    HALF2(c[idx]) = reg_c;
  }
}

__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 6 < N) {
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);
    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);

    half2 reg_c_0 = __hadd2(reg_a_0, reg_b_0);
    half2 reg_c_1 = __hadd2(reg_a_1, reg_b_1);
    half2 reg_c_2 = __hadd2(reg_a_2, reg_b_2);
    half2 reg_c_3 = __hadd2(reg_a_3, reg_b_3);

    HALF2(c[idx + 0]) = reg_c_0;
    HALF2(c[idx + 2]) = reg_c_1;
    HALF2(c[idx + 4]) = reg_c_2;
    HALF2(c[idx + 6]) = reg_c_3;
  } else {
    for (int i = 0; i < 8 && (idx + i) < N; ++i) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
    }
  }
}

__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_a[8], pack_b[8], pack_c[8];

  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
  }

  if ((idx + 7) < N) {
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  } else {
    for (int i = 0; i < 8 && (idx + i) < N; ++i) {
      c[idx + i] = pack_c[i];
    }
  }
}

void elementwise_add_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto block_size = state.get_int64("BlockSize");
  const auto variant = state.get_string("Variants");

  const bool is_fp16 = (variant.find("f16") != std::string::npos);

  if (is_fp16) {
    thrust::device_vector<half> a(N);
    thrust::device_vector<half> b(N);
    thrust::device_vector<half> c(N);

    thrust::fill(a.begin(), a.end(), __float2half(1.0f));
    thrust::fill(b.begin(), b.end(), __float2half(2.0f));

    state.add_element_count(N, "NumFLOPs");
    state.add_global_memory_reads<nvbench::int16_t>(N * 2);
    state.add_global_memory_writes<nvbench::int16_t>(N);

    state.exec([&](nvbench::launch &launch) {
      if (variant == "f16") {
        const auto grid_size = CEIL(N, block_size);
        elementwise_add_f16_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()),
          thrust::raw_pointer_cast(b.data()),
          thrust::raw_pointer_cast(c.data()),
          N
        );
      } else if (variant == "f16x2") {
        const auto grid_size = CEIL(CEIL(N, 2), block_size);
        elementwise_add_f16x2_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()),
          thrust::raw_pointer_cast(b.data()),
          thrust::raw_pointer_cast(c.data()),
          N
        );
      } else if (variant == "f16x8") {
        const auto grid_size = CEIL(CEIL(N, 8), block_size);
        elementwise_add_f16x8_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()),
          thrust::raw_pointer_cast(b.data()),
          thrust::raw_pointer_cast(c.data()),
          N
        );
      } else if (variant == "f16x8_pack") {
        const auto grid_size = CEIL(CEIL(N, 8), block_size);
        elementwise_add_f16x8_pack_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()),
          thrust::raw_pointer_cast(b.data()),
          thrust::raw_pointer_cast(c.data()),
          N
        );
      }
    });
  } else {
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
}

NVBENCH_BENCH(elementwise_add_bench)
  .add_int64_power_of_two_axis("N", nvbench::range(21, 21, 1))
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(9, 9, 1))
  .add_string_axis("Variants", {"fp32", "fp32x4", "f16", "f16x2", "f16x8", "f16x8_pack"});