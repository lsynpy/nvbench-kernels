#include <cuda_fp16.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(a) (reinterpret_cast<float4 *>(&(a))[0])
#define HALF2(a) (reinterpret_cast<half2 *>(&(a))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// y = max(0, x)
__global__ void relu_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = fmax(0.0f, x[idx]);
  }
}

__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if ((idx + 0) < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmax(0.0f, reg_x.x);
    reg_y.y = fmax(0.0f, reg_x.y);
    reg_y.z = fmax(0.0f, reg_x.z);
    reg_y.w = fmax(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}

__global__ void relu_f16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = __hmax(__float2half(0.0f), x[idx]);
  }
}

__global__ void relu_f16x2_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
    reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

// unpack f16x8
__global__ void relu_f16x8_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const half f = __float2half(1.0f);

  half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);

  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

  reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
  reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
  reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
  reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
  reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
  reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
  reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
  reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);
  if ((idx + 0) < N) {
    HALF2(y[idx + 0]) = reg_y_0;
  }
  if ((idx + 2) < N) {
    HALF2(y[idx + 2]) = reg_y_1;
  }
  if ((idx + 4) < N) {
    HALF2(y[idx + 4]) = reg_y_2;
  }
  if ((idx + 6) < N) {
    HALF2(y[idx + 6]) = reg_y_3;
  }
}

// pack f16x8
__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

#pragma unroll

  for (int i = 0; i < 8; i += 2) {
    HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
  }
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

void relu_bench(nvbench::state &state) {
  const auto N = state.get_int64("N");
  const auto block_size = state.get_int64("BlockSize");
  const auto variant = state.get_string("Variants");

  const bool is_fp16 = (variant.find("f16") != std::string::npos);

  state.add_element_count(N * 3, "NumFLOPs");

  if (is_fp16) {
    thrust::device_vector<half> a(N);
    thrust::device_vector<half> b(N);

    thrust::fill(a.begin(), a.end(), __float2half(1.0f));

    state.add_global_memory_reads<nvbench::int16_t>(N);
    state.add_global_memory_writes<nvbench::int16_t>(N);

    state.exec([&](nvbench::launch &launch) {
      if (variant == "f16") {
        const auto grid_size = CEIL(N, block_size);
        relu_f16_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N
        );
      } else if (variant == "f16x2") {
        const auto grid_size = CEIL(CEIL(N, 2), block_size);
        relu_f16x2_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N
        );
      } else if (variant == "f16x8") {
        const auto grid_size = CEIL(CEIL(N, 8), block_size);
        relu_f16x8_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N
        );
      } else if (variant == "f16x8_pack") {
        const auto grid_size = CEIL(CEIL(N, 8), block_size);
        relu_f16x8_pack_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N
        );
      }
    });
  } else {
    thrust::device_vector<float> a(N);
    thrust::device_vector<float> b(N);

    thrust::fill(a.begin(), a.end(), 1.0f);

    state.add_global_memory_reads<nvbench::int32_t>(N);
    state.add_global_memory_writes<nvbench::int32_t>(N);

    state.exec([&](nvbench::launch &launch) {
      if (variant == "f32") {
        const auto grid_size = CEIL(N, block_size);
        relu_f32_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N
        );
      } else if (variant == "f32x4") {
        const auto grid_size = CEIL(CEIL(N, 4), block_size);
        relu_f32x4_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N
        );
      }
    });
  }
}

NVBENCH_BENCH(relu_bench)
  .add_int64_power_of_two_axis("N", nvbench::range(21, 21, 1))
  .add_int64_power_of_two_axis("BlockSize", nvbench::range(9, 9, 1))
  .add_string_axis("Variants", {"f32", "f32x4", "f16", "f16x2", "f16x8", "f16x8_pack"});

/*
## relu_bench

### [0] NVIDIA GeForce RTX 3050

|       N        | BlockSize |  Variants  | Samples | GPU Time  | Noise |  Elem/s  | GlobalMem BW | BWUtil |
|----------------|-----------|------------|---------|-----------|-------|----------|--------------|--------|
| 2^21 = 2097152 | 2^9 = 512 |        f32 |   5888x | 85.094 us | 0.86% |  73.935G | 197.161 GB/s | 88.01% |
| 2^21 = 2097152 | 2^9 = 512 |      f32x4 |   5904x | 84.704 us | 0.84% |  74.276G | 198.069 GB/s | 88.41% |
| 2^21 = 2097152 | 2^9 = 512 |        f16 |  10192x | 49.108 us | 1.59% | 128.116G | 170.821 GB/s | 76.25% |
| 2^21 = 2097152 | 2^9 = 512 |      f16x2 |  11216x | 44.621 us | 1.59% | 140.998G | 187.997 GB/s | 83.92% |
| 2^21 = 2097152 | 2^9 = 512 |      f16x8 |  11168x | 44.822 us | 1.51% | 140.365G | 187.154 GB/s | 83.54% |
| 2^21 = 2097152 | 2^9 = 512 | f16x8_pack |  11248x | 44.456 us | 1.54% | 141.522G | 188.696 GB/s | 84.23% |

*/