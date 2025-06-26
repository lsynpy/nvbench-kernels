#include <nvbench/nvbench.cuh>
#include <nvbench/test_kernels.cuh>
#include <thrust/device_vector.h>

void throughput_bench(nvbench::state &state) {
  // Allocate input data:
  const std::size_t num_values = 64 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<nvbench::int32_t>(num_values, "DataSize");
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  state.exec([&input, &output, num_values](nvbench::launch &launch) {
    (void)num_values; // clang thinks this is unused...
    nvbench::copy_kernel<<<256, 256, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()), num_values);
  });
}
NVBENCH_BENCH(throughput_bench);