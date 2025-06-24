# See apt.kitware.com for updated CMake versions on Ubuntu.
cmake_minimum_required(VERSION 3.30.4)

project(NVBenchDemo CUDA)

include(cmake/CPM.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")

file(GLOB_RECURSE KERNEL_SOURCES "kernels/*.cu")

add_custom_target(mybench.all)

set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)

foreach(kernel_src IN LISTS KERNEL_SOURCES)
  get_filename_component(kernel_name ${kernel_src} NAME_WE)
  set(target_name "${kernel_name}.out")

  add_executable(${target_name} ${kernel_src})
  target_link_libraries(${target_name} PRIVATE nvbench::main)

  set_target_properties(${target_name} PROPERTIES
    COMPILE_FEATURES "cuda_std_${CUDA_STANDARD}"
  )

  add_dependencies(mybench.all ${target_name})

  add_test(
    NAME ${target_name}
    COMMAND $<TARGET_FILE:${target_name}>
  )
endforeach()
