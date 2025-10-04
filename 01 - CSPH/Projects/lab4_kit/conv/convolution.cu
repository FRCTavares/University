/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans)                \
  {                                        \
    cudaAssert((ans), __FILE__, __LINE__); \
  }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n",
            cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#else
#define cudaCheckError(ans) ans
#endif

// Macro for accessing filter array
#define id_filter(f, c, j, i, num_filters, num_channels, filter_size) ((f) * (num_channels) * (filter_size) * (filter_size) + (c) * (filter_size) * (filter_size) + (j) * (filter_size) + (i))
// Macro for accessing image array
#define id_img(c, j, i, num_channels, width) ((j) * (num_channels) * (width) + (i) * (num_channels) + (c))

void printCudaInfo() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}

/**
 * @brief CUDA kernel for performing convolution on input data with given filters.
 *
 * @param data Pointer to the input data array.
 * @param output Pointer to the output data array.
 * @param filters Pointer to the filters array.
 * @param num_filters Number of filters.
 * @param num_channels Number of channels in the input data.
 * @param filter_size Size of the filter (square).
 * @param input_h Height of the input data.
 * @param input_w Width of the input data.
 * @param output_h Height of the output data.
 * @param output_w Width of the output data.
 *
 * @details This kernel performs a convolution operation on the input data using the provided filters.
 */
__global__ void convolution_kernel(float *data, float *output, float *filters, int num_filters, int num_channels, int filter_size, int input_h, int input_w, int output_h, int output_w) {
  // TODO:
  // Here you should write your kernel to perform convolution using CUDA cores

  return;
}

/**
 * @brief Performs convolution on input data with given filters using CUDA cores.
 *
 * @param data Pointer to the input data array.
 * @param output Pointer to the output data array.
 * @param filters Pointer to the filters array.
 * @param num_filters Number of filters.
 * @param num_channels Number of channels in the input data.
 * @param filter_size Size of the filter (square).
 * @param input_h Height of the input data.
 * @param input_w Width of the input data.
 * @param output_h Height of the output data.
 * @param output_w Width of the output data.
 * @param warmup Flag to indicate whether to print performance metrics.
 *
 * @details This function performs a convolution operation on the input data using the provided filters.
 */
void convolution_gpu_cuda_cores(float *data, float *output, float *filters, int num_filters, int num_channels, int filter_size, int input_h, int input_w, int output_h, int output_w, bool warmup = false) {
  // Allocate device memory
  float *device_data;
  float *device_filters;
  float *device_output;

  cudaCheckError(cudaMalloc(&device_data, input_h * input_w * num_channels * sizeof(float)));
  cudaCheckError(cudaMalloc(&device_filters, num_filters * num_channels * filter_size * filter_size * sizeof(float)));
  cudaCheckError(cudaMalloc(&device_output, output_h * output_w * num_filters * sizeof(float)));

  // Copy data to device
  cudaCheckError(cudaMemcpy(device_data, data, input_h * input_w * num_channels * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(device_filters, filters, num_filters * num_channels * filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice));

  // TODO:
  // Here you should configure your kernel launch parameters
  int threadsPerBlock = 1;
  int numBlocks = 1;

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * The kernel is executed between the start and stop events.
   */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Launch kernel
  convolution_kernel<<<numBlocks, threadsPerBlock>>>(device_data, device_output, device_filters, num_filters, num_channels, filter_size, input_h, input_w, output_h, output_w);

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * Here the time is recorded and printed.
   * The performance is calculated in GFLOPS.
   */
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (!warmup) {
    printf("Kernel Time (CUDA Cores): %f ms\n", elapsedTime);
    printf("GFLOPS (CUDA Cores): %f\n", 2.0 * filter_size * filter_size * num_channels * num_filters * output_w * output_h / (elapsedTime * 1e-3) / 1e9);
  }

  // Copy output to host
  cudaCheckError(cudaMemcpy(output, device_output, output_h * output_w * num_filters * sizeof(float), cudaMemcpyDeviceToHost));

  // Free device memory
  cudaCheckError(cudaFree(device_data));
  cudaCheckError(cudaFree(device_filters));
  cudaCheckError(cudaFree(device_output));
}