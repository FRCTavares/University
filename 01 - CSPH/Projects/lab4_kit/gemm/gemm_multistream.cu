#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "gemm_singleFP32.cuh"


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
// cublas API error checking
#define CUBLAS_CHECK(err)                                                  \
  do {                                                                     \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                            \
    }                                                                      \
  } while (0)
#else
#define cudaCheckError(ans) ans
#define CUBLAS_CHECK(ans) ans
#endif

// Multi-stream custom FP32 kernel
// Here we will run one GEMM per stream using the cudaBlockKernel_FP32 from the previous lab. It will process "batch count" GEMMs, one per stream.
void gemm_fp32_multiStream(float **A_batch, float **B_batch, float **C_batch, int m, int n, int k, int batchCount, cudaStream_t *streams, bool warmup = false) {
  
  // Declare device pointers arrays, each vector will store the pointers to the matrices in the GPU
  std::vector<float*> devA(batchCount), devB(batchCount), devC(batchCount);

  // Allocate memory and copy the matrices to the GPU, each operation is assigned to its stream. Notice the usage of Async
  for (int b = 0; b < batchCount; b++) {
    cudaMallocAsync((void**)&devA[b], sizeof(float)*m*k, streams[b]); 
    cudaMallocAsync((void**)&devB[b], sizeof(float)*k*n, streams[b]);
    cudaMallocAsync((void**)&devC[b], sizeof(float)*m*n, streams[b]);
    cudaMemcpyAsync(devA[b], A_batch[b], sizeof(float)*m*k, cudaMemcpyHostToDevice, streams[b]);
    cudaMemcpyAsync(devB[b], B_batch[b], sizeof(float)*k*n, cudaMemcpyHostToDevice, streams[b]);
    cudaMemcpyAsync(devC[b], C_batch[b], sizeof(float)*m*n, cudaMemcpyHostToDevice, streams[b]);
  }
  // Wait for all streams to finish
  cudaDeviceSynchronize();
  int N = n;
  dim3 threadsPerBlock(LBLK, LBLK);
  dim3 blocks(updiv(n, LBLK), updiv(n, LBLK));
    // Timing only the kernel launches, as done on the other profilings
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
 // Iterate by each matrix pair, assigning its GEMM to a Stream
  for(int b = 0; b < batchCount; b++) {
    //For example, here we could use only 4 streams and where b is, we would use b % 4. This would lead to a different work distribution. 
    //The 0 in the arguments means we don't use Shared Memory
    cudaBlockKernel_FP32<<<blocks, threadsPerBlock, 0, streams[b]>>>(N, devA[b], devB[b], devC[b]);
  }
  // We don't need to sync after the kernel, only after the memory transfer so that we synchronize all the streams. 
  for(int b = 0; b < batchCount; b++) {
    cudaMemcpyAsync(C_batch[b], devC[b], sizeof(float)*m*n, cudaMemcpyDeviceToHost, streams[b]);
  }
  cudaDeviceSynchronize(); // this will synchronize all the streams

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (!warmup) {
    printf("Multi-stream FP32 (kernel only): %d batches, Time: %.3f ms, GFLOPS: %.3f\n",
           batchCount, elapsedTime, batchCount * 2.0 * m * n * k / (elapsedTime * 1e-3) / 1e9);
  }
  // Free the memory of each matrix
  for (int b = 0; b < batchCount; b++) {
    cudaFree(devA[b]);
    cudaFree(devB[b]);
    cudaFree(devC[b]);
  }
}

// Multi-stream cuBLAS GEMM (FP32/FP16/TF32)
// the logic for copying and allocating the data is similar to the gemm_fp32_multiStream version
// To use multiple streams here, you need to bind the stream to the calling handle, the stream is already created. So, it would be like
// cublasSetStream(handle, streams[i]);
// More information at https://docs.nvidia.com/cuda/cublas/#parallelism-with-streams

void cublas_gemm_multiStream(
    float **A_batch, float **B_batch, float **C_batch,
    int m, int n, int k, int batchCount,
    cudaStream_t *streams,
    const char* mode,
    cublasComputeType_t computeType,
    bool warmup = false
) {
    // Declare device pointers arrays, each vector will store the pointers to the matrices in the GPU
    std::vector<float*> d_A(batchCount), d_B(batchCount), d_C(batchCount);

    // Create CuBLAS handle as usual 
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Allocate memory and copy the matrices to the GPU, each operation is assigned to its stream. Notice the usage of Async
    for (int i = 0; i < batchCount; i++) {
        cudaMallocAsync(&d_A[i], sizeof(float)*m*k, streams[i]);
        cudaMallocAsync(&d_B[i], sizeof(float)*k*n, streams[i]);
        cudaMallocAsync(&d_C[i], sizeof(float)*m*n, streams[i]);

        cudaMemcpyAsync(d_A[i], A_batch[i], sizeof(float)*m*k, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B[i], B_batch[i], sizeof(float)*k*n, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_C[i], C_batch[i], sizeof(float)*m*n, cudaMemcpyHostToDevice, streams[i]);
    }
    // Wait for all streams to finish
    cudaDeviceSynchronize();

    const float alpha = 1.f, beta = 0.f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (!warmup) cudaEventRecord(start, 0);

     // Iterate by each matrix pair, assigning its GEMM to a Stream
    for (int i = 0; i < batchCount; i++) {
        CUBLAS_CHECK(cublasSetStream(handle, streams[i])); // bind the handle to a stream
        // Here the parameters are the same as in our single stream version
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_A[i], CUDA_R_32F, m,
            d_B[i], CUDA_R_32F, k,
            &beta,
            d_C[i], CUDA_R_32F, m,
            computeType, CUBLAS_GEMM_DEFAULT
        ));
    }
    
    cudaDeviceSynchronize();

    if (!warmup) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Multi-stream cuBLAS (%s, %s, kernel only): %d batches, Time: %.3f ms, GFLOPS: %.3f\n",
               mode, cublasComputeTypeToString(computeType), batchCount, elapsedTime,
               batchCount * 2.0 * m * n * k / (elapsedTime * 1e-3) / 1e9);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back asynchronously
    for (int i = 0; i < batchCount; i++) {
        cudaMemcpyAsync(C_batch[i], d_C[i], sizeof(float)*m*n, cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize(); // synchronize all the streams
    // free the memory
    for (int i = 0; i < batchCount; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    CUBLAS_CHECK(cublasDestroy(handle));
}


