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

// This is the batched version of our typical CuBLAS kernel. Instead of a single GEMM, it will perform as many as we tell it to. In this case, it is given by the argument "batch count".
// In other words, it means that we will command the kernel to perform "batch count" AxB matrix multiplications.
// It only uses a single stream and computes all the results in a single kernel. The logic is the same as the previous functions but with some extra details. This will be explained in the comments near the code.
// For more information about cublasGemmBatchedEx, visit: https://docs.nvidia.com/cuda/cublas/#cublasgemmbatchedex
void cublas_gemm_batched_fp32(
    float **A_batch, float **B_batch, float **C_batch,
    int m, int n, int k, int batchCount,
    cudaStream_t *streams,
    const char* mode,
    cublasComputeType_t computeType,
    bool warmup = false
) {

    //Instead of a single matrix like in the previous GEMMs, we will be feeding the CuBLAS function "batch count" GEMMs. For this we need pointers for each matrix, both for CPU and GPU matrices
    //Here we allocate CPU pointers
    std::vector<float*> d_A(batchCount), d_B(batchCount), d_C(batchCount); //here we are allocating an array of float pointers: Each pointer will point to a matrix
    std::vector<float*> h_Aptrs(batchCount), h_Bptrs(batchCount), h_Cptrs(batchCount); //here we are allocating an array of float pointers: Each pointer will point to a matrix

    // Here we allocate memory and copy each pair of matrices to the GPU. 
    for (int i = 0; i < batchCount; i++) {
      cudaMalloc(&d_A[i], sizeof(float)*m*k);
      cudaMalloc(&d_B[i], sizeof(float)*k*n);
      cudaMalloc(&d_C[i], sizeof(float)*m*n);
      cudaMemcpy(d_A[i], A_batch[i], sizeof(float)*m*k, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B[i], B_batch[i], sizeof(float)*k*n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_C[i], C_batch[i], sizeof(float)*m*n, cudaMemcpyHostToDevice);
      //here we store the pointers to where the matrices are stored in the GPU. This is needed later as an argument of the cublasGemmBatchedEx()
      h_Aptrs[i] = d_A[i]; 
      h_Bptrs[i] = d_B[i];
      h_Cptrs[i] = d_C[i];
    }
    // Here we simply allocate pointers of pointers to store where the matrices are stored in the memory. 
    float **d_Aptrs, **d_Bptrs, **d_Cptrs;
    cudaMalloc(&d_Aptrs, batchCount * sizeof(float*));
    cudaMalloc(&d_Bptrs, batchCount * sizeof(float*));
    cudaMalloc(&d_Cptrs, batchCount * sizeof(float*));

    // we use .data() because std::vector<float*> has that built-in function to retrieve its pointer 
    cudaMemcpy(d_Aptrs, h_Aptrs.data(), batchCount * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bptrs, h_Bptrs.data(), batchCount * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cptrs, h_Cptrs.data(), batchCount * sizeof(float*), cudaMemcpyHostToDevice);

    // Create the cublas handle as usual
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.f, beta = 0.f;

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * Here the time is recorded and printed.
   * The performance is calculated in GFLOPS.
   */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (!warmup) cudaEventRecord(start, 0);

    CUBLAS_CHECK(cublasGemmBatchedEx(
    handle,                 // cuBLAS context handle
    CUBLAS_OP_N,            
    CUBLAS_OP_N,           
    m,                    
    n,                    
    k,                     
    &alpha,                
    (void**)d_Aptrs,        // device array of pointers to A matrices
    CUDA_R_32F,             // data type of A
    m,                     
    (void**)d_Bptrs,        // device array of pointers to B matrices
    CUDA_R_32F,             // data type of B
    k,                      
    &beta,                  // scalar multiplier for existing C
    (void**)d_Cptrs,        // device array of pointers to C matrices
    CUDA_R_32F,             // data type of C
    m,                      
    batchCount,             // number of matrices in the batch
    computeType,            // computation precision/type
    CUBLAS_GEMM_DEFAULT     // algorithm selection (default)
));

    if (!warmup) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Batched cuBLAS (%s, %s, kernel only): %d batches, Time: %.3f ms, GFLOPS: %.3f\n",
               mode, cublasComputeTypeToString(computeType), batchCount, elapsedTime,
               batchCount * 2.0 * m * n * k / (elapsedTime * 1e-3) / 1e9);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Copy the results back from the GPU
    for (int i = 0; i < batchCount; i++) {
      cudaMemcpy(C_batch[i], d_C[i], sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    // Free the memory 
    for (int i = 0; i < batchCount; i++) {
      cudaFree(d_A[i]);
      cudaFree(d_B[i]);
      cudaFree(d_C[i]);
    }
    cudaFree(d_Aptrs);
    cudaFree(d_Bptrs);
    cudaFree(d_Cptrs);
    CUBLAS_CHECK(cublasDestroy(handle));
}

