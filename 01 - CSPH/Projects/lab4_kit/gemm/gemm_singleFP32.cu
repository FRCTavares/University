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

const char* cublasComputeTypeToString(cublasComputeType_t computeType) {
    switch (computeType) {
        case CUBLAS_COMPUTE_16F:
            return "FP16";
        case CUBLAS_COMPUTE_32F:
            return "FP32";
        case CUBLAS_COMPUTE_32F_FAST_16F:
            return "FP16 Tensor Cores";
        case CUBLAS_COMPUTE_32F_FAST_TF32:
            return "TF32 Tensor Cores";
        default:
            return "Unknown";
    }
}



/*The code from the previous lab*/
__global__ void cudaBlockKernel_FP32(int N, float *dmatA, float *dmatB, float *dmatC) {
    // Assume that thread block contains submatrix of size LBLK x LBLK
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int bi = threadIdx.y;
    int bj = threadIdx.x;

    float sum = 0.0; // Accumulate result for C[i][j]

    // Shared space for two submatrices of A and B
    __shared__ float subA[LBLK*LBLK];
    __shared__ float subB[LBLK*LBLK];

    // Loop over k to compute product of all submatrices A[i][k] and B[k][j]
    for (int k = 0; k < N; k+= LBLK) {
    	// Grab the two submatrices
    	if (i < N && k+bj < N)
    	    subA[RM(bi,bj,LBLK)] = dmatA[RM(i,k+bj,N)];
    	else
    	    subA[RM(bi,bj,LBLK)] = 0.0;

    	if (j < N && k+bi < N)
    	    subB[RM(bi,bj,LBLK)] = dmatB[RM(k+bi,j,N)];
    	else
    	    subB[RM(bi,bj,LBLK)] = 0.0;

    	// Wait until entire block gets filled
    	__syncthreads();

    	// Generate contribution to C[i][j] of these submatrices
    	for (int bk = 0; bk < LBLK; bk++)
    	    sum += subA[RM(bi,bk,LBLK)] * subB[RM(bk,bj,LBLK)];

    	// Wait until all products computed
    	__syncthreads();
    }
    if (i < N && j < N)
	   dmatC[RM(i,j,N)] = sum;
}
