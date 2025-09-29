#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CycleTimer.h"

#include "matrix.h"

// Integer division, rounding up
static inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

/* Transpose matrix */
__global__ void
cudaMatTransposeKernel(int N, const float  *dmatS, float *dmatD) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N)
    	return;
    dmatD[CM(i,j,N)] = dmatS[RM(i,j,N)];
}

__global__ void
cudaSimpleKernel(int N, float*  dmatA, float* dmatB, float * dmatC) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N)
    	return;
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
    	sum += dmatA[RM(i,k,N)] * dmatB[RM(k,j,N)];
    }
    dmatC[RM(i,j,N)] = sum;
}

__global__ void
cudaSimpleKernelInverted(int N, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N)
    return;
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
    sum += dmatA[RM(i,k,N)] * dmatB[RM(k,j,N)];
    }
    dmatC[RM(i,j,N)] = sum;
}

__global__ void
cudaTransposedKernel(int N, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N)
    	return;
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
    	sum += dmatA[RM(i,k,N)] * dmatB[CM(k,j,N)];
    }
    dmatC[RM(i,j,N)] = sum;
}

__global__ void
cudaBlockKernel(int N, float *dmatA, float *dmatB, float *dmatC) {
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


/* Preallocated blocks */
static int allocN = -1;
static float *aDevData = NULL;
static float *bDevData = NULL;
static float *tDevData = NULL;
static float *gDevData = NULL;
static float *sDevData = NULL;
static float *tHostData = NULL;
static float *gHostData = NULL;

void cudaSetup(int N, float *aData, float *bData, float *gData) {
    if (allocN == N)
    	return;
    if (allocN > 0) {
    	cudaFree(sDevData);
    	cudaFree(aDevData);
    	cudaFree(bDevData);
    	cudaFree(tDevData);
    	cudaFree(gDevData);
    }
    if (N > 0) {
    	cudaMalloc((void **) &aDevData, N*N * sizeof(float));
    	cudaMalloc((void **) &bDevData, N*N * sizeof(float));
    	cudaMalloc((void **) &tDevData, N*N * sizeof(float));
    	cudaMalloc((void **) &sDevData, N*N * sizeof(float));
    	tHostData = (float *) calloc(N*N, sizeof(float));
    }
    gHostData = gData;
    cudaMemcpy(aDevData, aData, N*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bDevData, bData, N*N * sizeof(float), cudaMemcpyHostToDevice);
    allocN = N;
}

// Get scratch for matrix
static float *cudaScratchMatrix(int N) {
    if (allocN != N) {
    	setup(N);
    }
    return sDevData;
}

void cudaMultMatrixSimple(int N, float *dmatA, float *dmatB, float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    cudaSimpleKernel<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}

void cudaMultMatrixSimpleInverted(int N, float *dmatA, float *dmatB, float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    cudaSimpleKernelInverted<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}

void cudaMultMatrixTransposed(int N, float *dmatA, float *dmatB, float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    float *tranB = cudaScratchMatrix(N);
    cudaMatTransposeKernel<<<blocks, threadsPerBlock>>>(N, dmatB, tranB);
    cudaTransposedKernel<<<blocks, threadsPerBlock>>>(N, dmatA, tranB, dmatC);
}

void cudaMultMatrixBlocked(int N, float *dmatA, float *dmatB, float *dmatC)
{
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    cudaBlockKernel<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}

void cuBLASkernel(int N, float *dmatA, float *dmatB, float *dmatC) 
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1, beta = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dmatB, N, dmatA, N, &beta, dmatC, N);
    cublasDestroy(handle);
}


static int cudaRunMM(int N, mmul_t method) {
    switch (method) {
        case MMUL_CUDA_REFERENCE:
        	cudaMultMatrixSimple(N, aDevData, bDevData, tDevData);
    	break;
        case MMUL_CUDA_INVERTED_REFERENCE:
            cudaMultMatrixSimpleInverted(N, aDevData, bDevData, tDevData);
        break;
        case MMUL_CUDA_TRANSPOSE:
        	cudaMultMatrixTransposed(N, aDevData, bDevData, tDevData);
    	break;
        case MMUL_CUDA_BLK:
        	cudaMultMatrixBlocked(N, aDevData, bDevData, tDevData);
    	break;
        case MMUL_CUBLAS:
        	cuBLASkernel(N, aDevData, bDevData, tDevData);
    	break;
        default:
        	fprintf(stderr, "Haven't implemented method yet\n");
    	return 0;
    }
    return 1;
}

double cudaBenchMM(int N, mmul_t method) {
    // Should already have done the setup
    if (allocN != N) {
    	setup(N);
    }
    if (!cudaRunMM(N, method))
    	return 1000.0;
    cudaMemcpy(tHostData, tDevData, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    if (checkMatrix(N, tHostData, gHostData) > 0)
    	return 1000.0;
    /* Now do the real benchmarking */
    long ops = (long) 2 * N * N * N;
    long runs = (targetOps+ops-1)/ops;
    double startTime = CycleTimer::currentSeconds();
    for (long r = 0; r < runs; r++)
    	cudaRunMM(N, method);
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double ms = (endTime - startTime) * 1000.0;
    double gflops = (long) (runs*ops)/ms * 1e-6;
    //fprintf(stderr, "%ld runs, %ld ops/run, %.2f ms, %.3f GFlops\n", runs, ops, ms, gflops);
    return gflops;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
