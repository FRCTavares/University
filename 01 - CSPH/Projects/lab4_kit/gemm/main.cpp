#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <cmath>
#include <cstring>
#include <string>

#include <cuda.h>
#include <cublas_v2.h>

#include "../utils/CycleTimer.h"

// define col major access
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

#define RM(r, c, width) ((r) * (width) + (c))
/* Find element based on row-major ordering for 3 dimensions */
#define RM3(r, c, d, width) RM(RM(r, c, width), d, width)

/* Find element based on column-major ordering */
#define CM(r, c, height) ((c) * (height) + (r))

void printCudaInfo();

// Function declarations

void gemm_fp32(float *A_RM, float *B_RM, float *C_RM, int m, int n, int k, bool warmup = false);

void cublas_gemm_fp32(float *A_CM, float *B_CM, float *C_CM, int m, int n, int k, bool warmup = false);

void cublas_gemm_fp16(float *A_CM, float *B_CM, float *C_CM, int m, int n, int k, bool warmup = false);

void cublas_gemm_tf32(float *A_CM, float *B_CM, float *C_CM, int m, int n, int k, bool warmup = false);

void gemm_cpu(float *A_CM, float *B_CM, double *C_CM, double *C_RM, int m, int n, int k);

double euclidean_distance(double *A_CM, float *B_CM, int n);


// Code to perform Multi GEMMs
// This will perform multiple GEMMs using two type of kernels:
// - batched: will only use one stream and do all the GEMMs in a single call
// - multiStream: Will perform multiple GEMMs using multiple Streams
void multiGEMM(int m, int n, int k, int numStreams) {
    // Number of batches = numStreams
    int batchCount = numStreams; // This will be the number of Matrix Multiplications that we will perform
    cudaStream_t *streams = new cudaStream_t[numStreams]; // Allocate stream pointer
    // Create Streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate batches of matrices
    float **A_batch = new float*[batchCount];
    float **B_batch = new float*[batchCount];
    float **C_batch = new float*[batchCount];
    // Allocate each matrix
    for (int i = 0; i < batchCount; i++) {
        A_batch[i] = new float[m * k];
        B_batch[i] = new float[k * n];
        C_batch[i] = new float[m * n];
        // Initialize with simple values for demonstration
        for (int j = 0; j < m * k; j++) A_batch[i][j] = j % k;
        for (int j = 0; j < k * n; j++) B_batch[i][j] = j % n;
    }

    // Externals
    extern void cublas_gemm_batched_fp32(float **A_batch, float **B_batch, float **C_batch,
                                         int m, int n, int k, int batchCount,
                                         cudaStream_t *streams, const char* mode,
                                         cublasComputeType_t computeType, bool warmup);

    extern void gemm_fp32_multiStream(float **A_batch, float **B_batch, float **C_batch,
                                      int m, int n, int k, int batchCount,
                                      cudaStream_t *streams, bool warmup);

    extern void gemm_fp16_multiStream(float **A_batch, float **B_batch, float **C_batch,
                                      int m, int n, int k, int batchCount,
                                      cudaStream_t *streams, bool warmup);

    extern void cublas_gemm_multiStream(float **A_batch, float **B_batch, float **C_batch,
                                        int m, int n, int k, int batchCount,
                                        cudaStream_t *streams,
                                        const char* mode,
                                        cublasComputeType_t computeType,
                                        bool warmup);
    // Here we will warmup all the funtions that use multiple streams and batching
    // Warmup for each type
    cublas_gemm_batched_fp32(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "FP32", CUBLAS_COMPUTE_32F, true);
    cublas_gemm_batched_fp32(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "FP16", CUBLAS_COMPUTE_32F_FAST_16F, true);
    cublas_gemm_batched_fp32(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "TF32", CUBLAS_COMPUTE_32F_FAST_TF32, true);
    cublas_gemm_multiStream(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "FP32", CUBLAS_COMPUTE_32F, true);
    cublas_gemm_multiStream(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "FP16", CUBLAS_COMPUTE_32F_FAST_16F, true);
    cublas_gemm_multiStream(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "TF32", CUBLAS_COMPUTE_32F_FAST_TF32, true);
    gemm_fp32_multiStream(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, true);

    // Timed runs
    // Here we will profile our programs
    cublas_gemm_batched_fp32(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "FP32", CUBLAS_COMPUTE_32F, false);
    cublas_gemm_batched_fp32(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "FP16", CUBLAS_COMPUTE_32F_FAST_16F, false);
    cublas_gemm_batched_fp32(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "TF32", CUBLAS_COMPUTE_32F_FAST_TF32, false);
    cublas_gemm_multiStream(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "FP32", CUBLAS_COMPUTE_32F, false);
    cublas_gemm_multiStream(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "FP16", CUBLAS_COMPUTE_32F_FAST_16F, false);
    cublas_gemm_multiStream(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, "TF32", CUBLAS_COMPUTE_32F_FAST_TF32, false);
    gemm_fp32_multiStream(A_batch, B_batch, C_batch, m, n, k, batchCount, streams, false);

    // Cleanup
    for (int i = 0; i < batchCount; i++) {
        delete[] A_batch[i];
        delete[] B_batch[i];
        delete[] C_batch[i];
    }
    delete[] A_batch;
    delete[] B_batch;
    delete[] C_batch;
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

int main(int argc, char **argv) {
  int m = 128;
  int n = 128;
  int k = 128;
  int numStreams = 1; // Default

  printCudaInfo();

  // parse command line arguments
  int o;
  while ((o = getopt(argc, argv, "m:n:k:a:s:h")) != -1) switch (o) {
      case 'a':
        m = n = k = atoi(optarg);
        break;
      case 'm':
        m = atoi(optarg);
        break;
      case 'n':
        n = atoi(optarg);
        break;
      case 'k':
        k = atoi(optarg);
        break;
      case 's':
        numStreams = atoi(optarg);
        break;
      case 'h':
        fprintf(stdout,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=128]\n\t-n \t N "
                "dimension [int] [default=128]\n\t-k \t K dimension [int] "
                "[default=128]\n\t-a \t All "
                "dimensions [int]\n\n",
                argv[0]);
        exit(EXIT_SUCCESS);
      default:
        fprintf(stderr,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=128]\n\t-n \t N "
                "dimension [int] [default=128]\n\t-k \t K dimension [int] "
                "[default=128]\n\t-a \t All "
                "dimensions [int]\n\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  printf("Performing GEMM with dimensions: %d x %d x %d\n\n", m, n, k);


  if (numStreams > 1) {
    multiGEMM(m, n, k, numStreams);
    return 0;
  }

  // allocate memory for matrices
  float *A_CM = new float[m * k];
  float *B_CM = new float[k * n];
  float *A_RM = new float[m * k];
  float *B_RM = new float[k * n];
  double *C_CM = new double[m * n];
  double *C_RM = new double[m * n];
  float *C_fp32 = new float[m * n];
  float *C_fp32_your_implementation = new float[m * n];
  float *C_fp16 = new float[m * n];
  float *C_tf32 = new float[m * n];

  // initialize A_CM and B_CM with random values, assumed to be in col_major format
  #pragma omp parallel for
  for (int i = 0; i < m ; i++){
    for(int j = 0; j < k; j++){
      A_CM[CM(i,j,m)] =  static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 10));
      A_RM[RM(i,j,k)] = A_CM[CM(i,j,m)]; //store our A in both Row Major and Column Major
    }
  }
  #pragma omp parallel for
  for(int i = 0; i < k ; i++){
    for(int j = 0; j < n; j++){
      B_CM[CM(i,j,k)] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 10));
      B_RM[RM(i,j,n)] = B_CM[CM(i,j,k)]; //store our B in both Row Major and Column Major
    }
  
     
  }
  // CPU GEMM
  gemm_cpu(A_CM, B_CM, C_CM, C_RM, m, n, k);

  // warm up GPU to get more accurate timings
  cublas_gemm_fp32(A_RM, B_RM, C_fp32, m, n, k, true);
  cublas_gemm_fp32(A_RM, B_RM, C_fp32, m, n, k, true);
  cublas_gemm_fp32(A_RM, B_RM, C_fp32, m, n, k, true);

  // CUBLAS GEMM for FP32, FP16, and TF32
  double startTime = CycleTimer::currentSeconds();
  cublas_gemm_fp32(A_CM, B_CM, C_fp32, m, n, k);
  double endTime = CycleTimer::currentSeconds();
  printf("GPU time (FP32): %.3f ms\n\n", 1000.f * (endTime - startTime));


  // warm up kernel for FP16
  cublas_gemm_fp16(A_CM, B_CM, C_fp16, m, n, k, true);

  // Measure performance of FP16
  startTime = CycleTimer::currentSeconds();
  cublas_gemm_fp16(A_CM, B_CM, C_fp16, m, n, k);
  endTime = CycleTimer::currentSeconds();
  printf("GPU time (FP16): %.3f ms\n\n", 1000.f * (endTime - startTime));

  // warm up kernel for TF32
  cublas_gemm_tf32(A_CM, B_CM, C_tf32, m, n, k, true);

  // Measure performance of TF32
  startTime = CycleTimer::currentSeconds();
  cublas_gemm_tf32(A_CM, B_CM, C_tf32, m, n, k);
  endTime = CycleTimer::currentSeconds();
  printf("GPU time (TF32): %.3f ms\n\n", 1000.f * (endTime - startTime));


  // warm up kernel for your implementation
  gemm_fp32(A_RM, B_RM, C_fp32_your_implementation, m, n, k, true);
  // Measure performance of your implementation
  startTime = CycleTimer::currentSeconds();
  gemm_fp32(A_RM, B_RM, C_fp32_your_implementation, m, n, k);
  endTime = CycleTimer::currentSeconds();
  printf("GPU time (CUDA Core FP32): %.3f ms\n\n", 1000.f * (endTime - startTime));

  // compare results with CPU GEMM
  printf("Euclidean distance between CPU and FP32 (Your implementation): %.9f\n", euclidean_distance(C_RM, C_fp32_your_implementation, m * n));
  printf("Euclidean distance between CPU and FP32 (CuBLAS): %.9f\n", euclidean_distance(C_CM, C_fp32, m * n));
  printf("Euclidean distance between CPU and FP16 (CuBLAS): %.9f\n", euclidean_distance(C_CM, C_fp16, m * n));
  printf("Euclidean distance between CPU and TF32 (CuBLAS): %.9f\n", euclidean_distance(C_CM, C_tf32, m * n));

  // free memory
  delete[] A_CM;
  delete[] B_CM;
  delete[] C_CM;
  delete[] C_fp32;
  delete[] C_tf32;

  return 0;
}

/**
 * @brief Performs General Matrix Multiply (GEMM) on CPU.
 *
 * This function computes the product of two matrices A_CM and B_CM, and stores the result in matrix C_CM.
 * The matrices are stored in column-major order.
 *
 * @param A_CM Pointer to the first input matrix (m x k).
 * @param B_CM Pointer to the second input matrix (k x n).
 * @param C_CM Pointer to the output matrix (m x n).
 * @param m M dimension.
 * @param n N dimension.
 * @param k K dimension.
 */
void gemm_cpu(float *A_CM, float *B_CM, double *C_CM, double *C_RM, int m, int n, int k) {
  #pragma omp parallel for
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      double sum = 0;
      for (int l = 0; l < k; l++) sum += A_CM[IDX2C(i, l, m)] * B_CM[IDX2C(l, j, k)];
      C_CM[IDX2C(i, j, m)] = sum;
      C_RM[RM(i,j,n)] = sum;
    }
}

/**
 * @brief Computes the Euclidean distance between two matrices.
 *
 * This function calculates the Euclidean distance between two matrices A_CM and B_CM
 * of length n=w*h. The Euclidean distance is defined as the square root of the sum
 * of the squared differences between corresponding elements of the matrices.
 *
 * @param A_CM Pointer to the first matrix (array of doubles) - CPU result.
 * @param B_CM Pointer to the second matrix (array of floats) - corresponding GPU result.
 * @param n The number of elements in each matrix.
 * @return The Euclidean distance between the two matrices.
 */
double euclidean_distance(double *A_CM, float *B_CM, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++) sum += pow((A_CM[i] - B_CM[i]), 2);
  return sqrt(sum);
}