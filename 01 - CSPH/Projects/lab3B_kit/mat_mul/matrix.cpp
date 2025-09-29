#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "matrix.h"
#include "CycleTimer.h"

#include "mkl.h"

#define MATRIX 1
/* Error Tolerance */
float errTolerance = 0.001;
/* Limit to errors before giving up */
int errLimit = 10;
/* Optimum number of operations for measurements */
/* Want around 100ms for 1GF */
long targetOps = 100000000;

/* Preallocated blocks */
static int allocN = -1;
static float *aData = NULL;
static float *bData = NULL;
static float *tData = NULL;
static float *gData = NULL;
static float *sData = NULL;

/* Forward pointers */
void multMatrixSimple(int N, float *matA, float *matB, float *matC);

static float rdata() {
    return 1.0 + ((float) rand()/RAND_MAX) * 9.0;
}

// static float ddata(int i, int j) {
//     return i == j ? 1.0 : 0.0;
// }

static float adata(int i, int j, int N) {
    return rdata();
    //    return ddata(i, j);
}

static float bdata(int i, int j, int N) {
    return rdata();
    //    return ddata(i, j);
}


void setup(int N) {
    if (allocN == N)
		return;
    if (allocN > 0) {
		free(sData);
		free(aData);
		free(bData);
		free(tData);
		free(gData);
    }
    if (N > 0) {
		aData = (float *) calloc(N*N, sizeof(float));
		bData = (float *) calloc(N*N, sizeof(float));
		tData = (float *) calloc(N*N, sizeof(float));
		gData = (float *) calloc(N*N, sizeof(float));
		sData = (float *) calloc(N*N, sizeof(float));
		/* Generate random data for A & B */
		for (int i = 0; i < N; i++)
		    for (int j = 0; j < N; j++) {
				// Random numbers between 1.0 and 10.0
				aData[RM(i,j,N)] = adata(i,j,N);
				bData[RM(i,j,N)] = bdata(i,j,N);
		    }
    }
    /* Generate reference data */
    multMatrixSimple(N, aData, bData, gData);
    allocN = N;
    // Get things ready on the Cuda side
    cudaSetup(N, aData, bData, gData);
}


// Get scratch for matrix
static float *scratchMatrix(int N) {
    if (allocN != N) {
		setup(N);
    }
    return sData;
}

// Test two matrices for equality.  Return number of mismatches
int checkMatrix(int N, float *matTest, float *matGood) {
    int errCount = 0;
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++) {
	    int idx = RM(i,j,N);
	    float test = matTest[idx];
	    float good = matGood[idx];
	    float err = (test-good)/test;
	    if (err < -errTolerance || err > errTolerance) {
			if (++errCount <= errLimit)
			    fprintf(stderr, "\tMismatch.  N=%d.\ttest[%d][%d] = %.3f.  good[%d][%d] = %.3f\n", N, i, j, test, i, j, good);
	    }   
	}
    if (errCount > 0) {
		fprintf(stderr, "\t%d errors\n", errCount);
    }
    return errCount;
}

// Transpose a matrix
void transposeMatrix(int N, float *matS, float *matD) {
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	    matD[CM(i,j,N)] = matS[RM(i,j,N)];
}

// Standard multiplication
void multMatrixSimple(int N, float *matA, float *matB, float *matC) {
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++) {
	    float sum = 0.0;	    
	    for (int k = 0; k < N; k++)
			sum += matA[RM(i,k,N)] * matB[RM(k,j,N)];
	    matC[RM(i,j,N)] = sum;
	}
}

// Multiplication, first transposing B 
void multMatrixTransposed(int N, float *matA, float *matB, float *matC) {
    float *tranB = scratchMatrix(N);
    transposeMatrix(N, matB, tranB);
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++) {
	    float sum = 0.0;	    
	    for (int k = 0; k < N; k++)
			sum += matA[RM(i,k,N)] * tranB[RM(j,k,N)];
	    matC[RM(i,j,N)] = sum;
	}
}

void multMatrixTransposeBlocked(int N, float *matA, float *matB, float *matC) {
    float *tranB = scratchMatrix(N);
    transposeMatrix(N, matB, tranB);
    /* Zero out C */
    memset(matC, 0, N * N * sizeof(float));
    int i, j, k;
    for (i = 0; i <= N-SBLK; i+= SBLK) {
	for (j = 0; j <= N-SBLK; j+= SBLK) {
	    for (k = 0; k <= N-SBLK; k+=SBLK) {
		for (int bi = 0; bi < SBLK; bi++) 
		    for (int bj = 0; bj < SBLK; bj++) {
				float sum = 0.0;
				for (int bk =0; bk < SBLK; bk++)
				    sum += matA[RM(i+bi,k+bk,N)] * tranB[RM(j+bj,k+bk,N)];
				matC[RM(i+bi,j+bj,N)] += sum;
		    }
	    }
	    // Finish rest of k
	    for (int bi = 0; bi < SBLK; bi++) 
		for (int bj = 0; bj < SBLK; bj++) {
		    float sum = 0.0;
		    for (int rk = k; rk < N; rk++)
				sum += matA[RM(i+bi,rk,N)] * tranB[RM(j+bj,rk,N)];
		    matC[RM(i+bi,j+bj,N)] += sum;
		}
	}
	// Finish rest of j
	for (int bi = 0; bi < SBLK; bi++)
	    for (int rj = j; rj < N; rj++) {
			float sum = 0.0;	    
			for (k = 0; k < N; k++)
			    sum += matA[RM(i+bi,k,N)] * tranB[RM(rj,k,N)];
			matC[RM(i+bi,rj,N)] += sum;
	    }
    }
    // Finish rest of i
    for (int ri = i; ri < N; ri++)
	for (j = 0; j < N; j++) {
	    float sum = 0.0;	    
	    for (k = 0; k < N; k++)
			sum += matA[RM(ri,k,N)] * tranB[RM(j,k,N)];
	    matC[RM(ri,j,N)] += sum;
	}
}

// Multiplication, first transposing B 
void multMatrixMKL(int N, float *matA, float *matB, float *matC) {

	/* initialization code is skipped for brevity (do a dummy dsecnd() call to improve accuracy of timing) */
	float alpha = 1.0, beta = 1.0;
	/* first call which does the thread/buffer initialization */
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, matA, N, matB, N, beta, matC, N);
}



static int runMM(int N, mmul_t method) {
    switch (method) {
	    case MMUL_REFERENCE:
			multMatrixSimple(N, aData, bData, tData);
		break;
	    case MMUL_TRANSPOSE:
			multMatrixTransposed(N, aData, bData, tData);
		break;
	    case MMUL_TRANSPOSE_BLK:
			multMatrixTransposeBlocked(N, aData, bData, tData);
		break;
	    case MMUL_MKL:
			multMatrixMKL(N, aData, bData, tData);
		break;
	    default:
			fprintf(stderr, "Haven't implemented method yet\n");
		return 0;
    }
    return 1;
}

double benchMM(int N, mmul_t method) {
    setup(N);
    if (!runMM(N, method))
		return 1000.0;
    if (checkMatrix(N, tData, gData) > 0)
		return 1000.0;
    /* Now do the real benchmarking */
    long ops = (long) 2 * N * N * N;
    long runs = (targetOps+ops-1)/ops;
    double startTime = CycleTimer::currentSeconds();
    for (long r = 0; r < runs; r++)
		runMM(N, method);
    double endTime = CycleTimer::currentSeconds();
    double ms = (endTime - startTime) * 1000.0;
    double gflops = (long) (runs*ops)/ms * 1e-6;
    //fprintf(stderr, "%ld runs, %ld ops/run, %.2f ms, %.3f GFlops\n", runs, ops, ms, gflops);
    return gflops;
}
