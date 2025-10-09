#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cmath>

#include "../common/matrix.hpp"

typedef double (*rand_f)(const Matrix &A, const Matrix &Bt, Matrix &C, const Vector &X, const Vector &Y, Vector &Z);

struct rand_impl
{
    const char* name;
    rand_f func;
};

// computes C = A x B, Z = 2 * X^Y and C[i][j] += Z[j] on the CPU in a single thread
double mk_sequential(const Matrix &A, const Matrix &Bt, Matrix &C, const Vector &X, const Vector &Y, Vector &Z)
{
    int M = A.rows;
    int N = Bt.cols;
    int K = A.cols;

    double start_time = omp_get_wtime();

    for (int cr = 0; cr < M; cr++)
    {
        for (int cc = 0; cc < N; cc++)
        {
            float val = 0.0;
            for (int k = 0; k < K; k++)
            {
                val += A.data[cr * K + k] * Bt.data[cc * K + k];
            }
            C.data[cr * N + cc] = val;
        }
    }

    for (int i = 0; i < Z.size; i++)
    {
        Z.data[i] = 2.0 * powf(X.data[i], Y.data[i]);
    }

    for (int cr = 0; cr < M; cr++)
    {
        for (int cc = 0; cc < N; cc++)
        {
            C.data[cr * N + cc] += Z.data[cc];
        }
    }

    return omp_get_wtime() - start_time;
}

// computes C = A x B, Z = 2 * X^Y and C[i][j] += Z[j] on the CPU with multiple threads
double mk_parallel_cpu(const Matrix &A, const Matrix &Bt, Matrix &C, const Vector &X, const Vector &Y, Vector &Z)
{
    int M = A.rows;
    int N = Bt.cols;
    int K = A.cols;

    double start_time = omp_get_wtime();

    #pragma omp parallel loop
    for (int cr = 0; cr < M; cr++)
    {
        for (int cc = 0; cc < N; cc++)
        {
            float val = 0.0;
            for (int k = 0; k < K; k++)
            {
                val += A.data[cr * K + k] * Bt.data[cc * K + k];
            }
            C.data[cr * N + cc] = val;
        }
    }

    #pragma omp parallel loop
    for (int i = 0; i < Z.size; i++)
    {
        Z.data[i] = 2.0 * powf(X.data[i], Y.data[i]);
    }

    #pragma omp parallel loop
    for (int cr = 0; cr < M; cr++)
    {
        for (int cc = 0; cc < N; cc++)
        {
            C.data[cr * N + cc] += Z.data[cc];
        }
    }

    return omp_get_wtime() - start_time;
}

// computes C = A x B, Z = 2 * X^Y and C[i][j] += Z[j] on the GPU, using teams
double mk_parallel_gpu(const Matrix &A, const Matrix &Bt, Matrix &C, const Vector &X, const Vector &Y, Vector &Z)
{
    int M = A.rows;
    int N = Bt.cols;
    int K = A.cols;

    const float* A_data = &A.data[0];
    const float* Bt_data = &Bt.data[0];
    float* C_data = &C.data[0];

    int vec_sz = X.size;
    const float* X_data = &X.data[0];
    const float* Y_data = &Y.data[0];
    float* Z_data = &Z.data[0];

    double start_time = omp_get_wtime();

    #pragma omp target data \
        map(to: A_data[0:M*K], Bt_data[0:K*N], X_data[0:vec_sz], Y_data[0:vec_sz]) \
        map(from: C_data[0:M*N], Z_data[0:vec_sz])
    {
        #pragma omp target teams distribute
        for (int cr = 0; cr < M; cr++)
        {
            #pragma omp loop
            for (int cc = 0; cc < N; cc++)
            {
                float val = 0.0;
                for (int k = 0; k < K; k++)
                {
                    val += A_data[cr * K + k] * Bt_data[cc * K + k];
                }
                C_data[cr * N + cc] = val;
            }
        }

        #pragma omp target teams loop
        for (int i = 0; i < vec_sz; i++)
        {
            Z_data[i] = 2.0 * powf(X_data[i], Y_data[i]);
        }

        #pragma omp target teams loop
        for (int cr = 0; cr < M; cr++)
        {
            for (int cc = 0; cc < N; cc++)
            {
                C_data[cr * N + cc] += Z_data[cc];
            }
        }
    }

    return omp_get_wtime() - start_time;

}

/* it's up to you how you wish to parallelize this implementation, but as a hint, look at the memory and
computational complexities of each kernels. Also, determine if there are any dependencies between the
kernels, and what that implies. */
double mk_parallel_opt(const Matrix &A, const Matrix &Bt, Matrix &C, const Vector &X, const Vector &Y, Vector &Z)
{
    int M = A.rows;
    int N = Bt.cols;
    int K = A.cols;

    // Here the matrix and vector data is already expressed in simple arrays
    // The A matrix is M * K
    // B is N * K (since it is transposed)
    // C is M * N
    const float* A_data = &A.data[0];
    const float* Bt_data = &Bt.data[0];
    float* C_data = &C.data[0];

    int vec_sz = X.size;
    const float* X_data = &X.data[0];
    const float* Y_data = &Y.data[0];
    float* Z_data = &Z.data[0];

    double start_time = omp_get_wtime();

    // Kernel 1, matrix multiplication
    for (int cr = 0; cr < M; cr++)
    {
        for (int cc = 0; cc < N; cc++)
        {
            float val = 0.0;
            for (int k = 0; k < K; k++)
            {
                val += A_data[cr * K + k] * Bt_data[cc * K + k];
            }
            C_data[cr * N + cc] = val;
        }
    }

    // Kernel 2, Z[i] = 2 * X[i] ^ Y[i]
    for (int i = 0; i < vec_sz; i++)
    {
        Z_data[i] = 2.0 * powf(X_data[i], Y_data[i]);
    }

    // Kernel 3, C[i][j] += Z[j]
    for (int cr = 0; cr < M; cr++)
    {
        for (int cc = 0; cc < N; cc++)
        {
            C_data[cr * N + cc] += Z_data[cc];
        }
    }

    return omp_get_wtime() - start_time;
}

int main(int argc, char *argv[])
{
    int mat_size = argc > 1 ? atoi(argv[1]) : 1024;
    int vec_size = argc > 2 ? atoi(argv[2]) : 10000000;

    if(vec_size<mat_size)
        vec_size=mat_size;

    Matrix A(mat_size, mat_size);
    Matrix Bt(mat_size, mat_size); //
    Matrix C(mat_size, mat_size);

    Vector X(vec_size);
    Vector Y(vec_size);
    Vector Z(vec_size);

    // get the gold matrix and vector to compare the results to
    mk_sequential(A, Bt, C, X, Y, Z);
    Matrix gold_matrix = C.copy();
    Vector gold_vector = Z;

    std::vector<rand_impl> implementations = {
        {"sequential", mk_sequential},
        {"omp parallel", mk_parallel_cpu},
        {"omp gpu", mk_parallel_gpu},
        {"optimized", mk_parallel_opt}
    };

    for (auto impl: implementations)
    {
        constexpr int repetitions = 5;
        double elapsed_s = 0.0;
        // repeat the test 5 times
        for (int i = 0; i < repetitions; i++)
            elapsed_s += impl.func(A, Bt, C, X, Y, Z);
        // calculate the average
        elapsed_s /= static_cast<double>(repetitions);
        C.compare(gold_matrix);
        Z.compare(gold_vector);

        printf("%s: %f\n", impl.name, elapsed_s);
    }

    return 0;
}
