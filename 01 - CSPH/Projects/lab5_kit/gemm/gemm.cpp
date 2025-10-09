#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#include "../common/matrix.hpp"

typedef double (*gemm_f)(const Matrix &A, const Matrix &Bt, Matrix &C);

struct gemm_impl
{
    const char* name;
    gemm_f func;
};


// computes A x B = C on the CPU in a single thread
double gemm_sequential(const Matrix &A, const Matrix &Bt, Matrix &C)
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

    return omp_get_wtime() - start_time;
}

// computes A x B = C on the CPU with multiple threads
double gemm_parallel_cpu(const Matrix &A, const Matrix &Bt, Matrix &C)
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

    return omp_get_wtime() - start_time;
}

// computes A x B = C on the GPU
double gemm_parallel_gpu(const Matrix &A, const Matrix &Bt, Matrix &C)
{
    int M = C.rows;
    int N = Bt.cols;
    int K = A.cols;

    // Here the matrix data is already expressed in simple arrays
    // The A matrix is M * K
    // B is N * K (since it is transposed)
    // C is M * N
    const float* A_data = &A.data[0];
    const float* Bt_data = &Bt.data[0];
    float* C_data = &C.data[0];

    double start_time = omp_get_wtime();

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

    

    return omp_get_wtime() - start_time;
}

int main(int argc, char *argv[])
{
    int mat_size = argc > 1 ? atoi(argv[1]) : 1024;

    Matrix A(mat_size, mat_size);
    Matrix Bt(mat_size, mat_size); //
    Matrix C(mat_size, mat_size);

    // get the gold matrix to compare the results to
    gemm_sequential(A, Bt, C);
    Matrix gold_matrix = C.copy();

    std::vector<gemm_impl> implementations = {
        {"sequential", gemm_sequential},
        {"omp parallel", gemm_parallel_cpu},
        {"omp gpu", gemm_parallel_gpu}
    };

    for (auto impl: implementations)
    {
        constexpr int repetitions = 5;
        double elapsed_s = 0.0;
        // repeat the test 5 times
        for (int i = 0; i < repetitions; i++)
            elapsed_s += impl.func(A, Bt, C);
        // calculate the average
        elapsed_s /= static_cast<double>(repetitions);
        C.compare(gold_matrix);

        printf("%s: %f\n", impl.name, elapsed_s);
    }

    return 0;
}
