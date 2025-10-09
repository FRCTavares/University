#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <math.h>

#include "../common/matrix.hpp"

#define MAX_RESIDUAL 0.01
#define ITER_CHUNK 32

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define ABS(X) ((X) > 0 ? (X) : -(X))

struct jcb_result
{
    double elapsed;
    int iterations;
};

typedef jcb_result (*jcb_f)(Matrix &grid);

struct jcb_impl
{
    const char* name;
    jcb_f func;
};

void jacobi_grid_init(Matrix &grid)
{
    // zero everything
    std::fill(grid.data.begin(), grid.data.end(), 0.0);

    // init the side edges to 1
    for (int r = 0; r < grid.rows; r++)
    {
        int idx = r * grid.cols;
        grid.data[idx] = 1.0;
        grid.data[idx + grid.cols - 1] = 1.0;
    }
}

void print_grid(Matrix& grid)
{
    for (int i = 0; i < grid.rows; i++)
    {
        for (int j = 0; j < grid.cols; j++)
        {
            printf("%.1f ", grid.data[i * grid.cols + j]);
        }
        printf("\n");
    }
}

jcb_result jacobi_sequential(Matrix &grid)
{
    Matrix copy = grid.copy();
    int rows = grid.rows;
    int cols = grid.cols;
    float* data = &grid.data[0];
    float* data_new = &copy.data[0];

    int iterations = 0;

    double start_time = omp_get_wtime();

    float residual = 1.e9;
    while (residual > MAX_RESIDUAL)
    {
        // for every row and column of the matrix, except the edges
        for (int r = 1; r < rows - 1; r++)
            for (int c = 1; c < cols - 1; c++)
                // update each cell based on its neighbors
                data_new[r * cols + c] = 0.25 * (
                    data[(r + 0) * cols + (c + 1)] +  // right
                    data[(r + 1) * cols + (c + 0)] +  // top
                    data[(r + 0) * cols + (c - 1)] +  // left
                    data[(r - 1) * cols + (c + 0)]    // bottom
                );


        // compute the max residual, and copy the elements back to `data`
        residual = 0.0;
        for (int r = 1; r < rows - 1; r++)
        {
            for (int c = 1; c < cols - 1; c++)
            {
                int idx = r * cols + c;
                residual = MAX(std::abs(data_new[idx] - data[idx]), residual);
                data[idx] = data_new[idx];
            }
        }

        iterations++;
    }

    return {omp_get_wtime() - start_time, iterations};
}

jcb_result jacobi_parallel(Matrix &grid)
{
    Matrix copy = grid.copy();
    int rows = grid.rows;
    int cols = grid.cols;
    float* data = &grid.data[0];
    float* data_new = &copy.data[0];

    int iterations = 0;

    double start_time = omp_get_wtime();

    float residual = 1.e9;
    while (residual > MAX_RESIDUAL)
    {
        // for every row and column of the matrix, except the edges
        #pragma omp parallel loop
        for (int r = 1; r < rows - 1; r++)
            #pragma omp loop
            for (int c = 1; c < cols - 1; c++)
                // update each cell based on its neighbors
                data_new[r * cols + c] = 0.25 * (
                    data[(r + 0) * cols + (c + 1)] +  // right
                    data[(r + 1) * cols + (c + 0)] +  // top
                    data[(r + 0) * cols + (c - 1)] +  // left
                    data[(r - 1) * cols + (c + 0)]    // bottom
                );


        // compute the max residual, and copy the elements back to `data`
        residual = 0.0;
        #pragma omp parallel loop reduction(max: residual)
        for (int r = 1; r < rows - 1; r++)
        {
            #pragma omp loop
            for (int c = 1; c < cols - 1; c++)
            {
                int idx = r * cols + c;
                residual = MAX(std::abs(data_new[idx] - data[idx]), residual);
                data[idx] = data_new[idx];
            }
        }

        iterations++;
    }

    return {omp_get_wtime() - start_time, iterations};
}

jcb_result jacobi_gpu(Matrix &grid)
{
    Matrix copy = grid.copy();
    int rows = grid.rows;
    int cols = grid.cols;
    float* data = &grid.data[0];
    float* data_new = &copy.data[0];

    int iterations = 0;

    double start_time = omp_get_wtime();

    float residual = 1.0;
    /* Both kernels can be offloaded to the GPU quite effectively. Try to recall the OpenMP
    construct that can help you avoid redundant data transfers. */ 
    while (residual > MAX_RESIDUAL)
    {
        /* This kernel should be pretty straightforward to parallelize on the GPU */
        for (int r = 1; r < rows - 1; r++)
            for (int c = 1; c < cols - 1; c++)
            {
                // update each cell based on its neighbors
                int idx = r * cols + c;
                data_new[idx] = 0.25 * (
                    data[(r + 0) * cols + (c + 1)] +  // right
                    data[(r + 1) * cols + (c + 0)] +  // top
                    data[(r + 0) * cols + (c - 1)] +  // left
                    data[(r - 1) * cols + (c + 0)]    // bottom
                );
            }

        // compute the max residual, and copy the elements back to `data`
        residual = 0.0;
        /* This kernel is a little trickier, you just need to keep two things in mind. First, don't
        forget to map the residual. Secondly, remember that all threads need to work together to arrive
        at the maximum residual, also known as a reduction. OpenMP supports a bunch of reduction
        identifiers, not just a reduction sum! https://www.openmp.org/spec-html/5.0/openmpsu107.html */
        for (int r = 1; r < rows - 1; r++)
            for (int c = 1; c < cols - 1; c++)
            {
                int idx = r * cols + c;
                residual = MAX(std::abs(data_new[idx] - data[idx]), residual);
                data[idx] = data_new[idx];
            }

        iterations++;
    }

    return {omp_get_wtime() - start_time, iterations};
}

int main(int argc, char *argv[])
{
    int mat_size = argc > 1 ? atoi(argv[1]) : 20000;

    Matrix grid = Matrix(mat_size, mat_size);
    jacobi_grid_init(grid);

    std::vector<jcb_impl> implementations = {
        //{"sequential", jacobi_sequential},
        {"parallel", jacobi_parallel},
        {"gpu", jacobi_gpu},
    };

    for (auto impl: implementations)
    {
        auto g = grid.copy();
        auto result = impl.func(g);

        //print_grid(g);
        printf("%s: %d iterations, %f\n", impl.name, result.iterations, result.elapsed);
    }
}