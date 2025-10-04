#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

static inline int updiv(int n, int d)
{
    return (n + d - 1) / d;
}

int runCPU(int threadsPerBlock, int N, int *hostY, int *cpuX, int *cpuY, int *cpuZ, int *cpuW, int *cpuResult);
int runGPU(int threadsPerBlock, int N, int *hostY, int *gpuX, int *gpuY, int *gpuZ, int *gpuW, int *gpuResult);
void printCudaInfo();

int cpu_condition(int i, int *A)
{
    /// TODO: CONDITION
    /// FILL HERE YOUR CONDITION
    /// you should change the code below!
    if ((A[i] < A[i - 2]) &&
        (A[i] < A[i - 1]) &&
        (A[i] < A[i + 1]) &&
        (A[i] < A[i + 2]))
        return 1;

    return 0;
}

void check_results(int size, int *gold, int *result)
{
    for (int i = 0; i < size; i++)
    {
        if (gold[i] != result[i])
        {
            printf("Error at position: [%d] | current value: %d, expecting: %d.\n",
                   i, result[i], gold[i]);
            return;
        }
    }
    printf("Outputs are correct!\n");
}

int main(int argc, char **argv)
{
    int N = 1024 * 1024;
    int *hostY = new int[N];

    printf("Array size: %d\n", N);
    printf("THREADS_PER_BLOCK %d\n", THREADS_PER_BLOCK);

    /// TODO: INIT HOSTY
    // Code to initizalize hostY
    // COPY BELOW THE CODE MARKED AS "Init HostY"
    // you should change the code below!
    for (int i = 0; i < N; i++)
    {
        int m = i % 16;
        if (m < 5)
        {
            hostY[i] = m + 1;
        }
        else if (m < 9)
        {
            hostY[i] = 9 - m;
        }
        else if (m < 10)
        {
            hostY[i] = 2;
        }
        else if (m < 15)
        {
            hostY[i] = 3;
        }
        else
        {
            hostY[i] = 2;
        }
    }

    /*for (int i = 0; i < 32; i++)
    {
        printf("Index %d -> Value: %d\n", i, hostY[i]);
    }
    printf("...\n");*/

    /// RUN CPU IMPLEMENTATION ///
    // everything is fully implemented in cpu_code.cpp!!!!
    int *cpuX = new int[N];
    int *cpuY = new int[N];
    int *cpuZ = new int[N];
    int *cpuW = new int[N];
    int *cpuResult = new int[N];
    int cpuCount = 0;
    cpuCount = runCPU(THREADS_PER_BLOCK, N, hostY,        // inputs
                      cpuX, cpuY, cpuZ, cpuW, cpuResult); // outputs

    /// RUN GPU IMPLEMENTATION ///
    // this is the code you need to complete in gpu_code.cu
    int *gpuX = new int[N];
    int *gpuY = new int[N];
    int *gpuZ = new int[N];
    int *gpuW = new int[N];
    int *gpuResult = new int[N];
    int gpuCount = 0;

    gpuCount = runGPU(THREADS_PER_BLOCK, N, hostY,        // inputs
                      gpuX, gpuY, gpuZ, gpuW, gpuResult); // outputs

    // validate results
    printf("Checking gpuX: ");
    check_results(N, cpuX, gpuX);
    printf("Checking gpuY: ");
    check_results(N, cpuY, gpuY);
    printf("Checking gpuZ: ");
    check_results(N, cpuZ, gpuZ);
    printf("Checking gpuW: ");
    check_results(N, cpuW, gpuW);
    printf("Checking gpuResult: ");
    check_results(N, cpuResult, gpuResult);

    printf("Checking output size: ");
    if (cpuCount != gpuCount)
    {
        printf("Error: Expected %d, got %d.\n", cpuCount, gpuCount);
    }
    else
    {
        printf("Outputs are correct!\n");
    }

    delete[] hostY;

    delete[] cpuX;
    delete[] cpuY;
    delete[] cpuZ;
    delete[] cpuW;
    delete[] cpuResult;
    delete[] gpuX;
    delete[] gpuY;
    delete[] gpuZ;
    delete[] gpuW;
    delete[] gpuResult;

    return 0;
}
