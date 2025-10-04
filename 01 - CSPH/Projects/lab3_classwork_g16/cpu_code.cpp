#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

int cpu_condition(int i, int *A);

static inline int updiv(int n, int d)
{
    return (n + d - 1) / d;
}

void cpu_initX(int threadsPerBlock, int N, int *cpuX)
{
    int numBlocks = updiv(N, threadsPerBlock);
    for (int i = 0; i < numBlocks; i++)
        for (int j = 0; j < threadsPerBlock; j++)
            cpuX[i * threadsPerBlock + j] = i;

    /*for (int i : {0, 13, 54, 200, 300, 500, 600, 1000, 10000, 100000})
    {
        printf("cpuX[%d] = %d\n", i, cpuX[i]);
    }*/
}

void cpu_initY(int size, int *hostY, int *cpuY)
{
    for (int i = 0; i < size; i++)
        cpuY[i] = hostY[i];
}

void cpu_makeZ(int size, int *cpuX, int *cpuY, int *cpuZ)
{
    for (int i = 0; i < size; i++)
        cpuZ[i] = cpuX[i] * cpuY[i];
}

void cpu_makeW(int size, int *cpuZ, int *cpuW)
{
    cpuW[0] = 0;
    cpuW[1] = 0;
    cpuW[size - 1] = 0;
    cpuW[size - 2] = 0;
    for (int i = 2; i < size - 2; i++)
        cpuW[i] = cpu_condition(i, cpuZ);
}

int cpu_find_pattern(int size, int *input, int *output)
{
    int count = 0;
    for (int i = 2; i < size - 2; i++)
    {
        if (cpu_condition(i, input) == 1)
        {
            output[count] = input[i];
            count++;
        }
    }
    return count;
}

void cpu_exclusive_scan(int size, int *input, int *output)
{
    output[0] = 0;
    for (int i = 1; i < size; i++)
    {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// int cpu_find_pattern2(int size, int* cpuW, int* cpuZ, int *output) {
//     int count = 0;
//     for (int i = 0; i < size-1; i++) {
//         if (cpuW[i] != cpuW[i+1]) {
//             output[count] = cpuZ[i];
//             count++;
//         }
//     }
//     return count;
// }

void cpu_print(int size, int *input)
{
    for (int i = 0; i < size; i++)
    {
        printf("[%d]=%d \n", i, input[i]);
    }
}

int runCPU(int threadsPerBlock, int N, int *hostY, int *cpuX, int *cpuY, int *cpuZ, int *cpuW, int *cpuResult)
{
    /// CPU IMPLEMENTATION ///
    cpu_initX(threadsPerBlock, N, cpuX);
    cpu_initY(N, hostY, cpuY);
    cpu_makeZ(N, cpuX, cpuY, cpuZ);
    cpu_makeW(N, cpuZ, cpuW);
    // cpu_print(N, cpuW);
    int cpuCount = cpu_find_pattern(N, cpuZ, cpuResult);
    // cpu_exclusive_scan(N, cpuW, cpuS);
    // cpu_print(cpuCount,cpuResult);
    // printf("count=%d \n", cpuCount);
    return cpuCount;
}
