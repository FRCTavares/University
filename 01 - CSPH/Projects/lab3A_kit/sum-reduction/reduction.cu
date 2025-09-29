#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"




// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
reduction_kernel(int N, int* array, int stride) {

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0 && (i+s)<N) {
            array[i*stride] += array[(i + s)*stride];
        }
    __syncthreads();
    }

}


int reductionCuda(int N, int* array) {

    
    const int threadsPerBlock = 512;

    int result;

    // this is "THE TRICK" to obtain the rounded-up number of thread blocks (i.e., compute the ceiling)
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;


    int *device_array;
    // STUDENTS TODO: allocate device memory buffers on the GPU using cudaMalloc, and copy host data to it using CudaMemcpy.

        
    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    double startTimeKernel = CycleTimer::currentSeconds();

    int stride=1;

    //
    // STUDENTS TODO: finish the loop that launches the kernels
    // In each loop iteration, you have to update the following variables:
    // N - represents the number of elements that still need to be reduced
    // stride - represents the distance between each element in the array
    // blocks - Number of blocks to be launched

    while(N>1){
        reduction_kernel<<<blocks, threadsPerBlock>>>(N, device_array,stride);
        cudaDeviceSynchronize();
    }
    
    double endTimeKernel = CycleTimer::currentSeconds();
    //
    // STUDENTS TODO: copy result from GPU back to CPU using cudaMemcpy
    //
    
    
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
		errCode, cudaGetErrorString(errCode));
    }


    double overallDuration = endTime - startTime;
    double overallDurationKernel = endTimeKernel - startTimeKernel;

    printf("Kernel time: %.3f ms\n", 1000.f * overallDurationKernel);

    //
    // STUDENTS TODO: free memory buffers on the GPU using cudaFree
    //

    return result;
    
}

void printCudaInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
