#include "header.cuh"
//Normalization kernel, uses multiple threads and shared memory to calculate the average, slide 54
__global__ void layernorm_kernel(float* x, const float* gamma, const float* beta, int seq_len, int d_model, float eps=1e-5f) {
    int token_idx = blockIdx.x;         
    int tid = threadIdx.x;              

    extern __shared__ float sdata[];    

    if(token_idx >= seq_len) return;

    // Step 1: compute mean
    float sum = 0.0f;
    for(int i = tid; i < d_model; i += blockDim.x){
        sum += x[token_idx*d_model + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // reduce within block
    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if(tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float mean = sdata[0] / d_model;
    __syncthreads();

    // Step 2: compute variance
    float var_sum = 0.0f;
    for(int i = tid; i < d_model; i += blockDim.x){
        float val = x[token_idx*d_model + i] - mean;
        var_sum += val * val;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if(tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / d_model + eps);
    __syncthreads();

    // Step 3: normalize and apply gamma/beta
    for(int i = tid; i < d_model; i += blockDim.x){
        int idx = token_idx*d_model + i;
        x[idx] = (x[idx] - mean) * inv_std * gamma[i] + beta[i];
    }
}
// CPU function that will call the CUDA Kernel responsible for the normalization (slide 54)
void layernorm_forward(float* x, const float* gamma, const float* beta, int seq_len, int d_model) {
    int threads = 256;                       
    int blocks = seq_len;                    
    size_t shared_mem = threads * sizeof(float); 
    layernorm_kernel<<<blocks, threads, shared_mem>>>(x, gamma, beta, seq_len, d_model);
}