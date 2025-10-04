#include "header.cuh"
//Applies the GELU activation function element-wise
__global__ void gelu_kernel(float* x, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    const float sqrt_2_over_pi = 0.7978845608f;
    float val = x[idx];
    x[idx] = 0.5f * val * (1.0f + tanhf(sqrt_2_over_pi*(val + 0.044715f*val*val*val)));
}


// GEMV Kernel
// This kernel is responsible for Matrix Vector Multiplication
// It takes 5 arguments:
// mat - matrix
// vec - vector
// out - output
// n_same - dimension by which we perform dot product
// n_diff - remaining matrix dimension
// This kernel will calculate the output row by row, advancing both the matrix row and vector in their common dimension
__global__ void gemv(float *mat, float *vec, float *out, int n_same, int n_diff){
    for(int i=0;i<n_diff;i++){
        float sum=0;
        for(int j=0;j<n_same;j++){
            sum+=mat[j*n_diff+i]*vec[j];
        }
        out[i]=sum;
    } 
}
//MLP Forward
// This layer is composed by 5 kernel calls
// It calls twice the gemv kernel, twice the add_bias_kernel and once the GELU
// This layer starts by expanding the input to four times the model size (1st gemv), applies the activation unit and returns to model size(2nd gemv)
// This corresponds to slide 65
void mlp_forward(cublasHandle_t handle, float* input, float* output, TransformerBlockWeights& w, int d_model){
    float* tmp; cudaMalloc(&tmp, 4*d_model*sizeof(float));
    
    int threads = 256;
    int blocks=(4*d_model + threads -1)/threads;

    gemv<<<1,1>>>( w.mlp_fc_w,input, tmp, d_model, 4*d_model);

    add_bias_kernel<<<(4*d_model+threads-1)/threads, threads>>>(tmp, w.mlp_fc_b, 4*d_model, 4*d_model);
    gelu_kernel<<<blocks, threads>>>(tmp, 4*d_model);
   
    threads=256;
    blocks=(d_model + threads -1)/threads;
    gemv<<<1,1>>>(w.mlp_proj_w,tmp,output,4*d_model,d_model);
    add_bias_kernel<<<blocks, threads>>>(output, w.mlp_proj_b, d_model, d_model);
  
}
