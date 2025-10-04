#include "header.cuh"

int attention_cnt = 0;
cublasComputeType_t g_computeType_attention = CUBLAS_COMPUTE_32F; //  Flag to change the compute type

//Perform element-wise scaling
__global__ void scale_vector(float* x, float scale, int n){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= n) return;
    x[idx] *= scale;
}


// Multi-Head Attention forward (QKV → softmax → projection)
//Function responsible for the whole Attention Block, performs all GEMMs using CuBLAS
//Corresponds to the start of slide 57
void attention_forward(cublasHandle_t handle,
                       float* hidden_seq,      // [seq_len x d_model] row-major
                       float* output_seq,      // [seq_len x d_model] row-major
                       int seq_len,
                       TransformerBlockWeights& w,
                       int d_model,
                       int num_heads)
{
    int head_dim = d_model / num_heads;
    float alpha = 1.0f, beta = 0.0f;

    // Allocate QKV: [seq_len x 3*d_model] 
    float *qkv, *h_seq;
    cudaMalloc(&qkv, seq_len * 3 * d_model * sizeof(float));
    h_seq = (float*) malloc(seq_len * d_model * sizeof(float));
    cudaMemcpy(h_seq, hidden_seq, d_model * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
   
    #ifdef TIMERS_ATTENTION 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;
        cudaEventRecord(start);
    #endif
    //First GEMM done, corresponds to the highlighted in slide 58
    // It does model projection
    cublasGemmEx(handle,  CUBLAS_OP_T, CUBLAS_OP_T,
            seq_len, 3*d_model, d_model,
            &alpha,
            hidden_seq, CUDA_R_32F, d_model,
            w.qkv_w, CUDA_R_32F,  3*d_model,
            &beta,
            qkv, CUDA_R_32F, seq_len,
            g_computeType_attention ,  
            CUBLAS_GEMM_DEFAULT);

    #ifdef TIMERS_ATTENTION

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GEMM1: " << milliseconds << " ms" << std::endl;
    #endif
    int threads = 256;
    int blocks = (seq_len * 3 * d_model + threads - 1) / threads;

    #ifdef TIMERS_ATTENTION 
    milliseconds = 0;
    cudaEventRecord(start);
    #endif
    //Adding the respective Bias, still in slide 58
    add_bias_kernel_CM<<<blocks, threads>>>(qkv, w.qkv_b, seq_len * 3 * d_model, seq_len);
    #ifdef TIMERS_ATTENTION
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "add_bias_kernel_CM1: " << milliseconds << " ms" << std::endl;
    #endif
   

    // Split Q, K, V
    float* Q = qkv;                       // [seq_len x d_model]
    float* K = qkv + seq_len * d_model;   // [seq_len x d_model]
    float* V = qkv + 2 * seq_len * d_model; // [seq_len x d_model]


    // Allocate attention output buffer
    float* attn_output;
    cudaMalloc(&attn_output, seq_len * d_model * sizeof(float));

    // Multi-head attention
    // This split here corresponds to slide 59. We split our matrix into the number of heads
    for(int h = 0; h < num_heads; h++){
        float* Qh = Q + h * head_dim*seq_len;   // [seq_len x head_dim]
        float* Kh = K + h * head_dim*seq_len;   // [seq_len x head_dim]
        float* Vh = V + h * head_dim*seq_len;   // [seq_len x head_dim]
        float* out_h = attn_output + h * head_dim*seq_len; // [seq_len x head_dim]

        // Allocate attention scores: [seq_len x seq_len]
        float* attn_scores;
        cudaMalloc(&attn_scores, seq_len * seq_len * sizeof(float));
        #ifdef TIMERS_ATTENTION 
            milliseconds = 0;
            cudaEventRecord(start);
        #endif
        //GEMM of Q x K^T 
        cublasGemmEx(handle,  CUBLAS_OP_N, CUBLAS_OP_T,
                seq_len, seq_len, head_dim,
                &alpha,
                Kh, CUDA_R_32F, seq_len,
                Qh, CUDA_R_32F,  seq_len,
                &beta,
                attn_scores, CUDA_R_32F, seq_len,
                g_computeType_attention ,  
                CUBLAS_GEMM_DEFAULT);
        #ifdef TIMERS_ATTENTION
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);  
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "GEMM2: " << milliseconds << " ms" << std::endl;
        #endif
       
        // Scale attention scores by sqrt(head_dim)
        float scale = 1.0f / sqrtf((float)head_dim);
        int attn_threads = 256;
        int attn_blocks = (seq_len * seq_len + attn_threads - 1) / attn_threads;
        //Scaling and Softmax, corresponds to slide 60
        scale_vector<<<attn_blocks, attn_threads>>>(attn_scores, scale, seq_len * seq_len);
        blocks = (seq_len*32 + threads - 1) / threads;
        softmax_warp_rows<<<blocks, threads>>>(attn_scores, seq_len);
       
        #ifdef TIMERS_ATTENTION 
            milliseconds = 0;
            cudaEventRecord(start);
        #endif
        //Multiplication by Vi, corresponds to slide 61
        cublasGemmEx(handle,  CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len, head_dim, seq_len, 
            &alpha,
            attn_scores, CUDA_R_32F, seq_len,
                Vh, CUDA_R_32F,  seq_len,
            &beta,
            out_h, CUDA_R_32F, seq_len,
            g_computeType_attention ,  
            CUBLAS_GEMM_DEFAULT);

        #ifdef TIMERS_ATTENTION
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);  
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "GEMM3: " << milliseconds << " ms" << std::endl;
        #endif
        
       
        cudaFree(attn_scores);
    }


    #ifdef TIMERS_ATTENTION 
        milliseconds = 0;
        cudaEventRecord(start);
    #endif
    //Multiplication to return to the model dimensions, slide 62
    cublasGemmEx(handle,  CUBLAS_OP_N, CUBLAS_OP_T,
                d_model, seq_len, d_model,  
                &alpha,
                w.attn_proj_w, CUDA_R_32F, d_model,
                attn_output, CUDA_R_32F,  seq_len,
                &beta,
                output_seq, CUDA_R_32F, d_model,
                g_computeType_attention ,  
                CUBLAS_GEMM_DEFAULT);

    #ifdef TIMERS_ATTENTION
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "GEMM4: " << milliseconds << " ms" << std::endl;
    #endif
   

    #ifdef TIMERS_ATTENTION 
        milliseconds = 0;
        cudaEventRecord(start);
        #endif
        blocks = (seq_len * d_model + threads - 1) / threads;
        //Respective Bias add
        add_bias_kernel<<<blocks, threads>>>(output_seq, w.attn_proj_b, seq_len * d_model, d_model);
        #ifdef TIMERS_ATTENTION
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "add_bias_kernel_CM2: " << milliseconds << " ms" << std::endl;
    #endif

    cudaFree(qkv);
    cudaFree(attn_output);
}
