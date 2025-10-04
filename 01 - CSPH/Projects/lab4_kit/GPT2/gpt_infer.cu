#include "header.cuh"


//Print Counters
int layer_norm1_cnt = 0;
int residual_cnt = 0;
int layer_norm2_cnt = 0;
int transformer_cnt = 0;

//Flag to define the compute type of the GEMMs in this file 
cublasComputeType_t g_computeType_infer = CUBLAS_COMPUTE_32F;
//kernel for residual Add, this corrresponds to slide 63
__global__ void residual_add_kernel(const float* x, const float* y, float* out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    out[idx] = x[idx] + y[idx];
}

// ------------------- Forward Functions -------------------
// this fucntion englobes all the transformer layers: Most importantly the Multi Head attention and the MLP forward
// To get a grasp of the functionas and CUDA Kernels we call here, put next to you the class slides with a focus on slide 50
// With this slide, you will be able to understand the mapping 1 to 1
void transformer_block_forward(cublasHandle_t handle,
                               float* hidden_seq,     // [seq_len, d_model]
                               float* tmp_seq,        // temporary buffer, [seq_len, d_model]
                               int seq_len,
                               TransformerBlockWeights& w,
                               int d_model,
                               int num_heads)
{
    int threads = 256;
    int blocks;

    // ---------------- Residual copy ----------------
    float* residual;
    cudaMalloc(&residual, seq_len * d_model * sizeof(float));
    cudaMemcpy(residual, hidden_seq, seq_len * d_model * sizeof(float), cudaMemcpyDeviceToDevice);

    // ---------------- LayerNorm 1 ----------------
    #ifdef TIMERS_GENERAL   
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;
        cudaEventRecord(start);
    #endif
    for(int t=0; t < seq_len; t++){
        layernorm_forward(hidden_seq + t*d_model, w.ln1_gamma, w.ln1_beta, 1, d_model);
        cudaDeviceSynchronize();
    }
    #ifdef TIMERS_GENERAL

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if(layer_norm1_cnt < NUMBER_OF_PRINTS ){
        std::cout << "layernorm_forward: " << milliseconds << " ms" << std::endl;
    }
     layer_norm1_cnt++;
    
    #endif

    // ---------------- Multi-head Attention ----------------
    #ifdef TIMERS_GENERAL   
        milliseconds = 0;
        cudaEventRecord(start);
    #endif
    attention_forward(handle, hidden_seq, tmp_seq, seq_len, w, d_model, num_heads);
    #ifdef TIMERS_GENERAL
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if(attention_cnt < NUMBER_OF_PRINTS){
            std::cout << "attention_forward: " << milliseconds << " ms" << std::endl;
            attention_cnt++;
        }
        
    #endif

    // ---------------- Residual Add ----------------
    
    #ifdef TIMERS_GENERAL   
        milliseconds = 0;
        cudaEventRecord(start);
    #endif
    blocks = (seq_len * d_model + threads - 1)/threads;
    residual_add_kernel<<<blocks, threads>>>(residual, tmp_seq, hidden_seq, seq_len*d_model);
    #ifdef TIMERS_GENERAL
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if(residual_cnt < NUMBER_OF_PRINTS){
            std::cout << "residual_add_kernel: " << milliseconds << " ms" << std::endl;
            residual_cnt++;
        }
        
    #endif
    cudaFree(residual);
    cudaMemcpy(tmp_seq,hidden_seq,seq_len*d_model*sizeof(float),cudaMemcpyDeviceToDevice);

    // ---------------- LayerNorm 2 + MLP ----------------
    #ifdef TIMERS_GENERAL   
        milliseconds = 0;
        cudaEventRecord(start);
    #endif
    for(int t=0; t<seq_len; t++){
        layernorm_forward(tmp_seq + t*d_model, w.ln2_gamma, w.ln2_beta, 1, d_model);
        mlp_forward(handle, tmp_seq + t*d_model, tmp_seq + t*d_model, w, d_model);
    }
    #ifdef TIMERS_GENERAL
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if(layer_norm2_cnt < NUMBER_OF_PRINTS){
            std::cout << "layernorm_forward + mlp_forward: " << milliseconds << " ms" << std::endl;
        }
        layer_norm2_cnt++;
        
    #endif


    // ---------------- Residual Add after MLP ----------------
    #ifdef TIMERS_GENERAL   
        milliseconds = 0;
        cudaEventRecord(start);
    #endif
    residual_add_kernel<<<blocks, threads>>>(hidden_seq, tmp_seq, hidden_seq, seq_len*d_model);
    #ifdef TIMERS_GENERAL
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if(residual_cnt < NUMBER_OF_PRINTS){
            std::cout << "residual_add_kernel: " << milliseconds << " ms" << std::endl;
            residual_cnt++;
        }
        
    #endif
}

//Our main function to predict tokens, responsible for the embedding (slide 53), the transformer block, normalization layer (slide 54) and LM Head (slide 68)
int gpt2_forward_predict_token(cublasHandle_t handle,
                               GPT2MediumWeights& weights,
                               const std::vector<int>& token_ids,
                               int d_model, int vocab_size,
                               int num_heads)
{
    int seq_len = token_ids.size();

    // ---------------- Copy token IDs to GPU ----------------
    int* d_token_ids; 
    cudaMalloc(&d_token_ids, seq_len*sizeof(int));
    cudaMemcpy(d_token_ids, token_ids.data(), seq_len*sizeof(int), cudaMemcpyHostToDevice);

    // ---------------- Compute embeddings ----------------
    float* hidden_seq; 
    cudaMalloc(&hidden_seq, seq_len*d_model*sizeof(float));
    int threads = 256;
    int blocks = (seq_len*d_model + threads - 1)/threads;
    gather_token_pos_embeddings<<<blocks, threads>>>(
        weights.wte, weights.wpe, 
        d_token_ids, hidden_seq, seq_len, d_model);
    cudaDeviceSynchronize();


    // Temporary buffer for Transformer outputs
    float* tmp_seq; 
    cudaMalloc(&tmp_seq, seq_len*d_model*sizeof(float));

    // ---------------- Transformer blocks ----------------
     // Each block processes the full sequence
    for(int i=0; i<24; i++){
        #ifdef TIMERS_GENERAL
        if(transformer_cnt < NUMBER_OF_PRINTS){
            std::cout << "**************************************" << std::endl;
        }
        auto start_time = std::chrono::high_resolution_clock::now();
        #endif

        transformer_block_forward(handle, hidden_seq, tmp_seq, seq_len, weights.blocks[i], d_model, num_heads);
        
        #ifdef TIMERS_GENERAL
         auto end_time = std::chrono::high_resolution_clock::now();
        if(transformer_cnt < NUMBER_OF_PRINTS){
            std::chrono::duration<double> elapsed = end_time - start_time;
            std::cout << "Time to generate transformer_block_forward " <<seq_len << " tokens: "<< elapsed.count() << " seconds." << std::endl;
            std::cout << "**************************************" << std::endl;
        }
        transformer_cnt++;
        #endif
    }

    // ---------------- Final LayerNorm for last token ----------------
    layernorm_forward(hidden_seq,
                  weights.ln_f_gamma,
                  weights.ln_f_beta,
                  seq_len, d_model);


    // ---------------- LM head ----------------
    float* logits; 
    cudaMalloc(&logits, vocab_size*seq_len*sizeof(float));
    float alpha = 1.0f, beta = 0.0f;
    
        // LM Head, slide 68
    cublasGemmEx(handle,  CUBLAS_OP_T, CUBLAS_OP_N,
                vocab_size, seq_len, d_model, 
                &alpha,
                weights.lm_head, CUDA_R_32F, d_model,
                hidden_seq, CUDA_R_32F,  d_model,
                &beta,
                logits, CUDA_R_32F, vocab_size,
                g_computeType_infer ,  
                CUBLAS_GEMM_DEFAULT);

    // ---------------- Copy logits to host and pick argmax ----------------
    std::vector<float> h_logits(vocab_size);
    cudaMemcpy(h_logits.data(), logits+vocab_size*(seq_len-1), vocab_size*sizeof(float), cudaMemcpyDeviceToHost); 
    for(int token_id : banned_tokens) {
        h_logits[token_id] = -1e9f; // probability ≈ 0
    }
    
    std::vector<int> recent_tokens;
    if(token_ids.size() >= 15) 
        recent_tokens.assign(token_ids.end()-15, token_ids.end());
    else 
        recent_tokens = token_ids;
    int predicted_token = sample_next_token( 
        h_logits.data(),      
        vocab_size,
        0.9f,
        75,                    // top-k
        recent_tokens,         // repetition tracking
        1.15f,                   // repetition penalty
        0.8f
    );
    // ---------------- Cleanup ----------------
    cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if(error != cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }
    cudaFree(d_token_ids);
    cudaFree(hidden_seq);
    cudaFree(tmp_seq);
    cudaFree(logits);

    return predicted_token;
}