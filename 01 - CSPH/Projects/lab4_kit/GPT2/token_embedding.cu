#include "header.cuh"
//Simply performs the token embedding element-wise, slide 53
__global__ void gather_token_pos_embeddings(const float* wte, const float* wpe,
                                            const int* token_ids,
                                            float* out, int seq_len, int d_model)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = tid / d_model;
    int dim_idx   = tid % d_model;

    if(token_idx < seq_len) {
        int token_id = token_ids[token_idx];
        out[token_idx * d_model + dim_idx] =  wte[token_id * d_model + dim_idx] + wpe[token_idx * d_model + dim_idx];
    }
}
