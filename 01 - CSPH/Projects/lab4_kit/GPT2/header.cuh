
#pragma once

// ------------------- CUDA / cuBLAS -------------------
#include <cuda_runtime.h>
#include <cublas_v2.h>


// ------------------- Standard C++ -------------------
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>


// ------------------- Structures -------------------
struct TransformerBlockWeights {
    float* qkv_w;  float* qkv_b;
    float* attn_proj_w; float* attn_proj_b;
    float* mlp_fc_w; float* mlp_fc_b;
    float* mlp_proj_w; float* mlp_proj_b;
    float* ln1_gamma; float* ln1_beta;
    float* ln2_gamma; float* ln2_beta;
};

struct GPT2MediumWeights {
    float* wte;  // token embeddings
    float* wpe;  // positional embeddings
    TransformerBlockWeights blocks[24]; // GPT2-medium has 24 transformer blocks
    float* ln_f_gamma; float* ln_f_beta;
    float* lm_head;
};
// ------------------- MACROS -------------------
// #define TIMERS_GENERAL
// #define TIMERS_ATTENTION
#define NUMBER_OF_PRINTS 30



//Forbidden Tokens!
const int banned_tokens[] = {
    0, 3228, 10185, 13896, 50184, 34635, 37160, 2474, 40754, 48220,
    13679, 49296, 8133, 26290, 19588, 28265, 28112, 43179, 48443, 48725,
    22857, 42720, 36463, 1, 40484, 15931, 30543, 18109, 4943, 48774,
    12340, 11074, 15341, 1600, 2430, 26793, 1911, 26214, 42924, 26700,
    1298, 2404, 34713, 47182, 15473, 48219, 20598, 26358, 32509, 8351,
    8172, 5320, 22039, 23984, 13984, 17912, 8973, 33116, 34171, 23785,
    20662, 25719, 13018, 11919, 42785, 24426, 15327, 2, 2235, 21017, 4242,
    7804, 14468, 29113, 29953, 34206, 3, 13702, 36737, 47113, 35307, 38892,
    4, 39658, 16626, 36917, 4407, 15920, 18823, 49563, 7441, 33963, 7225,
    48529, 26525, 39850, 5, 25226, 6, 29653, 7061, 39115, 35384, 44648,
    11537, 33809, 27691, 24036, 3256, 40264, 41707, 29001, 4458, 30827,
    26488, 10354, 17020, 44167, 30960, 20520, 1549, 1183, 1101, 821, 338,
    470, 1053, 7, 7203, 16763, 39434, 10786, 19510, 3419, 28955, 35430,
    22784, 22446, 33529, 9783, 39893, 46491, 32590, 25106, 19529, 27097,
    42744, 20995, 438, 44785, 6329, 650, 30934, 23031, 26866, 982, 45537,
    35937, 32284, 10541, 32501, 26171, 24305, 1783, 19351, 22369, 3880,
    47232, 43801, 10097, 46904, 34507, 3784, 49146, 22831, 13, 526, 32203,
    19570, 33283, 41424, 18161, 32535, 48082, 2637, 11496, 13531, 50113,
    12195, 2014, 12179, 15729, 47308, 15885, 1539, 44388, 7874, 9816, 492,
    986, 9313, 23029, 1106, 12359, 16317, 25780, 2109, 34617, 44274, 4181,
    49129, 27754, 8864, 23193, 44825, 22345, 40720, 19571, 11207, 15089,
    29847, 25970, 32756, 40791, 3693, 8183, 13557, 44587, 37585, 13402,
    43735, 40670, 14, 30487, 31113, 32624, 29006, 34729, 15211, 35343, 28404,
    47454, 16327, 11757, 1003, 20379, 9705, 16150, 27246, 49704, 15913, 20924,
    47835, 27643,198,26,8
};


// ------------------- Function declarations -------------------
#pragma once
extern __global__ void softmax_warp_rows(float* x, int seq_len);
int sample_next_token(const float* logits, int vocab_size,int top_k,float top_p,const std::vector<int>& recent_tokens , float repetition_penalty , float temperature );
void load_all_weights(GPT2MediumWeights& weights, const std::string& folder, size_t d_model, size_t vocab_size, size_t max_pos);
float* load_bin_cuda(const std::string& filename, size_t n_elements);
std::vector<float> load_bin_host(const std::string& filename, size_t n_elements);
__global__ void add_bias_kernel(float* x, const float* b, int n, int cols);
__global__ void add_bias_kernel_CM(float* x, const float* b, int n, int cols);
__global__ void gather_token_pos_embeddings(const float* wte, const float* wpe, const int* token_ids, float* out, int seq_len, int d_model);

// Full forward for one token (returns predicted token ID)
int gpt2_forward_predict_token(cublasHandle_t handle,
                               GPT2MediumWeights& weights,
                               const std::vector<int>& token_ids,
                               int d_model, int vocab_size,
                               int num_heads);

// Existing helper functions
void layernorm_forward(float* x, const float* gamma, const float* beta, int seq_len, int d_model);
void attention_forward(cublasHandle_t handle,
                       float* hidden_seq,     
                       float* output_seq,     
                       int seq_len,
                       TransformerBlockWeights& w,
                       int d_model,
                       int num_heads);
void mlp_forward(cublasHandle_t handle, float* input, float* output, TransformerBlockWeights& w, int d_model);
void transformer_block_forward(cublasHandle_t handle,
                               float* hidden_seq,     // [seq_len, d_model]
                               float* tmp_seq,        // temporary buffer, [seq_len, d_model]
                               int seq_len,
                               TransformerBlockWeights& w,
                               int d_model,
                               int num_heads);



