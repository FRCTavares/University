#include "header.cuh"


//General add bias kernel, done element-wise
__global__ void add_bias_kernel(float* x, const float* b, int n, int cols){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= n) return;
    int col = idx % cols;
    x[idx] += b[col];
}
//General add bias kernel, done element-wise but assume Column Major
__global__ void add_bias_kernel_CM(float* x, const float* b, int n, int cols){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= n) return;
    int row = idx/cols;
  
    x[idx] += b[row];
}

//Optimized Softmax
__global__ void softmax_warp_rows(float* x, int seq_len)
{
    const unsigned FULL_MASK = 0xffffffffu;
    int num_rows = seq_len;
    // Warp info inside this block
    int warp_id_in_block = threadIdx.x / 32;   // which warp inside the block
    int lane            = threadIdx.x % 32;    // lane id within the warp

    // Global warp index -> row index
    int global_warp_id = blockIdx.x * (blockDim.x / 32) + warp_id_in_block;
    if (global_warp_id >= num_rows) return;

    int row_offset = global_warp_id * seq_len;

    // ---- Step 1: max reduction within warp ----
    float local_max = -3.402823466e+38f;
    for (int i = lane; i < seq_len; i += 32) {
        float v = x[row_offset + i];
        if (v > local_max) local_max = v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    float row_max = __shfl_sync(FULL_MASK, local_max, 0);

    // ---- Step 2: sum of exponentials ----
    float local_sum = 0.0f;
    for (int i = lane; i < seq_len; i += 32) {
        float e = expf(x[row_offset + i] - row_max);
        x[row_offset + i] = e;    // keep exp for reuse
        local_sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    float row_sum = __shfl_sync(FULL_MASK, local_sum, 0);

    // ---- Step 3: normalize ----
    float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    for (int i = lane; i < seq_len; i += 32) {
        x[row_offset + i] *= inv_sum;
    }
}

// ---------------- Global RNG ----------------
std::mt19937 gen(std::random_device{}());
//Function to choose our output token, currently it is tuned to be rigid
int sample_next_token(
    const float* logits,           // Pointer to logits array
    int vocab_size,                // Size of vocabulary
    int top_k = 50,                // Top-k filtering
    float top_p = 0.9f,            // Top-p (nucleus) filtering
    const std::vector<int>& recent_tokens = {}, // For repetition penalty
    float repetition_penalty = 1.3f,
    float temperature = 1.0f) {
    // --- Create a mutable copy of logits ---
    std::vector<float> adjusted_logits(logits, logits + vocab_size);

    // --- 1. Repetition penalty ---
    for (int token_id : recent_tokens) {
        if (token_id >= 0 && token_id < vocab_size) {
            //  Correctly apply penalty to both positive and negative logits
            if (adjusted_logits[token_id] > 0) {
                adjusted_logits[token_id] /= repetition_penalty;
            } else {
                adjusted_logits[token_id] *= repetition_penalty;
            }
        }
    }

    // ---  Handle greedy decoding case (temperature = 0) ---
    if (temperature == 0.0f) {
        return std::max_element(adjusted_logits.begin(), adjusted_logits.end()) - adjusted_logits.begin();
    }

    // --- 2. Top-k selection ---
    // Create a vector of indices to sort
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    // Reduce k if vocab_size is smaller
    int k = std::min(top_k, vocab_size);

    // Partially sort indices based on their corresponding logit values
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&adjusted_logits](int i1, int i2) {
            return adjusted_logits[i1] > adjusted_logits[i2];
        });

    // --- 3. Compute softmax over the top-k logits ---
    std::vector<float> topk_probs;
    topk_probs.reserve(k);
    float max_logit = adjusted_logits[indices[0]]; // For numerical stability

    float sum_exp = 0.0f;
    for (int i = 0; i < k; ++i) {
        // Apply temperature and calculate exponent
        float p = std::exp((adjusted_logits[indices[i]] - max_logit) / temperature);
        topk_probs.push_back(p);
        sum_exp += p;
    }

    // Normalize to get probabilities
    for (int i = 0; i < k; ++i) {
        topk_probs[i] /= sum_exp;
    }

    // --- 4. Top-p (nucleus) filtering ---
    // The probabilities in topk_probs are already sorted
    float cum_prob = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < k; ++i) {
        cum_prob += topk_probs[i];
        nucleus_size++;
        if (cum_prob >= top_p) {
            break;
        }
    }

    // --- 5. Final sampling ---
    // Create a distribution from the nucleus probabilities
    std::vector<float> final_probs(topk_probs.begin(), topk_probs.begin() + nucleus_size);

    // Re-normalize the nucleus probabilities so they sum to 1
    float nucleus_sum = 0.0f;
    for(float p : final_probs) nucleus_sum += p;
    if (nucleus_sum > 0.0f) {
        for(float& p : final_probs) p /= nucleus_sum;
    } else {
        // Fallback to uniform distribution if all probabilities are zero
        for(float& p : final_probs) p = 1.0f / final_probs.size();
    }
    
    std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
    int sampled_idx_in_nucleus = dist(gen);

    // Map the sampled index back to the original token ID
    int token_id = indices[sampled_idx_in_nucleus];
    
    return token_id;
}

