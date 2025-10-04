#include "header.cuh"

//This file is just to load weights
// ------------------- Helper to load .bin -------------------
std::vector<float> load_bin_host(const std::string& filename, size_t n_elements) {
    std::vector<float> data(n_elements);
    std::ifstream f(filename, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << filename << "\n"; exit(1); }
    f.read(reinterpret_cast<char*>(data.data()), n_elements * sizeof(float));
    f.close();
    return data;
}

float* load_bin_cuda(const std::string& filename, size_t n_elements) {
    float* d_ptr;
    std::vector<float> host = load_bin_host(filename, n_elements);
    cudaMalloc(&d_ptr, n_elements * sizeof(float));
    cudaMemcpy(d_ptr, host.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice);
    return d_ptr;
}

// ------------------- Load All Weights -------------------
void load_all_weights(GPT2MediumWeights& weights, const std::string& folder, size_t d_model=1024, size_t vocab_size=50257, size_t max_pos=1024) {
    auto join = [&](const std::string& f) { return folder + "/" + f; };

    weights.wte = load_bin_cuda(join("wte.bin"), vocab_size*d_model);
    weights.wpe = load_bin_cuda(join("wpe.bin"), max_pos*d_model);

    for (int i = 0; i < 24; i++) {
        weights.blocks[i].qkv_w = load_bin_cuda(join("qkv_weight_" + std::to_string(i) + ".bin"), 3*d_model*d_model);
        weights.blocks[i].qkv_b = load_bin_cuda(join("qkv_bias_" + std::to_string(i) + ".bin"), 3*d_model);

        weights.blocks[i].attn_proj_w = load_bin_cuda(join("attn_proj_weight_" + std::to_string(i) + ".bin"), d_model*d_model);
        weights.blocks[i].attn_proj_b = load_bin_cuda(join("attn_proj_bias_" + std::to_string(i) + ".bin"), d_model);

        weights.blocks[i].mlp_fc_w = load_bin_cuda(join("mlp_fc_weight_" + std::to_string(i) + ".bin"), 4*d_model*d_model);
        weights.blocks[i].mlp_fc_b = load_bin_cuda(join("mlp_fc_bias_" + std::to_string(i) + ".bin"), 4*d_model);
        weights.blocks[i].mlp_proj_w = load_bin_cuda(join("mlp_proj_weight_" + std::to_string(i) + ".bin"), d_model*4*d_model);
        weights.blocks[i].mlp_proj_b = load_bin_cuda(join("mlp_proj_bias_" + std::to_string(i) + ".bin"), d_model);

        weights.blocks[i].ln1_gamma = load_bin_cuda(join("ln1_gamma_" + std::to_string(i) + ".bin"), d_model);
        weights.blocks[i].ln1_beta  = load_bin_cuda(join("ln1_beta_" + std::to_string(i) + ".bin"), d_model);
        weights.blocks[i].ln2_gamma = load_bin_cuda(join("ln2_gamma_" + std::to_string(i) + ".bin"), d_model);
        weights.blocks[i].ln2_beta  = load_bin_cuda(join("ln2_beta_" + std::to_string(i) + ".bin"), d_model);
    }

    weights.ln_f_gamma = load_bin_cuda(join("ln_f_gamma.bin"), d_model);
    weights.ln_f_beta  = load_bin_cuda(join("ln_f_beta.bin"), d_model);
    weights.lm_head    = load_bin_cuda(join("lm_head.bin"), vocab_size*d_model);

    std::cout << "✅ All weights loaded from folder: " << folder << "\n";
}
