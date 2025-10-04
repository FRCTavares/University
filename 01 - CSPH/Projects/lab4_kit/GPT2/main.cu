#include "header.cuh"

// ------------------- Main -------------------
int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "Usage: ./gpt2_infer \"<token_ids_comma_separated>\"\n";
        return 1;
    }

    // ------------------- Hardcoded weights folder -------------------
    std::string folder = "/extra/csph/gpt2/gpt2_medium_bin";

    // ------------------- Parse input token IDs -------------------
    std::string input_tokens_str = argv[1];
    std::vector<int> tokens;
    std::stringstream ss(input_tokens_str);
    std::string token;
    while(std::getline(ss, token, ',')) {
        tokens.push_back(std::stoi(token));
    }

    std::vector<int> generated = tokens;

    // ------------------- Generation parameters -------------------
    
    int d_model = 1024; //size of the model's hidden dimension
    int vocab_size = 50257; // vocabulary size
    int num_heads = 16; // number of attention heads
    int num_generate = (argc >= 3) ? std::stoi(argv[2]) : 10; // generate 10 tokens

     // ------------------- Initialize CUDA and load weights -------------------
    GPT2MediumWeights weights;
    cudaSetDevice(0);
    load_all_weights(weights, folder, d_model,vocab_size,d_model);

    cublasHandle_t handle;
    cublasCreate(&handle);
   std::cout << "Number of inputs: " << generated.size() << std::endl;
    
    for(int step = 0; step < num_generate; step++){
        int next_token = gpt2_forward_predict_token(handle, weights, generated, d_model, vocab_size, num_heads);
        generated.push_back(next_token);
    }
    

    // ------------------- Print results -------------------
    std::cout << "Input token IDs: ";
    for(int t : tokens) std::cout << t << " ";
    std::cout << "\n";

    std::cout << "Generated token IDs: ";
    for(size_t i = 0; i < generated.size(); ++i) {
        std::cout << generated[i];
        if(i != generated.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    
    // ------------------- Cleanup -------------------
    cublasDestroy(handle);
    

    // Free GPU memory
    cudaFree(weights.wte);
    cudaFree(weights.wpe);
    cudaFree(weights.ln_f_gamma);
    cudaFree(weights.ln_f_beta);
    cudaFree(weights.lm_head);
    for(int i=0;i<24;i++){
        cudaFree(weights.blocks[i].qkv_w);
        cudaFree(weights.blocks[i].qkv_b);
        cudaFree(weights.blocks[i].attn_proj_w);
        cudaFree(weights.blocks[i].attn_proj_b);
        cudaFree(weights.blocks[i].mlp_fc_w);
        cudaFree(weights.blocks[i].mlp_fc_b);
        cudaFree(weights.blocks[i].mlp_proj_w);
        cudaFree(weights.blocks[i].mlp_proj_b);
        cudaFree(weights.blocks[i].ln1_gamma);
        cudaFree(weights.blocks[i].ln1_beta);
        cudaFree(weights.blocks[i].ln2_gamma);
        cudaFree(weights.blocks[i].ln2_beta);
    }
    cudaDeviceReset();

    return 0;
}
