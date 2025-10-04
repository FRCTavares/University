# CUDA compiler
NVCC = /usr/local/cuda/bin/nvcc

# Architecture and optimization
ARCH = -arch=sm_89
OPT = -O3

# Include paths (if needed)
INCLUDES = -I/usr/local/cuda/include

# Libraries
LIBS = -lcublas -lcudart

# Target executable
TARGET = gpt2_infer
SRC = gpt_infer.cu main.cu load_weights.cu aux.cu mlp_forward.cu attention_forward.cu layer_norm.cu token_embedding.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(OPT) $(ARCH) $(INCLUDES) -o $@ $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)
