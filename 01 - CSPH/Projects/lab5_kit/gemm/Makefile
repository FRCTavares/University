PATH := $(CURDIR)/../common/tools:$(PATH)
CC = g++
CFLAGS = -g -O3 -mfma -mavx2 -fopenmp -foffload=-lm -Wall -fcf-protection=none -fno-stack-protector -I../

gemm: gemm.cpp
	chmod +x $(CURDIR)/../common/tools/ptxas
	$(CC) $(CFLAGS) gemm.cpp -o gemm

clean:
	rm -f gemm

