PATH := $(CURDIR)/../common/tools:$(PATH)
CC = g++
CFLAGS = -g -O3 -mfma -mavx2 -fopenmp -foffload=-lm -Wall -fcf-protection=none -fno-stack-protector -I../

multikernel: multikernel.cpp
	chmod +x $(CURDIR)/../common/tools/ptxas
	$(CC) $(CFLAGS) multikernel.cpp -o multikernel

clean:
	rm -f multikernel

