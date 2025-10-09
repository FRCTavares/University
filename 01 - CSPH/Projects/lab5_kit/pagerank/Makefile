PATH := $(CURDIR)/../common/tools:$(PATH)
CC = g++
CFLAGS = -g -O3 -mfma -mavx2 -fopenmp -foffload=-lm -Wall -fcf-protection=none -fno-stack-protector -I../

pagerank: pagerank.cpp
	chmod +x $(CURDIR)/../common/tools/ptxas
	$(CC) $(CFLAGS) -std=c++11 pagerank.cpp ../common/graph.cpp -o pagerank

clean:
	rm -f pagerank

