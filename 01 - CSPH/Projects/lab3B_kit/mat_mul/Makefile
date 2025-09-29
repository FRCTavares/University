EXECUTABLE := matrix

CU_FILES   := cudaMatrix.cu

CU_DEPS    := matrix.h

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

MKLROOT=/opt/intel/oneapi/mkl

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g  -DMKL_ILP64 -I"${MKLROOT}/include"
CUDA=cuda
LDFLAGS=-L/usr/local/$(CUDA)/lib64/ -lcudart -lcublas  -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
NVCC=/usr/local/$(CUDA)/bin/nvcc
NVCCFLAGS=-O3 -m64 -arch=sm_89 -g

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/matrix.o $(OBJDIR)/cudaMatrix.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
