EXECUTABLE := lab3_classwork

CU_FILES   := gpu_code.cu

CU_DEPS    :=

CC_FILES   := main.cpp, cpu_code.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture sm_89


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/gpu_code.o  $(OBJDIR)/cpu_code.o


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
