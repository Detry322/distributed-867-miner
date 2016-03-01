TARGET := miner

objects = main.o step_a.o sha.o
# sha.o step_a.o step_b.o

HOST_COMPILER := clang++
INCLUDE_PATH := /Developer/NVIDIA/CUDA-7.5/samples/common/inc
CUDA_PATH ?= /Developer/NVIDIA/CUDA-7.5
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) -I$(INCLUDE_PATH)

all: $(objects)
	$(NVCC) -arch=sm_30 $(objects) -o $(TARGET)

%.o: %.cu
	$(NVCC) -arch=sm_30 -I. -dc $< -o $@

clean:
	rm -f *.o $(TARGET)
