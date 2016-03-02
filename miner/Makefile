TARGET := miner

objects = main.o step_a.o sha.o
# sha.o step_a.o step_b.o

HOST_COMPILER := clang++
INCLUDE_PATH := /Developer
CUDA_PATH ?= /Developer/NVIDIA/CUDA-7.5
UNAME = $(shell uname)

ifeq ($(UNAME),Darwin)
	NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) -I$(INCLUDE_PATH)
else
	NVCC := nvcc -I/usr/local/cuda-7.0/samples/common/inc -I/usr/local/cuda/include
endif

all: $(objects)
	$(NVCC) -arch=sm_30 $(objects) -o $(TARGET)

%.o: %.cu
	$(NVCC) -arch=sm_30 -I. -dc $< -o $@

clean:
	rm -f *.o $(TARGET)
