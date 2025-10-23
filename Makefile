NVCC = nvcc
CXX = g++

# Auto-detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')

# Fallback to sm_70 if detection fails
ifeq ($(GPU_ARCH),)
    GPU_ARCH = 70
    $(warning Could not detect GPU architecture, defaulting to sm_70)
endif

CUDA_FLAGS = -arch=sm_$(GPU_ARCH) -O3
CXX_FLAGS = -std=c++11 -O3

# Source directory
SRC_DIR = src

.PHONY: all clean test info

all: nbody_gpu nbody_cpu

info:
	@echo "Detected GPU architecture: sm_$(GPU_ARCH)"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv

nbody_gpu: main.o nbody_gpu.o nbody_cpu.o utils.o
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

nbody_cpu: main.o nbody_cpu.o nbody_gpu.o utils.o
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/nbody.h
	$(NVCC) $(CUDA_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/main.cpp

nbody_gpu.o: $(SRC_DIR)/nbody_gpu.cu $(SRC_DIR)/nbody.h
	$(NVCC) $(CUDA_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/nbody_gpu.cu

nbody_cpu.o: $(SRC_DIR)/nbody_cpu.cpp $(SRC_DIR)/nbody.h
	$(NVCC) $(CUDA_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/nbody_cpu.cpp

utils.o: $(SRC_DIR)/utils.cpp $(SRC_DIR)/nbody.h
	$(NVCC) $(CUDA_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/utils.cpp

clean:
	rm -f *.o nbody_gpu nbody_cpu

test: all
	./scripts/run_tests.sh