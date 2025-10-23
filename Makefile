// ==================== Makefile ====================
// Save this as 'Makefile'
/*
NVCC = nvcc
CXX = g++
CUDA_FLAGS = -arch=sm_70 -O3
CXX_FLAGS = -std=c++11 -O3

all: nbody_gpu nbody_cpu

nbody_gpu: main.o nbody_gpu.o utils.o
	$(NVCC) $(CUDA_FLAGS) -o nbody_gpu main.o nbody_gpu.o utils.o

nbody_cpu: main.o nbody_cpu.o utils.o
	$(CXX) $(CXX_FLAGS) -o nbody_cpu main.o nbody_cpu.o utils.o

main.o: main.cpp nbody.h
	$(CXX) $(CXX_FLAGS) -c main.cpp

nbody_gpu.o: nbody_gpu.cu nbody.h
	$(NVCC) $(CUDA_FLAGS) -c nbody_gpu.cu

nbody_cpu.o: nbody_cpu.cpp nbody.h
	$(CXX) $(CXX_FLAGS) -c nbody_cpu.cpp

utils.o: utils.cpp nbody.h
	$(CXX) $(CXX_FLAGS) -c utils.cpp

clean:
	rm -f *.o nbody_gpu nbody_cpu

test: all
	./scripts/run_tests.sh
*/