// ==================== main.cpp ====================
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include "nbody.h"

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <input_file> <n_steps> <output_file> [--cpu|--gpu-naive|--gpu-shared]\n";
    std::cerr << "  Default: --gpu-shared\n";
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
    }
    
    std::string input_file = argv[1];
    int n_steps = std::atoi(argv[2]);
    std::string output_file = argv[3];
    
    // Parse execution mode
    enum Mode { CPU, GPU_NAIVE, GPU_SHARED };
    Mode mode = GPU_SHARED;
    
    if (argc > 4) {
        if (strcmp(argv[4], "--cpu") == 0) {
            mode = CPU;
        } else if (strcmp(argv[4], "--gpu-naive") == 0) {
            mode = GPU_NAIVE;
        } else if (strcmp(argv[4], "--gpu-shared") == 0) {
            mode = GPU_SHARED;
        } else {
            print_usage(argv[0]);
        }
    }
    
    // Load particles
    std::vector<Particle> particles;
    load_particles(input_file, particles);
    
    // Set simulation parameters
    SimulationParams params;
    params.n_particles = particles.size();
    params.n_steps = n_steps;
    params.dt = 0.01f;
    params.eps_squared = 0.01f;
    params.G = 6.67e-11f;
    
    std::cout << "Running N-Body simulation:\n";
    std::cout << "  Particles: " << params.n_particles << "\n";
    std::cout << "  Steps: " << params.n_steps << "\n";
    std::cout << "  Mode: ";
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    switch (mode) {
        case CPU:
            std::cout << "CPU\n";
            run_cpu_simulation(particles, params);
            break;
        case GPU_NAIVE:
            std::cout << "GPU Naive\n";
            run_gpu_simulation(particles, params, false);
            break;
        case GPU_SHARED:
            std::cout << "GPU Shared Memory\n";
            run_gpu_simulation(particles, params, true);
            break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Total execution time: " << duration.count() << " ms\n";
    
    // Save results
    save_particles(output_file, particles);
    
    return 0;
}