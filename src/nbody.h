// ==================== nbody.h ====================
#ifndef NBODY_H
#define NBODY_H

#include <cuda_runtime.h>
#include <vector>
#include <string>

// Simulation parameters
struct SimulationParams {
    int n_particles;
    int n_steps;
    float dt;
    float eps_squared;
    float G;
};

// Particle structure aligned for efficient GPU memory access
struct Particle {
    float3 position;
    float mass;
    float3 velocity;
    float padding;  // Ensures 32-byte alignment
};

// Function declarations
void run_cpu_simulation(std::vector<Particle>& particles, SimulationParams& params);
void run_gpu_simulation(std::vector<Particle>& particles, SimulationParams& params, bool use_shared);
void load_particles(const std::string& filename, std::vector<Particle>& particles);
void save_particles(const std::string& filename, const std::vector<Particle>& particles);
bool validate_results(const std::vector<Particle>& particles1, const std::vector<Particle>& particles2, float tolerance);

// CUDA error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#endif