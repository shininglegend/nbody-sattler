// ==================== nbody_gpu.cu ====================
#include "nbody.h"
#include <stdio.h>
#include <math.h>

// Constants for GPU execution
#define BLOCK_SIZE 256
#define TILE_SIZE 256

// Helper function to compute force between two particles
__device__ float3 compute_force(const Particle& p1, const Particle& p2, float G, float eps_squared) {
    float3 r;
    r.x = p2.position.x - p1.position.x;
    r.y = p2.position.y - p1.position.y;
    r.z = p2.position.z - p1.position.z;
    
    float dist_squared = r.x * r.x + r.y * r.y + r.z * r.z + eps_squared;
    float dist_sixth = dist_squared * dist_squared * dist_squared;
    float inv_dist_cube = rsqrtf(dist_sixth);  // 1/sqrt(dist^6) = 1/dist^3
    
    float f_scalar = G * p1.mass * p2.mass * inv_dist_cube;
    
    float3 force;
    force.x = f_scalar * r.x;
    force.y = f_scalar * r.y;
    force.z = f_scalar * r.z;
    
    return force;
}

// ========== PHASE 1: NAIVE GPU IMPLEMENTATION ==========
// TODO: Implement this kernel
// Each thread should compute forces for one particle by iterating through all others
__global__ void nbody_kernel_naive(
    Particle* particles_in,
    Particle* particles_out,
    int n_particles,
    float dt,
    float eps_squared,
    float G
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    
    // TODO: Student implementation
    // 1. Load particle i data
    // 2. Initialize force accumulator to zero
    // 3. Loop through all particles j != i
    //    - Compute pairwise force
    //    - Accumulate forces
    // 4. Update velocity: v_new = v_old + (F/m) * dt
    // 5. Update position: p_new = p_old + v_new * dt
    // 6. Write updated particle to particles_out
    
    // Hint: Use the compute_force helper function
    Particle p = particles_in[tid];
    
    float3 total_force = {0.0f, 0.0f, 0.0f};
    
    // Your code here...
    
    // Example structure (to be completed by student):
    // for (int j = 0; j < n_particles; j++) {
    //     if (j != tid) {
    //         Particle other = particles_in[j];
    //         float3 force = compute_force(p, other, G, eps_squared);
    //         total_force.x += force.x;
    //         total_force.y += force.y;
    //         total_force.z += force.z;
    //     }
    // }
    
    // Update velocity and position
    // ...
    
    // Write result
    // particles_out[tid] = p;
}


// ========== PHASE 2: SHARED MEMORY OPTIMIZATION ==========
// TODO: Implement this kernel using shared memory tiling
__global__ void nbody_kernel_shared(
    Particle* particles_in,
    Particle* particles_out,
    int n_particles,
    float dt,
    float eps_squared,
    float G
) {
    extern __shared__ Particle shared_particles[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    if (tid >= n_particles) return;
    
    // TODO: Student implementation
    // 1. Load particle i data
    // 2. Initialize force accumulator
    // 3. Loop through tiles:
    //    a. Cooperatively load TILE_SIZE particles into shared memory
    //    b. Synchronize threads
    //    c. Compute forces with particles in shared memory
    //    d. Synchronize before next tile
    // 4. Update velocity and position
    // 5. Write result
    
    // Your code here...
}

// Host function to launch GPU simulation
void run_gpu_simulation(std::vector<Particle>& particles, SimulationParams& params, bool use_shared) {
    int n_particles = params.n_particles;
    size_t size = n_particles * sizeof(Particle);
    
    // Allocate device memory
    Particle *d_particles_in, *d_particles_out;
    CHECK_CUDA(cudaMalloc(&d_particles_in, size));
    CHECK_CUDA(cudaMalloc(&d_particles_out, size));
    
    // Copy particles to device
    CHECK_CUDA(cudaMemcpy(d_particles_in, particles.data(), size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n_particles + threads_per_block - 1) / threads_per_block;
    
    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Run simulation
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int step = 0; step < params.n_steps; step++) {
        if (use_shared) {
            size_t shared_mem_size = TILE_SIZE * sizeof(Particle);
            nbody_kernel_shared<<<blocks, threads_per_block, shared_mem_size>>>(
                d_particles_in, d_particles_out, n_particles,
                params.dt, params.eps_squared, params.G
            );
        } else {
            nbody_kernel_naive<<<blocks, threads_per_block>>>(
                d_particles_in, d_particles_out, n_particles,
                params.dt, params.eps_squared, params.G
            );
        }
        
        // Swap buffers
        Particle* temp = d_particles_in;
        d_particles_in = d_particles_out;
        d_particles_out = temp;
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("GPU Execution Time (%s): %.3f ms\n", 
           use_shared ? "Shared Memory" : "Naive", milliseconds);
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(particles.data(), d_particles_in, size, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_particles_in));
    CHECK_CUDA(cudaFree(d_particles_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}