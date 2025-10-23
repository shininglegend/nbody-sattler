// ==================== nbody_cpu.cpp ====================
// Reference CPU implementation
#include "nbody.h"
#include <cmath>
#include <chrono>
#include <iostream>

void compute_forces_cpu(std::vector<Particle>& particles, 
                        std::vector<float3>& forces,
                        float G, float eps_squared) {
    int n = particles.size();
    
    // Initialize forces to zero
    for (int i = 0; i < n; i++) {
        forces[i].x = 0.0f;
        forces[i].y = 0.0f;
        forces[i].z = 0.0f;
    }
    
    // Compute pairwise forces
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dx = particles[j].position.x - particles[i].position.x;
            float dy = particles[j].position.y - particles[i].position.y;
            float dz = particles[j].position.z - particles[i].position.z;
            
            float dist_squared = dx*dx + dy*dy + dz*dz + eps_squared;
            float dist_sixth = dist_squared * dist_squared * dist_squared;
            float inv_dist_cube = 1.0f / std::sqrt(dist_sixth);
            
            float f_scalar = G * particles[i].mass * particles[j].mass * inv_dist_cube;
            
            float fx = f_scalar * dx;
            float fy = f_scalar * dy;
            float fz = f_scalar * dz;
            
            // Newton's third law: equal and opposite forces
            forces[i].x += fx;
            forces[i].y += fy;
            forces[i].z += fz;
            
            forces[j].x -= fx;
            forces[j].y -= fy;
            forces[j].z -= fz;
        }
    }
}

void integrate_cpu(std::vector<Particle>& particles,
                  const std::vector<float3>& forces,
                  float dt) {
    for (size_t i = 0; i < particles.size(); i++) {
        // Update velocity: v = v + (F/m) * dt
        particles[i].velocity.x += (forces[i].x / particles[i].mass) * dt;
        particles[i].velocity.y += (forces[i].y / particles[i].mass) * dt;
        particles[i].velocity.z += (forces[i].z / particles[i].mass) * dt;
        
        // Update position: p = p + v * dt
        particles[i].position.x += particles[i].velocity.x * dt;
        particles[i].position.y += particles[i].velocity.y * dt;
        particles[i].position.z += particles[i].velocity.z * dt;
    }
}

void run_cpu_simulation(std::vector<Particle>& particles, SimulationParams& params) {
    std::vector<float3> forces(params.n_particles);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < params.n_steps; step++) {
        compute_forces_cpu(particles, forces, params.G, params.eps_squared);
        integrate_cpu(particles, forces, params.dt);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "CPU Execution Time: " << duration.count() << " ms" << std::endl;
}