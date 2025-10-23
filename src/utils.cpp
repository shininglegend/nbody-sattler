// ==================== utils.cpp ====================
// Utility functions for loading, saving, and validating particles
#include "nbody.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>

void load_particles(const std::string& filename, std::vector<Particle>& particles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    int n_particles;
    file >> n_particles;
    particles.resize(n_particles);
    
    for (int i = 0; i < n_particles; i++) {
        file >> particles[i].position.x >> particles[i].position.y >> particles[i].position.z;
        file >> particles[i].mass;
        file >> particles[i].velocity.x >> particles[i].velocity.y >> particles[i].velocity.z;
        particles[i].padding = 0.0f;
    }
    
    file.close();
}

void save_particles(const std::string& filename, const std::vector<Particle>& particles) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        exit(1);
    }
    
    file << particles.size() << std::endl;
    file << std::fixed << std::setprecision(6);
    
    for (const auto& p : particles) {
        file << p.position.x << " " << p.position.y << " " << p.position.z << " ";
        file << p.mass << " ";
        file << p.velocity.x << " " << p.velocity.y << " " << p.velocity.z << std::endl;
    }
    
    file.close();
}

bool validate_results(const std::vector<Particle>& particles1, 
                     const std::vector<Particle>& particles2, 
                     float tolerance) {
    if (particles1.size() != particles2.size()) {
        std::cerr << "Error: Particle count mismatch!" << std::endl;
        return false;
    }
    
    float max_pos_diff = 0.0f;
    float max_vel_diff = 0.0f;
    int max_pos_idx = -1;
    int max_vel_idx = -1;
    
    for (size_t i = 0; i < particles1.size(); i++) {
        // Position difference
        float dx = particles1[i].position.x - particles2[i].position.x;
        float dy = particles1[i].position.y - particles2[i].position.y;
        float dz = particles1[i].position.z - particles2[i].position.z;
        float pos_diff = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (pos_diff > max_pos_diff) {
            max_pos_diff = pos_diff;
            max_pos_idx = i;
        }
        
        // Velocity difference
        float dvx = particles1[i].velocity.x - particles2[i].velocity.x;
        float dvy = particles1[i].velocity.y - particles2[i].velocity.y;
        float dvz = particles1[i].velocity.z - particles2[i].velocity.z;
        float vel_diff = std::sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
        
        if (vel_diff > max_vel_diff) {
            max_vel_diff = vel_diff;
            max_vel_idx = i;
        }
    }
    
    std::cout << "Validation Results:" << std::endl;
    std::cout << "  Max position difference: " << max_pos_diff 
              << " (particle " << max_pos_idx << ")" << std::endl;
    std::cout << "  Max velocity difference: " << max_vel_diff 
              << " (particle " << max_vel_idx << ")" << std::endl;
    
    bool passed = (max_pos_diff <= tolerance && max_vel_diff <= tolerance);
    
    if (passed) {
        std::cout << "✓ PASSED: Results match within tolerance (" << tolerance << ")" << std::endl;
    } else {
        std::cout << "✗ FAILED: Results differ beyond tolerance (" << tolerance << ")" << std::endl;
    }
    
    return passed;
}
