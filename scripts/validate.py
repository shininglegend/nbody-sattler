# ==================== validate.py ====================
#!/usr/bin/env python3
"""
Python validation script for comparing N-Body simulation outputs
"""
import sys
import numpy as np

def load_particles(filename):
    with open(filename, 'r') as f:
        n_particles = int(f.readline())
        particles = []
        for _ in range(n_particles):
            data = list(map(float, f.readline().split()))
            particles.append(data)
    return np.array(particles)

def validate(file1, file2, tolerance=1e-5):
    particles1 = load_particles(file1)
    particles2 = load_particles(file2)
    
    if particles1.shape != particles2.shape:
        print(f"✗ FAILED: Shape mismatch {particles1.shape} vs {particles2.shape}")
        return False
    
    # Check positions (columns 0-2)
    pos_diff = np.abs(particles1[:, 0:3] - particles2[:, 0:3])
    max_pos_diff = np.max(pos_diff)
    
    # Check velocities (columns 4-6)
    vel_diff = np.abs(particles1[:, 4:7] - particles2[:, 4:7])
    max_vel_diff = np.max(vel_diff)
    
    print(f"Max position difference: {max_pos_diff:.2e}")
    print(f"Max velocity difference: {max_vel_diff:.2e}")
    
    if max_pos_diff <= tolerance and max_vel_diff <= tolerance:
        print(f"✓ PASSED: Results match within tolerance ({tolerance})")
        return True
    else:
        print(f"✗ FAILED: Results differ beyond tolerance ({tolerance})")
        return False

def calculate_energy(particles):
    """Calculate total energy for conservation check"""
    n = len(particles)
    kinetic = 0.0
    potential = 0.0
    G = 6.67e-11
    eps_squared = 0.01
    
    for i in range(n):
        mass = particles[i, 3]
        vel = particles[i, 4:7]
        kinetic += 0.5 * mass * np.sum(vel**2)
        
        for j in range(i+1, n):
            r = particles[j, 0:3] - particles[i, 0:3]
            dist = np.sqrt(np.sum(r**2) + eps_squared)
            potential -= G * mass * particles[j, 3] / dist
    
    return kinetic + potential

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file1> <file2>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    success = validate(file1, file2)
    
    # Check energy conservation
    particles1 = load_particles(file1)
    energy = calculate_energy(particles1)
    print(f"Total energy: {energy:.2e} J")
    
    sys.exit(0 if success else 1)