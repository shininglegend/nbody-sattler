#!/usr/bin/env python3
"""
Generate random particle data for N-Body simulation testing
"""
import numpy as np
import sys

def generate_particles(n_particles, output_file):
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate random positions in [-1, 1]^3
    positions = np.random.uniform(-1, 1, (n_particles, 3))
    
    # Generate random masses in [0.5, 1.5]
    masses = np.random.uniform(0.5, 1.5, n_particles)
    
    # Generate small random velocities
    velocities = np.random.uniform(-0.1, 0.1, (n_particles, 3))
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(f"{n_particles}\n")
        for i in range(n_particles):
            f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f} ")
            f.write(f"{masses[i]:.6f} ")
            f.write(f"{velocities[i,0]:.6f} {velocities[i,1]:.6f} {velocities[i,2]:.6f}\n")
    
    print(f"Generated {n_particles} particles in {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <n_particles> <output_file>")
        sys.exit(1)
    
    n_particles = int(sys.argv[1])
    output_file = sys.argv[2]
    generate_particles(n_particles, output_file)