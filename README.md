# Assignment 2: N-Body Simulation on the GPU with CUDA

**CS 420 - Parallel Computing**  
**Release Date:** September 30, 2025  
**Due Date:** October 21, 2025, 11:59 PM  
**Points:** 25  

## Overview

In this assignment, you will implement a gravitational N-body simulation on the GPU using CUDA. The N-body problem simulates the motion of N particles under mutual gravitational attraction, where each particle exerts a force on every other particle. With O(N²) computational complexity, this problem is an ideal candidate for massive parallelization on a GPU.

You will progress through three implementation phases:
1. **Naive GPU Implementation:** Basic parallel version with global memory access
2. **Shared Memory Optimization:** Advanced version utilizing GPU shared memory
3. **Performance Analysis:** Profiling and benchmarking your implementations

## Learning Objectives

Upon completing this assignment, you will be able to:
- Implement massively parallel algorithms using the CUDA programming model
- Manage host-device memory transfers efficiently
- Optimize GPU kernels using shared memory to reduce global memory bandwidth requirements
- Use NVIDIA profiling tools to identify and resolve performance bottlenecks
- Analyze and document GPU performance characteristics

## Background: The N-Body Problem

### Physics Model

Each particle has:
- Position: (x, y, z)
- Velocity: (vx, vy, vz)  
- Mass: m

The gravitational force between particles i and j:
```
F_ij = G * (m_i * m_j) / r_ij²
```

Where:
- G is the gravitational constant (use G = 6.67e-11)
- r_ij is the distance between particles
- A softening factor ε² = 0.01 prevents singularities when particles are very close

### Integration Method

Use the simple Euler integration scheme:
```
v_new = v_old + a * dt
p_new = p_old + v_new * dt
```

Where acceleration a = F/m and dt is the time step (use dt = 0.01).

## Provided Framework

### Directory Structure
```
nbody/
├── Makefile
├── src/
│   ├── nbody.h           # Data structures and constants
│   ├── nbody_cpu.cpp     # Reference CPU implementation
│   ├── nbody_gpu.cu      # Your GPU implementation (template)
│   ├── main.cpp          # Driver program
│   └── utils.cpp         # Utility functions
├── tests/
│   ├── test_small.txt    # 256 particles
│   ├── test_medium.txt   # 2048 particles
│   └── test_large.txt    # 16384 particles
├── scripts/
│   ├── generate_input.py # Generate test cases
│   └── validate.py       # Validate output correctness
└── README.md
```

### Data Structures

```cpp
// nbody.h
struct Particle {
    float3 position;
    float mass;
    float3 velocity;
    float padding;  // Ensures alignment
};

struct SimulationParams {
    int n_particles;
    int n_steps;
    float dt;
    float eps_squared;
    float G;
};
```

## Implementation Requirements

### Phase 1: Naive GPU Implementation (10 points)

Implement the following kernel in `nbody_gpu.cu`:

```cuda
__global__ void nbody_kernel_naive(
    Particle* particles_in,
    Particle* particles_out,
    int n_particles,
    float dt,
    float eps_squared,
    float G
) {
    // Each thread computes forces for one particle
    // Iterate through all other particles in global memory
}
```

Requirements:
- Each thread handles one particle
- Read all particle data from global memory
- Compute total force from all other particles
- Update velocity and position
- Write results to output buffer

### Phase 2: Shared Memory Optimization (10 points)

Implement an optimized kernel using shared memory tiling:

```cuda
__global__ void nbody_kernel_shared(
    Particle* particles_in,
    Particle* particles_out,
    int n_particles,
    float dt,
    float eps_squared,
    float G
) {
    extern __shared__ Particle shared_particles[];
    
    // Tiled force calculation using shared memory
    // Load tiles of particles cooperatively
    // Compute interactions within tiles
}
```

Requirements:
- Use shared memory to cache particle data
- Implement tiling strategy (recommended tile size: 256 particles)
- Ensure coalesced memory access patterns
- Handle boundary conditions correctly

### Phase 3: Performance Analysis (5 points)

Profile your implementations and collect the following metrics:
- Kernel execution time
- Memory bandwidth utilization
- Global memory load/store efficiency
- Shared memory bank conflicts (for optimized version)
- Occupancy

Use NVIDIA Nsight Compute:
```bash
ncu --set full ./nbody_gpu test_large.txt 100
```

## Correctness Testing

### Self-Test Framework

The provided framework includes automatic correctness validation:

```bash
# Compile all versions
make all

# Run correctness tests
./scripts/run_tests.sh

# Validate output against reference
python3 scripts/validate.py output_gpu.txt output_cpu.txt
```

### Correctness Criteria
- Final particle positions must match CPU reference within tolerance (1e-5)
- Energy should be approximately conserved (< 1% drift over 100 steps)

## Performance Benchmarking

### Required Measurements

Measure and report performance for:
1. Serial CPU implementation (baseline)
2. Naive GPU kernel
3. Shared memory optimized GPU kernel

For each implementation, measure:
- Execution time for 100 simulation steps
- GFLOPS achieved
- Speedup relative to CPU baseline

### Test Cases
- Small: 256 particles (debugging)
- Medium: 2048 particles (correctness)
- Large: 16384 particles (performance)

## Deliverables

### 1. Code 
- `nbody_gpu.cu` with both kernel implementations
- Clean, well-commented code
- Proper error checking for CUDA calls

### 2. Performance Report 

Submit a PDF report (3-4 pages) containing:

#### Section 1: Implementation Details
- Description of your tiling strategy
- Shared memory usage calculation
- Any optimizations beyond the basic requirements

#### Section 2: Performance Analysis
- Performance comparison table:
  ```
  | Implementation | N=2048 Time | N=16384 Time | Speedup |
  |----------------|-------------|--------------|---------|
  | CPU Serial     |             |              |   1.0x  |
  | GPU Naive      |             |              |         |
  | GPU Shared Mem |             |              |         |
  ```
- Graphs showing:
  - Speedup vs. problem size
  - Memory bandwidth utilization
  
#### Section 3: Profiler Analysis
- Screenshots from Nsight Compute showing:
  - Memory access patterns
  - Kernel occupancy
  - Bottleneck analysis
- Explanation of how shared memory reduces global memory traffic

### 3. Written Questions 

Answer the following:
1. Calculate the theoretical memory bandwidth required for the naive implementation with N=16384 particles. How does this compare to your GPU's peak bandwidth?
2. Why does the shared memory optimization provide better performance? Quantify the reduction in global memory accesses.
3. What is the optimal tile size for your GPU? How did you determine this?

## Compilation and Execution

### Compilation
```bash
# Compile all versions
make all

# Compile only GPU version
make gpu

# Clean build files
make clean
```

### Execution
```bash
# Run GPU version
./nbody_gpu <input_file> <n_steps> <output_file>

# Example
./nbody_gpu tests/test_large.txt 100 output_gpu.txt

# Run with profiler
nvprof ./nbody_gpu tests/test_large.txt 100 output_gpu.txt
```

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Naive GPU Implementation** | 10 | - Correct force calculation <br>- Proper memory management <br>- Correct integration  |
| **Shared Memory Implementation** | 10 | - Correct tiling strategy <br>- Efficient shared memory usage <br>- Handles edge cases  |
| **Performance & Report** | 5 | - Achieves >10x speedup for large dataset <br>- Profiler usage and analysis - Clear explanations <br>- Complete analysis (5pts) |

### Performance Targets
- Naive GPU: >5x speedup over CPU for N=16384
- Shared Memory GPU: >10x speedup over CPU for N=16384

## Submission Instructions

Submit via Populi a single zip file containing:
```
username_a2.zip
├── src/
│   └── nbody_gpu.cu      # Your implementation
├── report.pdf            # Performance report
├── profiler_output/      # Nsight screenshots
└── README.txt            # Build/run instructions
```

## Tips and Hints

1. **Start Simple:** Get the naive version working correctly before optimizing
2. **Thread Organization:** Use a 1D grid and 1D blocks for simplicity
3. **Shared Memory Size:** Remember the 48KB/64KB limit per SM
4. **Bank Conflicts:** Padding shared memory arrays can reduce conflicts
5. **Warp Divergence:** All threads in a warp should follow the same execution path
6. **Error Checking:** Always check CUDA API calls:
   ```cuda
   #define CHECK_CUDA(call) do { \
       cudaError_t err = call; \
       if (err != cudaSuccess) { \
           printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(err)); \
           exit(1); \
       } \
   } while(0)
   ```

## Academic Integrity

This is an individual assignment. You may discuss high-level approaches with classmates, but all code must be your own. Using code from online sources without attribution or submitting another student's work will result in a zero grade and academic integrity violation report.

## Office Hours and Support

- Office Hours: Friday 1:00-2:00 PM or by appointment
- Populi: Post general questions (no code snippets)
- Email: For personal/grading questions

## Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- Lecture slides and examples from Weeks 5-8

---
*Note: Late submissions will be penalized 10% per day up to 3 days. Submissions more than 3 days late will not be accepted.*