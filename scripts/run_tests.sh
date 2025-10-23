// ==================== run_tests.sh ====================
#!/bin/bash
# Self-testing script for N-Body simulation

echo "========================================="
echo "N-Body Simulation Self-Test Suite"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to run test
run_test() {
    local test_name=$1
    local n_particles=$2
    local n_steps=$3
    
    echo ""
    echo "Test: $test_name (N=$n_particles, Steps=$n_steps)"
    echo "-----------------------------------------"
    
    # Generate test input
    python3 scripts/generate_particles.py $n_particles test_input.txt
    
    # Run CPU version (reference)
    echo "Running CPU version..."
    ./nbody_gpu test_input.txt $n_steps cpu_output.txt --cpu
    
    # Run GPU naive version
    echo "Running GPU naive version..."
    ./nbody_gpu test_input.txt $n_steps gpu_naive_output.txt --gpu-naive
    
    # Run GPU shared memory version
    echo "Running GPU shared memory version..."
    ./nbody_gpu test_input.txt $n_steps gpu_shared_output.txt --gpu-shared
    
    # Validate results
    echo ""
    echo "Validating GPU naive vs CPU:"
    python3 scripts/validate.py gpu_naive_output.txt cpu_output.txt
    
    echo ""
    echo "Validating GPU shared vs CPU:"
    python3 scripts/validate.py gpu_shared_output.txt cpu_output.txt
    
    # Cleanup
    rm -f test_input.txt cpu_output.txt gpu_naive_output.txt gpu_shared_output.txt
}

# Check if executable exists
if [ ! -f "nbody_gpu" ]; then
    echo -e "${RED}Error: nbody_gpu not found. Please run 'make all' first.${NC}"
    exit 1
fi

# Run test suite
echo "Starting test suite..."

# Small test for correctness
run_test "Small Correctness Test" 256 10

# Medium test
run_test "Medium Test" 2048 10

# Performance test
run_test "Performance Test" 16384 100

echo ""
echo "========================================="
echo "Test suite completed!"
echo "========================================="