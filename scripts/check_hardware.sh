#!/bin/bash
# still-waiting hardware checker
# Verifies Kepler compatibility and reports system status.
# Run as: ./scripts/check_hardware.sh
#
# Built by UnobligatedRascal — Ancient hardware, fresh ambition.

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}!${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }
section() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  still-waiting Hardware Check                           ║"
echo "║  Kepler (sm_37) Compatibility Report                    ║"
echo "╚══════════════════════════════════════════════════════════╝"

section "GPU Hardware"

if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found"
    echo "Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx"
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_major.compute_minor --format=csv,noheader,nounits | while IFS=',' read -r name driver mem_total mem_free cc; do
    name=$(echo "$name" | xargs)
    cc=$(echo "$cc" | xargs)
    mem_total=$(echo "$mem_total" | xargs)
    mem_free=$(echo "$mem_free" | xargs)
    
    echo "GPU: $name"
    echo "  Driver: $driver"
    echo "  Memory: ${mem_free}MB / ${mem_total}MB free"
    echo "  Compute: sm_$cc"
    
    if [[ "$cc" == "3.7" ]]; then
        info "Kepler sm_37 — this project was built for this GPU!"
    elif [[ "$cc" == "3.5" ]]; then
        info "Kepler sm_35 — compatible (adjust TORCH_CUDA_ARCH_LIST)"
    elif [[ "$cc" == "5."* ]]; then
        warn "Maxwell — official PyTorch supports this"
    elif [[ "$cc" == "6."* ]]; then
        warn "Pascal — official PyTorch supports this"
    elif [[ "$cc" == "7."* || "$cc" == "8."* ]]; then
        warn "Modern GPU — official PyTorch supports this"
    else
        warn "Unknown compute capability"
    fi
done

section "CUDA Toolkit"

if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
    info "CUDA version: $nvcc_version"
else
    warn "nvcc not found (CUDA toolkit may not be installed)"
fi

section "CPU and NUMA"

if command -v lscpu &> /dev/null; then
    cpu_model=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
    cores=$(nproc)
    echo "CPU: $cpu_model"
    info "Cores: $cores"
else
    warn "lscpu not available"
fi

numa_nodes=$(numactl --hardware 2>/dev/null | grep -c "^node" || echo "0")
if [[ "$numa_nodes" -gt 1 ]]; then
    info "NUMA nodes: $numa_nodes (NUMA pinning will give ~2.5x speedup)"
else
    warn "Single NUMA node detected (or numactl not available)"
fi

section "Python and PyTorch"

if command -v python3 &> /dev/null; then
    python_ver=$(python3 --version | cut -d' ' -f2)
    info "Python: $python_ver"
    
    if python3 -c "import torch" 2>/dev/null; then
        torch_ver=$(python3 -c "import torch; print(torch.__version__)")
        cuda_avail=$(python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')")
        info "PyTorch: $torch_ver"
        
        if [[ "$cuda_avail" == "yes" ]]; then
            gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
            info "CUDA available: yes"
            info "GPU: $gpu_name"
        else
            warn "CUDA not available in PyTorch (build PyTorch with CUDA support)"
        fi
    else
        warn "PyTorch not installed"
    fi
else
    error "Python3 not found"
fi

section "Build Tools"

for tool in cmake gcc g++ git curl wget; do
    if command -v "$tool" &> /dev/null; then
        ver=$($tool --version 2>/dev/null | head -1 | cut -d' ' -f3-)
        info "$tool: $ver"
    else
        warn "$tool not installed"
    fi
done

if command -v rustc &> /dev/null; then
    info "Rust: $(rustc --version)"
else
    warn "Rust not installed"
fi

if command -v node &> /dev/null; then
    info "Node.js: $(node --version)"
else
    warn "Node.js not installed (needed for GUI build)"
fi

section "Summary"

echo ""
if command -v nvidia-smi &> /dev/null && python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    info "Your system appears ready for still-waiting!"
    echo ""
    echo "Next steps:"
    echo "  1. Clone: git clone https://github.com/UnobligatedRascal/still-waiting.git"
    echo "  2. Setup: cd still-waiting && sudo ./scripts/setup.sh"
    echo "  3. Start: ./deploy/start_still_waiting.sh user"
else
    warn "Some prerequisites are missing. Run ./scripts/setup.sh to install them."
fi

echo ""
echo "Built by UnobligatedRascal — Ancient hardware, fresh ambition."
echo ""
