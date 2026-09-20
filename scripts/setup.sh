#!/bin/bash
# still-waiting automated setup script
# Detects Kepler hardware, installs dependencies, builds everything.
# Run as: ./scripts/setup.sh
#
# Built by UnobligatedRascal — Ancient hardware, fresh ambition.

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${GREEN}[setup]${NC} $1"; }
warn() { echo -e "${YELLOW}[setup]${NC} $1"; }
error() { echo -e "${RED}[setup]${NC} $1"; }
step() { echo -e "${BLUE}[step]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --skip-pytorch    Skip PyTorch build (use pre-built or existing install)"
    echo "  --skip-gui        Skip GUI build"
    echo "  --skip-orch       Skip orchestrator build"
    echo "  --pytorch-version VERSION   PyTorch version tag (default: v2.4.0-rc8)"
    echo "  --help            Show this help"
    exit 0
}

check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        error "This script requires root (sudo) for system package installation."
        error "Run: sudo $0 \$@"
        exit 1
    fi
}

check_kepler() {
    step "Checking for Kepler hardware..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. NVIDIA drivers not installed?"
        exit 1
    fi
    
    local gpu_info=$(nvidia-smi --query-gpu=name,compute_major.compute_minor --format=csv,noheader | head -1)
    local compute_cap=$(echo "$gpu_info" | awk '{print $2}')
    
    if [[ "$compute_cap" == "3.7" ]]; then
        info "Detected Kepler GPU (sm_37) — this project was built for your hardware!"
        return 0
    elif [[ "$compute_cap" == "3.5" ]]; then
        warn "Detected sm_35 (older Kepler). PyTorch build should work with TORCH_CUDA_ARCH_LIST=\"3.5 3.7\"."
        return 0
    elif [[ "$compute_cap" == "5."* ]]; then
        warn "Detected Maxwell (sm_5x). Consider using official PyTorch instead."
        return 0
    elif [[ "$compute_cap" == "6."* ]]; then
        warn "Detected Pascal (sm_6x). Official PyTorch supports your hardware."
        return 0
    elif [[ "$compute_cap" == "7."* || "$compute_cap" == "8."* || "$compute_cap" == "9."* ]]; then
        info "Detected modern GPU (sm_7+). Official PyTorch supports your hardware."
        return 0
    else
        warn "Unknown compute capability: $compute_cap"
        return 0
    fi
}

check_cuda() {
    step "Checking CUDA installation..."
    
    local cuda_version=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "not found")
    
    if [[ "$cuda_version" == "not found" ]]; then
        warn "CUDA not found or nvcc not in PATH."
        warn "You can install CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive"
        read -p "Continue without CUDA check? (y/N): " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
        return 0
    fi
    
    info "CUDA version: $cuda_version"
    
    if [[ "$cuda_version" == "11.6" || "$cuda_version" == "11.5" || "$cuda_version" == "11.4" || "$cuda_version" == "11.3" || "$cuda_version" == "11.2" || "$cuda_version" == "11.1" || "$cuda_version" == "11.0" || "$cuda_version" == "10."* ]]; then
        warn "CUDA $cuda_version detected. PyTorch 2.4 Kepler build tested with CUDA 11.8."
        warn "You may need to adjust build commands."
    fi
}

install_system_deps() {
    step "Installing system dependencies..."
    
    # Detect package manager
    local pkg_mgr=""
    if command -v apt-get &> /dev/null; then
        pkg_mgr="apt"
    elif command -v dnf &> /dev/null; then
        pkg_mgr="dnf"
    elif command -v yum &> /dev/null; then
        pkg_mgr="yum"
    elif command -v pacman &> /dev/null; then
        pkg_mgr="pacman"
    else
        error "Unsupported package manager. Please install dependencies manually:"
        error "  build-essential, cmake, gcc-11, g++-11, python3, python3-pip, python3-venv"
        exit 1
    fi
    
    info "Using package manager: $pkg_mgr"
    
    case "$pkg_mgr" in
        apt)
            apt-get update -qq
            apt-get install -y -qq \
                build-essential cmake g++-11 gcc-11 \
                python3 python3-dev python3-pip python3-venv \
                git curl wget \
                libopenblas-dev liblapack-dev \
                numactl ;;
        dnf|yum)
            $pkg_mgr install -y \
                gcc gcc-c++ cmake make \
                python3 python3-devel python3-pip python3-virtualenv \
                git curl wget \
                openblas-devel lapack-devel \
                numactl ;;
        pacman)
            pacman -S --noconfirm \
                base-devel cmake gcc \
                python python-pip \
                git curl wget \
                openblas lapack \
                numactl ;;
    esac
    
    info "System dependencies installed."
}

setup_rust() {
    step "Checking Rust installation..."
    
    if command -v rustc &> /dev/null; then
        info "Rust already installed: $(rustc --version)"
        return 0
    fi
    
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
    info "Rust installed: $(rustc --version)"
    
    # Add to current session
    export PATH="$HOME/.cargo/env:$PATH"
}

setup_python_env() {
    step "Setting up Python environment..."
    
    local venv_path="${PROJECT_DIR}/.venv"
    
    if [ ! -d "$venv_path" ]; then
        info "Creating virtual environment..."
        python3 -m venv "$venv_path"
    fi
    
    . "$venv_path/bin/activate"
    pip install --upgrade pip setuptools wheel
    
    info "Installing Python dependencies..."
    if [ -f "${PROJECT_DIR}/python/requirements.txt" ]; then
        pip install -r "${PROJECT_DIR}/python/requirements.txt"
    else
        # Core deps
        pip install "transformers>=4.40.0" "peft>=0.7.0" datasets accelerate
    fi
    
    info "Python environment ready: $venv_path"
    info "Activate with: source $venv_path/bin/activate"
}

build_pytorch() {
    local version="${1:-v2.4.0-rc8}"
    step "Building PyTorch $version for Kepler (sm_37)..."
    info "This will take 2-4 hours on typical hardware."
    
    local pytorch_dir="${HOME}/pytorch-kepler"
    
    if [ -d "$pytorch_dir" ]; then
        warn "Existing pytorch-kepler directory found: $pytorch_dir"
        read -p "Remove and re-clone? (y/N): " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] && rm -rf "$pytorch_dir"
    fi
    
    if [ ! -d "$pytorch_dir" ]; then
        info "Cloning PyTorch..."
        git clone --depth 1 --branch "$version" https://github.com/pytorch/pytorch.git "$pytorch_dir"
    fi
    
    cd "$pytorch_dir"
    git checkout "$version"
    
    # Create venv for build
    local build_venv="${pytorch_dir}/.venv-build"
    python3 -m venv "$build_venv"
    . "$build_venv/bin/activate"
    pip install numpy typing_extensions setuptools wheel
    
    info "Starting PyTorch build..."
    export CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11"
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST="3.7"
    export MAX_JOBS=$(nproc)
    export USE_CUDA=1
    export CUDA_HOME=/usr/local/cuda-11.8
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
    export CUDAFLAGS="-ccbin /usr/bin/g++-11"
    export CUDAHOSTCXX=/usr/bin/g++-11
    export CXX=/usr/bin/g++-11
    export CC=/usr/bin/gcc-11
    export CMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"
    
    python3 setup.py develop
    
    # Verify
    python3 -c "import torch; print(f'PyTorch {torch.__version__} built successfully')"
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    info "PyTorch build complete! Installed in: $build_venv"
}

build_orchestrator() {
    step "Building orchestrator..."
    
    . "$HOME/.cargo/env"
    cd "${PROJECT_DIR}/orchestrator"
    cargo build --release
    
    info "Orchestrator built: orchestrator/target/release/agent-orchestrator"
}

build_gui() {
    step "Building GUI..."
    
    if ! command -v node &> /dev/null; then
        warn "Node.js not found. Skipping GUI build."
        warn "Install Node.js 20+ and run: cd gui && npm install && npm run build"
        return 0
    fi
    
    cd "${PROJECT_DIR}/gui"
    npm install
    npm run build
    
    info "GUI built: gui/dist/"
}

start_services() {
    step "Starting services..."
    
    cd "$PROJECT_DIR"
    
    # Build GUI first if needed (for static files)
    if [ ! -d "gui/dist" ] && command -v node &> /dev/null; then
        cd gui && npm run build && cd ..
    fi
    
    # Kill existing
    pkill -f agent-orchestrator || true
    sleep 1
    
    # Start orchestrator
    cd orchestrator
    ORCH_STATIC_DIR="${PROJECT_DIR}/gui/dist" nohup ./target/release/agent-orchestrator \
        > "${PROJECT_DIR}/logs/orchestrator.log" 2>&1 &
    local pid=$!
    
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        local ip=$(hostname -I | awk '{print $1}')
        info "Orchestrator started (PID $pid)"
        info "GUI: http://${ip}:9999/"
        info "API: http://${ip}:9999/v1/"
    else
        error "Orchestrator failed to start. Check logs:"
        error "  tail -f ${PROJECT_DIR}/logs/orchestrator.log"
        exit 1
    fi
}

main() {
    local skip_pytorch=false
    local skip_gui=false
    local skip_orch=false
    local pytorch_version="v2.4.0-rc8"
    local start=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-pytorch) skip_pytorch=true; shift ;;
            --skip-gui) skip_gui=true; shift ;;
            --skip-orch) skip_orch=true; shift ;;
            --pytorch-version) pytorch_version="$2"; shift 2 ;;
            --start) start=true; shift ;;
            --help) usage ;;
            *) error "Unknown option: $1"; usage ;;
        esac
    done
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  still-waiting Setup                                    ║"
    echo "║  Private LLM training on Kepler hardware                ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    
    check_root
    check_kepler
    check_cuda
    install_system_deps
    setup_rust
    
    if [ "$skip_pytorch" = false ]; then
        # Check if PyTorch with CUDA is already available
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            info "Existing PyTorch with CUDA detected. Skipping build."
            info "Use --skip-pytorch explicitly if you want to force rebuild."
        else
            build_pytorch "$pytorch_version"
        fi
    else
        info "Skipping PyTorch build (--skip-pytorch)."
    fi
    
    setup_python_env
    
    if [ "$skip_orch" = false ]; then
        build_orchestrator
    fi
    
    if [ "$skip_gui" = false ]; then
        build_gui
    fi
    
    # Create logs dir
    mkdir -p "${PROJECT_DIR}/logs"
    
    echo ""
    info "Setup complete!"
    echo ""
    info "To start:"
    info "  cd ${PROJECT_DIR}"
    info "  source .venv/bin/activate"
    info "  ./deploy/start_still_waiting.sh user"
    echo ""
    info "Or run with --start next time for automatic launch."
    echo ""
    info "Built by UnobligatedRascal — Ancient hardware, fresh ambition."
    echo ""
    
    if [ "$start" = true ]; then
        start_services
    fi
}

main "$@"
