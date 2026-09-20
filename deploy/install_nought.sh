#!/bin/bash
# still-waiting: Native install script for Debian-based systems
# Run as: sudo ./install_nought.sh
# UnobligatedRascal — Making old hardware sing.

set -e

echo "=== still-waiting Installer ==="
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================
# Rust installation
# ============================================
echo "[1/5] Checking Rust..."
if ! command -v cargo &>/dev/null; then
    echo "  Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "  Rust already installed: $(cargo --version)"
fi

# ============================================
# CUDA verification
# ============================================
echo "[2/5] Verifying CUDA..."
if ! command -v nvcc &>/dev/null; then
    echo "  WARNING: CUDA not found in PATH"
    echo "  Expected CUDA toolkit at /usr/local/cuda"
    echo "  Install CUDA 11.8 or compatible:"
    echo "  https://developer.nvidia.com/cuda-toolkit-archive"
else
    echo "  CUDA: $(nvcc --version | head -1)"
fi

# Check GPU access
if command -v nvidia-smi &>/dev/null; then
    echo "  GPUs:"
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | head -8
fi

# ============================================
# Build Rust orchestrator
# ============================================
echo "[3/5] Building Rust orchestrator..."
cd "$PROJECT_DIR/orchestrator"
cargo build --release
echo "  Built: ./target/release/agent-orchestrator"

# ============================================
# Python environment
# ============================================
echo "[4/5] Setting up Python training environment..."

# Create venv next to project (customizable via PYTHON_VENV env var)
PYTHON_VENV="${PYTHON_VENV:-$PROJECT_DIR/.venv}"
python3 -m venv "$PYTHON_VENV"
source "$PYTHON_VENV/bin/activate"
pip install --upgrade pip

# CRITICAL for Kepler: Install Kepler-compatible PyTorch
# PyTorch 1.14.0+cu118 is last official wheel with sm_37 support
# For newer hardware, use pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo "  Installing PyTorch 1.14.0+cu118 (sm_37 support)..."
pip install torch==1.14.0+cu118 torchvision==0.15.0+cu118 torchaudio==0.14.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Transformers + training stack
echo "  Installing transformers stack..."
pip install \
    transformers==4.30.2 \
    peft==0.4.0 \
    accelerate==0.20.3 \
    datasets==2.14.6 \
    trl==0.4.7 \
    requests \
    tensorboard

# Verify PyTorch CUDA
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

deactivate

# ============================================
# Data directories
# ============================================
echo "[5/5] Creating data directories..."
mkdir -p "$PROJECT_DIR/data/checkpoints"
mkdir -p "$PROJECT_DIR/data/models"
mkdir -p "$PROJECT_DIR/data/training-data"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Project: $PROJECT_DIR"
echo "Venv: $PYTHON_VENV"
echo ""
echo "To run the orchestrator:"
echo "  cd $PROJECT_DIR"
echo "  ./orchestrator/target/release/agent-orchestrator"
echo ""
echo "To run a training job (example):"
echo "  source $PYTHON_VENV/bin/activate"
echo "  NUMA_NODE=0 CUDA_VISIBLE_DEVICES=0,1,2,3 \\"
echo "    numactl --cpunodebind=0 --membind=0 \\"
echo "    torchrun --nproc_per_node=4 --master_port=29500 \\"
echo "    python3 $PROJECT_DIR/python/worker.py <job_id> <config.json>"
echo ""
echo "API available at: http://0.0.0.0:9999/v1/training/jobs"
