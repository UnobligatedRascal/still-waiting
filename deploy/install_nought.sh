#!/bin/bash
# still-waiting: Native install script for NOUGHT (Debian Bookworm/Q4OS)
# Run as: sudo ./install_nought.sh
# UnobligatedRascal — Making old hardware sing.

set -e

echo "=== still-waiting Installer for NOUGHT ==="
echo "Hardware: 8x Tesla K80 GK210 sm_37, Dual Xeon E5-2697v4, 128GB ECC"
echo ""

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
    echo "  ERROR: CUDA not found. Expected CUDA 11.8 at /usr/local/cuda-11.8"
    exit 1
fi
echo "  CUDA: $(nvcc --version | head -1)"

# Check GPU access
echo "  GPUs:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | head -8

# ============================================
# Build Rust orchestrator
# ============================================
echo "[3/5] Building Rust orchestrator..."
cd /home/whistler/still-waiting/orchestrator
cargo build --release
echo "  Built: ./target/release/agent-orchestrator"

# ============================================
# Python environment (Kepler-compatible PyTorch)
# ============================================
echo "[4/5] Setting up Python training environment..."

# Create venv
PYTHON_VENV="/home/whistler/.venvs/still-waiting"
python3 -m venv "$PYTHON_VENV"
source "$PYTHON_VENV/bin/activate"
pip install --upgrade pip

# CRITICAL: Install Kepler-compatible PyTorch 1.14.0+cu118
# Default pip wheels dropped sm_37 support in PyTorch 2.0+
echo "  Installing PyTorch 1.14.0+cu118 (last sm_37 support)..."
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
mkdir -p /data/checkpoints
mkdir -p /data/models
mkdir -p /data/training-data

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To run the orchestrator:"
echo "  cd /home/whistler/still-waiting"
echo "  ./orchestrator/target/release/agent-orchestrator"
echo ""
echo "To run a training job (example):"
echo "  NUMA_NODE=0 CUDA_VISIBLE_DEVICES=0,1,2,3 \\"
echo "    numactl --cpunodebind=0 --membind=0 \\"
echo "    torchrun --nproc_per_node=4 --master_port=29500 \\"
echo "    python3 -m venv /home/whistler/.venvs/still-waiting \\"
echo "    /home/whistler/.venvs/still-waiting/bin/python python/worker.py <job_id> <config.json>"
echo ""
echo "API available at: http://0.0.0.0:8000/v1/training/jobs"
