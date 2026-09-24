#!/usr/bin/env bash
#
# STILL WAITING — All 8 GPU Test Script
# Validates DDP training across all 8 K80 GPUs on Kepler sm_37
#
# UnobligatedRascal — Making old hardware sing.
#
# Usage:
#   ./scripts/test_all_gpus.sh [model] [steps]
#   Model defaults to Qwen/Qwen2.5-0.5B-Instruct (safe for K80 VRAM)
#   Steps defaults to 10
#

set -euo pipefail

# Configuration
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
STEPS="${2:-50}"
NUM_GPUS=8
LOG_DIR="/home/whistler/still-waiting/python/logs"
CONFIG_DIR="/home/whistler/still-waiting/python/configs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/all_gpus_test_${TIMESTAMP}.log"
CONFIG_FILE="${CONFIG_DIR}/test_all_gpus_${TIMESTAMP}.json"
DATASET="wikitext/wikitext-2-raw-v1"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

cleanup() {
    log "Cleaning up temp config: ${CONFIG_FILE}"
    rm -f "${CONFIG_FILE}" 2>/dev/null || true
}
trap cleanup EXIT

# Verify CUDA is available
log "Checking CUDA availability..."
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found"
    exit 1
fi

GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ "${GPU_COUNT}" -ne "${NUM_GPUS}" ]; then
    warn "Expected ${NUM_GPUS} GPUs, found ${GPU_COUNT}"
fi

# Show GPU info
log "GPU inventory:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || nvidia-smi -L

# Create temp directories
mkdir -p "${LOG_DIR}"
mkdir -p "${CONFIG_DIR}"

# Write test config
cat > "${CONFIG_FILE}" << EOF
{
    "model_ref": "${MODEL}",
    "target_steps": ${STEPS},
    "config": {
        "model_precision": "custom_nf4",
        "dataset_path": "${DATASET}",
        "max_seq_length": 128,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.05,
        "max_steps": ${STEPS}
    }
}
EOF

log "Test config written to: ${CONFIG_FILE}"
log "Model: ${MODEL}"
log "Dataset: ${DATASET}"
log "Steps: ${STEPS}"
log "GPUs: ${NUM_GPUS}"

# Set NCCL environment for Kepler
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=600

# Launch torchrun with all 8 GPUs
log "Launching DDP training on all ${NUM_GPUS} GPUs..."
echo ""
echo "=== Training Output (logging to ${LOG_FILE}) ==="
echo ""

START_TIME=$(date +%s)

torchrun --nproc_per_node=${NUM_GPUS} \
    /home/whistler/still-waiting/python/worker.py \
    "test_all_gpus_${TIMESTAMP}" \
    "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=== Test Complete ==="
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    success "Training completed successfully"
else
    error "Training failed with exit code ${EXIT_CODE}"
fi

log "Elapsed time: ${ELAPSED}s"
log "Log file: ${LOG_FILE}"

# Report VRAM usage
echo ""
log "Post-test VRAM usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || nvidia-smi

# Check for common errors in log
if grep -q "RuntimeError" "${LOG_FILE}"; then
    error "RuntimeError detected in log"
    grep -A 5 "RuntimeError" "${LOG_FILE}" | head -20
fi

if grep -q "OutOfMemory" "${LOG_FILE}"; then
    error "OutOfMemory detected in log"
fi

if grep -q "CUBLAS_STATUS_ARCH_MISMATCH" "${LOG_FILE}"; then
    error "cuBLAS arch mismatch (using Ex API?) detected"
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    success "ALL 8 GPU TEST PASSED"
else
    error "ALL 8 GPU TEST FAILED"
fi

exit ${EXIT_CODE}