#!/usr/bin/env bash
#
# STILL WAITING — All 8 GPU Test Script
# Validates DDP training across all 8 K80 GPUs on Kepler sm_37
# Runs both custom NF4 and FP32 pipelines for comparison
#
# UnobligatedRascal — Making old hardware sing.
#
# Usage:
#   ./scripts/test_all_gpus.sh [model] [steps]
#   Model defaults to Qwen/Qwen2.5-0.5B-Instruct (safe for K80 VRAM)
#   Steps defaults to 50
#

set -euo pipefail

# Configuration
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
STEPS="${2:-50}"
NUM_GPUS=8
LOG_DIR="/home/whistler/still-waiting/python/logs"
CONFIG_DIR="/home/whistler/still-waiting/python/configs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
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

run_test() {
    local precision=$1
    local test_name=$2
    local log_file="${LOG_DIR}/${test_name}_${TIMESTAMP}.log"
    local config_file="${CONFIG_DIR}/${test_name}_${TIMESTAMP}.json"
    local job_id="test_${test_name}_${TIMESTAMP}"

    log "=== ${test_name} TEST ==="
    log "Precision: ${precision}"
    log "Model: ${MODEL}"
    log "Dataset: ${DATASET}"
    log "Steps: ${STEPS}"
    log "GPUs: ${NUM_GPUS}"
    log "Log: ${log_file}"

    # Write test config
    cat > "${config_file}" << EOF
{
    "model_ref": "${MODEL}",
    "target_steps": ${STEPS},
    "config": {
        "model_precision": "${precision}",
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

    # Launch torchrun
    log "Launching DDP training..."
    echo ""

    local start_time=$(date +%s)
    local exit_code=0

    torchrun --nproc_per_node=${NUM_GPUS} \
        /home/whistler/still-waiting/python/worker.py \
        "${job_id}" \
        "${config_file}" 2>&1 | tee "${log_file}"
    exit_code=${PIPESTATUS[0]}

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Extract loss from log
    local final_loss=$(grep -oP 'avg_loss=\K[0-9.]+' "${log_file}" | tail -1)
    if [ -z "${final_loss}" ]; then
        final_loss="N/A"
    fi

    # Extract peak VRAM from log (if available)
    local peak_vram=$(grep -oP 'PEAK VRAM[^:]*: alloc=\K[0-9.]+' "${log_file}" | head -1)
    if [ -z "${peak_vram}" ]; then
        peak_vram="N/A"
    fi

    # Check for errors
    local has_error=0
    if grep -q "RuntimeError" "${log_file}"; then
        has_error=1
        error "RuntimeError detected"
        grep -A 5 "RuntimeError" "${log_file}" | head -20
    fi
    if grep -q "OutOfMemory" "${log_file}"; then
        has_error=1
        error "OutOfMemory detected"
    fi
    if grep -q "CUBLAS_STATUS_ARCH_MISMATCH" "${log_file}"; then
        has_error=1
        error "cuBLAS arch mismatch detected"
    fi

    # Report results
    echo ""
    echo "=== ${test_name} Test Results ==="
    echo ""
    if [ ${has_error} -eq 1 ] || [ ${exit_code} -ne 0 ]; then
        error "${test_name} FAILED (exit ${exit_code})"
    else
        success "${test_name} PASSED"
    fi
    echo "Elapsed: ${elapsed}s"
    echo "Final loss: ${final_loss}"
    echo "Peak VRAM: ${peak_vram}GB"
    echo "Log: ${log_file}"
    echo ""

    # Cleanup config
    rm -f "${config_file}" 2>/dev/null || true

    # Store results for comparison
    eval "${test_name}_elapsed=${elapsed}"
    eval "${test_name}_loss=${final_loss}"
    eval "${test_name}_peak_vram=${peak_vram}"
    eval "${test_name}_exit=${exit_code}"

    return ${exit_code}
}

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

# Set NCCL environment for Kepler
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=600

# Run NF4 test
run_test "custom_nf4" "nf4"
NF4_EXIT=$?

# Clear cache between tests
log "Clearing CUDA cache between tests..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 2

# Run FP32 test
run_test "f32" "f32"
F32_EXIT=$?

# Report VRAM usage after all tests
echo ""
log "Post-test VRAM usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || nvidia-smi

# Comparison summary
echo ""
echo "=== COMPARISON SUMMARY ==="
echo ""
echo "Test        | Elapsed | Final Loss | Peak VRAM"
echo "------------|---------|------------|----------"
echo "NF4         | ${nf4_elapsed:-N/A}s      | ${nf4_loss:-N/A}       | ${nf4_peak_vram:-N/A}"
echo "FP32        | ${f32_elapsed:-N/A}s      | ${f32_loss:-N/A}       | ${f32_peak_vram:-N/A}"
echo ""

# Calculate speedup if both succeeded
if [ "${nf4_elapsed}" != "N/A" ] && [ "${f32_elapsed}" != "N/A" ] && [ ${f32_elapsed} -gt 0 ]; then
    speedup=$(python3 -c "print(f'{${f32_elapsed}/${nf4_elapsed}:.2f}')" 2>/dev/null || echo "N/A")
    echo "NF4 vs FP32 speedup: ${speedup}x"
fi

# Overall result
if [ ${NF4_EXIT} -eq 0 ] && [ ${F32_EXIT} -eq 0 ]; then
    echo ""
    success "ALL TESTS PASSED (NF4 + FP32)"
    exit 0
else
    echo ""
    error "SOME TESTS FAILED"
    exit 1
fi