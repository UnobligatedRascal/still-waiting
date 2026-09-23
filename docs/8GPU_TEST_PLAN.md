# 8-GPU Full Training Test Plan

> UnobligatedRascal — Ancient hardware, fresh ambition.

**Date**: 2026-09-25  
**Goal**: Validate full 8-GPU DDP training on NOUGHT with NF4 quantization.

---

## Prerequisites Checklist

### Hardware
- [x] All 8x Tesla K80 GPUs available (production llama_wukong paused)
- [x] NUMA topology configured (kernel-level: NUMA0→GPU0-3, NUMA1→GPU4-7)
- [ ] Verify VRAM: `nvidia-smi` — expect ~11.4GB free per GPU

### Software
- [x] PyTorch 2.4.0a0+sm_37 installed at `/home/whistler/pytorch-kepler`
- [x] venv311 active: `/home/whistler/venv311/bin/activate`
- [x] NF4 CUDA kernels deployed: `python/kernels/nf4_dequant.cu`, `nf4_cuda.py`, `nf4_kepler.py`
- [x] Worker.py fixed for DDP (rank-gated checkpoints/logs)
- [x] WHISTLER ↔ NOUGHT in sync (commit ca4932c)

### Storage
- [ ] Verify disk space: `df -h` — need ~50GB+ for checkpoints
- [ ] Checkpoint dir: `/home/whistler/still-waiting/python/checkpoints/`

### Orchestrator
- [ ] Orchestrator running: `pgrep -f agent-orchestrator`
- [ ] GUI accessible: `http://<NOUGHT_IP>:9999/`
- [ ] API accessible: `curl http://<NOUGHT_IP>:9999/v1/system/status`

---

## Test Scenarios (ordered by risk)

### Test 1: NF4 Inference — Single GPU Smoke Test

**Purpose**: Verify NF4 kernel loads and runs on Kepler.

```bash
cd /home/whistler/still-waiting/python
source /home/whistler/venv311/bin/activate
CUDA_VISIBLE_DEVICES=0 python test_nf4.py --test model --model Qwen/Qwen2.5-0.5B-Instruct
```

**Expected**: ~1GB VRAM, 7.1x compression, inference outputs text.

### Test 2: NF4 Training — Single GPU

**Purpose**: Verify NF4 forward + backward + optimizer works end-to-end.

```bash
cd /home/whistler/still-waiting/python
source /home/whistler/venv311/bin/activate
CUDA_VISIBLE_DEVICES=0 python test_nf4.py --test train --model Qwen/Qwen2.5-0.5B-Instruct --steps 20
```

**Expected**: Loss decreases, no OOM, no CUDA errors.

### Test 3: DDP F32 — 8 GPUs (no NF4)

**Purpose**: Verify DDP infrastructure works across all 8 GPUs before adding NF4 complexity.

```bash
cd /home/whistler/still-waiting/python
source /home/whistler/venv311/bin/activate
cat > /tmp/ddp_test_config.json << 'EOF'
{
  "model_ref": "Qwen/Qwen2.5-0.5B-Instruct",
  "config": {
    "model_precision": "f32",
    "dataset_path": null,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 512,
    "learning_rate": 2e-4,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "num_train_epochs": 1
  },
  "target_steps": 100
}
EOF

torchrun --nproc_per_node=8 --nnodes=1 python worker.py ddp_test_001 /tmp/ddp_test_config.json
```

**Expected**: 
- All 8 ranks initialize
- Loss decreases
- Checkpoints saved at step 2048 (or at completion if target_steps < 2048)
- No NCCL errors

### Test 4: NF4 + DDP — 8 GPUs (PRIMARY TEST)

**Purpose**: Validate NF4 quantized training across full 8-GPU cluster.

**Model options**:
- Safe: `Qwen/Qwen2.5-0.5B-Instruct` (~1.5GB NF4 per rank, trivial)
- Target: `Qwen/Qwen2.5-7B-Instruct` (~5-7GB NF4+LoRA+optimizer per rank)

```bash
cd /home/whistler/still-waiting/python
source /home/whistler/venv311/bin/activate

# For 7B test (requires clean GPUs):
cat > /tmp/nf4_8gpu_config.json << 'EOF'
{
  "model_ref": "Qwen/Qwen2.5-7B-Instruct",
  "config": {
    "model_precision": "custom_nf4",
    "dataset_path": null,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 512,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "num_train_epochs": 1
  },
  "target_steps": 500
}
EOF

torchrun --nproc_per_node=8 --nnodes=1 python worker.py nf4_8gpu_test_001 /tmp/nf4_8gpu_config.json
```

**Expected**:
- CUDA extension compiles on each rank (~5-10s first time, cached after)
- NF4 weights quantized: ~3.5GB → 7.1x compression per rank
- DDP broadcasts NF4 buffers from rank 0 to all ranks
- Training proceeds with loss decreasing
- Effective batch size: 1 × 8 (accum) × 8 (GPUs) = 64

### Test 5: NF4 + DDP — Real Dataset

**Purpose**: Validate with actual training data.

```bash
cat > /tmp/nf4_real_dataset_config.json << 'EOF'
{
  "model_ref": "Qwen/Qwen2.5-7B-Instruct",
  "config": {
    "model_precision": "custom_nf4",
    "dataset_path": "/path/to/training/data.jsonl",
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "num_train_epochs": 3
  },
  "target_steps": 2048
}
EOF

torchrun --nproc_per_node=8 --nnodes=1 python worker.py nf4_real_test_001 /tmp/nf4_real_dataset_config.json
```

---

## Monitoring Commands

```bash
# Watch GPU utilization across all 8 GPUs
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv'

# Watch training logs from orchestrator
tail -f /home/whistler/still-waiting/logs/orchestrator.log

# Watch worker output directly
tail -f /tmp/*.log  # or monitor stdout

# Check orchestrator API for job status
curl -s http://localhost:9999/v1/training/jobs | python3 -m json.tool

# Monitor checkpoint saves
watch -n 5 'ls -lh /home/whistler/still-waiting/python/checkpoints/'
```

---

## Expected VRAM Usage (per GPU)

| Component | 7B NF4 Estimate |
|-----------|-----------------|
| NF4 weights | ~3.5GB |
| LoRA adapters (r=16) | ~0.1GB |
| Optimizer states (AdamW, LoRA only) | ~0.4GB |
| Activations (batch=1, seq=512) | ~0.5-1.0GB |
| CUDA kernel overhead | ~0.5GB |
| **Total per GPU** | **~5-6GB** |
| **K80 available** | **~11.4GB** |
| **Headroom** | **~50%** |

**Note**: DDP replicates full model on each GPU. VRAM is per-rank.

---

## Failure Modes & Troubleshooting

### OOM (Out of Memory)
- Reduce batch_size to 1
- Reduce max_seq_length to 256
- Reduce gradient_accumulation_steps

### NCCL Error
- Check `NCCL_DEBUG=INFO` output
- Verify all 8 GPUs are accessible: `nvidia-smi`
- Check for PCIe/NVLink issues: `nvidia-smi topo -m`

### CUDA Kernel Failure
- Verify GCC 11: `gcc --version`
- Clear CUDA cache: `rm -rf ~/.cache/torch_extensions`
- Check kernel output: `TORCH_EXTENSIONS_DIR=/tmp/torch_ext`

### Checkpoint Corruption
- Verify only rank 0 saves: check worker.py rank gates
- Use `strace -p <pid>` to see file operations

### DDP Hang (all-reduce timeout)
- One GPU stuck → check `nvidia-smi` for asymmetric utilization
- Set longer timeout: `NCCL_TIMEOUT=900`
- Check for straggler processes

---

## Files Changed for This Test

- `python/worker.py` — Added:
  - NCCL environment variables for Kepler robustness
  - `is_main_rank()` helper
  - CUDA extension pre-warming before DDP init
  - Rank-gated checkpoint saves (avoid file corruption)
  - Rank-gated orchestrator notifications
  - Rank-gated log sends (avoid 8x spam)

---

*UnobligatedRascal — Ancient hardware, fresh ambition.*
