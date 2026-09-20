# still-waiting — TODO & Status

> UnobligatedRascal — Ancient hardware, fresh ambition.

---

## Status: 2026-09-19

### Completed
- [x] PyTorch 2.4.0a0+sm_37 built from source on NOUGHT
- [x] CUDA compute validated on Kepler (matmul, attention, backprop)
- [x] transformers 4.40.0 + peft 0.7.0 installed
- [x] LoRA training pipeline validated on Qwen2.5-0.5B-Instruct
- [x] Orchestrator running on port 9999 (all endpoints working)
- [x] Project repo initialized and committed

### Current Setup
- PyTorch: editable install from `/home/whistler/pytorch-kepler`
- Python venv: `/home/whistler/venv311`
- cuDNN: **NOT available** (version=None) — cuBLAS legacy APIs only
- GPUs: 8x Tesla K80, 11.2GB each, sm_37

---

## GPU Usage Rules (CRITICAL)

**Production llama_wukong is running** — most GPUs are occupied.

| GPUs | Status | Notes |
|------|--------|-------|
| 0,3,4,7 | **BUSY** | Primary production GPUs |
| 1,2,5,6 | **PARTIAL** | ~1.5-2GB free typically — use for simple tests only |
| ALL 8 | **REQUIRES USER** | Full testing across all GPUs must be done manually by user when production allows |

### VRAM Discipline (ALWAYS)
1. **Check available VRAM before every test:**
   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv
   ```
2. **Calculate realistic load:**
   - Base model size in VRAM: Qwen2.5-0.5B ≈ 1GB (F32), Llama-3-8B ≈ 16GB (F32), 7B ≈ 14GB
   - LoRA adds minimal overhead (~10-50MB depending on rank)
   - Optimizer states: 2x model size in F32 (Adam)
   - Batch processing: seq_length × batch_size × hidden_dim × 4 bytes
   - **Rule: never exceed 80% of available VRAM**

3. **Safe test models for partial GPUs (1,2,5,6):**
   - Qwen2.5-0.5B-Instruct (~1GB F32): SAFE — use for pipeline validation
   - TinyLlama-1.1B (~2.2GB F32): MAYBE — only with batch_size=1, no optimizer
   - Anything >3B: NOT safe on partial GPUs

4. **Full 8-GPU distributed tests:**
   - Only run when user confirms production load is reduced
   - Pin by NUMA: `numactl --cpunodebind=0 --membind=0` (GPU 0-3), `numactl --cpunodebind=1 --membind=1` (GPU 4-7)

---

## Next Steps

### High Priority
- [ ] Implement full training loop in `transformers_backend.py`
  - Replace dummy `train_step()` with actual dataset loading + optimizer step
  - Add gradient accumulation (Kepler needs this for effective batch size)
  - Manual gradient checkpointing (transformers' built-in may use unsupported ops)
- [ ] Deploy python worker code to NOUGHT (`/home/whistler/still-waiting/python/`)
- [ ] Test worker ↔ orchestrator IPC

### Medium Priority
- [ ] Test distributed training (DDP) across GPUs 1,2,5,6 (user-run)
- [ ] Build TUI (ratatui) for SSH monitoring
- [ ] Implement conductor logic skeleton
- [ ] Storage layout for checkpoints (`/data/checkpoints/` — verify space)

### Future
- [ ] GGUF export pipeline (for trained adapters → llama.cpp)
- [ ] Custom AdamW CUDA op (port from llama_wukong)
- [ ] FA tile kernel port if PyTorch FA rejects Kepler
- [ ] GUI wireframes (Tauri + SvelteKit)

---

## Quick Reference

### Activate environment on NOUGHT:
```bash
source /home/whistler/venv311/bin/activate
```

### Start orchestrator:
```bash
cd /home/whistler/still-waiting/orchestrator && cargo run --release
# OR via service: sudo systemctl start orchestrator
```

### Run worker (single GPU, NUMA0):
```bash
numactl --cpunodebind=0 --membind=0 \
  CUDA_VISIBLE_DEVICES=1 \
  python3 /home/whistler/still-waiting/python/worker.py <job_id> <config.json>
```

### Run worker (distributed, NUMA0):
```bash
numactl --cpunodebind=0 --membind=0 \
  torchrun --nproc_per_node=2 \
  python3 /home/whistler/still-waiting/python/worker.py <job_id> <config.json>
```

---

*Built by UnobligatedRascal.*
