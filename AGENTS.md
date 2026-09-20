# still-waiting — AGENTS Notes

> UnobligatedRascal private notes. Technical decisions, lessons learned, next steps.

## Hardware Facts (NOUGHT)

- **CPUs**: 2x Xeon E5-2697v4 (36 cores each, Broadwell), NUMA0 + NUMA1
- **RAM**: 128GB DDR4 ECC (64GB per NUMA domain)
- **GPUs**: 8x Tesla K80 GK210, ~11.5GB each, compute 3.7 (Kepler)
- **Topology**: NUMA0 → GPU 0-3; NUMA1 → GPU 4-7 (PIX-linked pairs)
- **CUDA**: Driver 470.256.02, CUDA 11.8 toolkit
- **Storage**: 256GB OS NVMe + 512GB DATA NVMe (TIGHT for training)

## Key Learnings from llama_wukong

1. **cuBLAS Ex APIs crash on Kepler**: `cublasGemmEx`, `cublasGemmStridedBatchedEx` → `CUBLAS_STATUS_ARCH_MISMATCH`. Must use legacy `cublasSgemm*`.
2. **FA works without tensor cores**: Tile/vector kernels run on sm_37, just slower.
3. **NUMA replication = 2.5x**: Non-negotiable. Pin workers with `numactl --cpunodebind=X --membind=X`.
4. **MMQ works**: Quantized matmul via DP4A is viable on Kepler.
5. **AdamW GPU kernel proven**: `opt-step-adamw.cu` runs on sm_37.

## Training Strategy

- **Baseline**: PyTorch 1.14 + Transformers 4.30 + PEFT LoRA
- **Target models**: 7B-13B LoRA fine-tuning (distributed across 8 GPUs)
- **Compute**: F32 (FP16 is slow without tensor cores)
- **Distributed**: torch.distributed DDP; NUMA0 runs 4 GPUs, NUMA1 runs 4 GPUs
- **Checkpoint every**: 2048 steps (surgical edits require dense checkpoints)

## Architecture Decisions

- **No Docker**: Native install on NOUGHT for direct CUDA access
- **Rust orchestrator**: Stable API surface, job management, conductor hooks
- **Python worker**: PyTorch ecosystem, easy backend swaps
- **GUI (TBD)**: Tauri + SvelteKit; connects over network to NOUGHT's :8000
- **TUI (TBD)**: Ratatui; SSH-accessible control panel

## "In-a-Billion Shots" — Concrete Targets

1. **Patch PyTorch 2.x for sm_37**: Re-add `arch=sm_37` to CUDA builds; route cuBLAS Ex→legacy
2. **Port FA tile kernel**: From llama_wukong's fattn.cu if PyTorch FA rejects Kepler
3. **Custom AdamW CUDA op**: Based on llama_wukong's proven kernel
4. **TurboQuant-style training cache**: Quantize activation caches to save VRAM

## TODO

- [x] Build PyTorch 2.4.0-rc8 with sm_37 support
- [x] Test CUDA compute on Kepler with PyTorch 2.4
- [x] Install transformers/PEFT stack
- [x] Validate LoRA training pipeline
- [x] Deploy python worker code to NOUGHT
- [x] Implement full training loop in transformers_backend.py
- [x] Build browser-based GUI (React+Vite+Tailwind)
- [ ] Test worker ↔ orchestrator IPC (end-to-end training job)
- [ ] Test distributed training across 8 GPUs
- [ ] Build TUI (ratatui) for SSH access
- [ ] Implement GGUF export pipeline
- [ ] Storage management (512GB is tight!)
- [ ] Conductor logic skeleton
- [ ] Fix nought-ssh extension (cached old password, needs restart)
- [ ] Implement Python worker's training loop
- [ ] Build basic TUI (ratatui) for SSH access
- [ ] Design GUI wireframes
- [ ] Test distributed training across 8 GPUs
- [ ] Implement GGUF export pipeline
- [ ] Storage management (512GB is tight!)
- [ ] Conductor logic skeleton

## API Notes

All endpoints under `/v1/` for OpenAI-style compatibility:
- `POST /v1/training/jobs` — create job
- `GET /v1/training/jobs/:id` — status
- `POST /v1/training/jobs/:id/pause|resume|checkpoint|conductor`
- `GET /v1/models` — list trained models

## Build Commands

```bash
# Rust orchestrator
cd orchestrator && cargo build --release

# Python env (NOUGHT)
pip install torch==1.14.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

---
*UnobligatedRascal — Ancient hardware, fresh ambition.*
