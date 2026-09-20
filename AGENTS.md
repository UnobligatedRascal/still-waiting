# still-waiting — AGENTS Notes

> UnobligatedRascal private notes. Technical decisions, lessons learned, next steps.

## Hardware Facts (NOUGHT)

- **CPUs**: 2x Xeon E5-2697v4 (36 cores each, Broadwell), NUMA0 + NUMA1
- **RAM**: 128GB DDR4 ECC (64GB per NUMA domain)
- **GPUs**: 8x Tesla K80 GK210, ~11.5GB each, compute 3.7 (Kepler)
- **Topology**: NUMA0 → GPU 0-3; NUMA1 → GPU 4-7 (PIX-linked pairs)
- **CUDA**: Driver 470.256.02, CUDA 11.8 toolkit
- **Storage**: 256GB OS NVMe + 512GB DATA NVMe (TIGHT for training)
- **Network**: 192.168.137.29, accessible from Windows workstation
- **User**: whistler, PW updated 2026-09-20: `T0I$YQ161913`
- **SSH**: Password auth works; nought-ssh extension cached old credentials (needs restart)

## Key Learnings from llama_wukong

1. **cuBLAS Ex APIs crash on Kepler**: `cublasGemmEx`, `cublasGemmStridedBatchedEx` → `CUBLAS_STATUS_ARCH_MISMATCH`. Must use legacy `cublasSgemm*`.
2. **FA works without tensor cores**: Tile/vector kernels run on sm_37, just slower.
3. **NUMA replication = 2.5x**: Non-negotiable. Pin workers with `numactl --cpunodebind=X --membind=X`.
4. **MMQ works**: Quantized matmul via DP4A is viable on Kepler.
5. **AdamW GPU kernel proven**: `opt-step-adamw.cu` runs on sm_37.

## Training Strategy

- **Baseline**: PyTorch 2.4.0a0+sm_37 + Transformers 4.40 + PEFT 0.7.0
- **Target models**: 7B-13B LoRA fine-tuning (distributed across 8 GPUs)
- **Compute**: F32 (FP16 is slow without tensor cores)
- **Distributed**: torch.distributed DDP; NUMA0 runs 4 GPUs, NUMA1 runs 4 GPUs
- **Checkpoint every**: 2048 steps (surgical edits require dense checkpoints)
- **cuDNN**: NOT available (version=None); cuBLAS legacy APIs only

## Architecture

```
Browser (:9999/) → orchestrator (Rust, :9999) → Python workers → Kepler GPUs
                    ├─ /v1/ API
                    ├─ Static GUI (SPA)
                    └─ Conductor hooks
```

### Decisions Made

- **No Docker**: Native install on NOUGHT for direct CUDA access
- **Browser GUI** (not Tauri): NOUGHT is headless; browser = zero install, any device, no CORS
- **Orchestrator serves GUI**: Single deployment artifact, `/` for SPA, `/v1/` for API
- **Workers spawned manually**: No auto-spawning yet; user launches via numactl + torchrun
- **Free-form job config**: Pass any hyperparams without changing Rust core
- **Checkpoint-first**: 2048-step cadence; surgical edits require dense history

## Session Summary — 2026-09-20

### Completed

- ✅ **Training loop** (`python/backends/transformers_backend.py`):
  - Full training step with gradient accumulation (default 8x)
  - AdamW optimizer + linear warmup scheduler
  - Dataset loading: JSONL files, HF datasets, dummy for validation
  - DDP support; periodic VRAM cache clearing every 64 steps
  - Configurable: batch_size, max_seq_length, LoRA params, LR

- ✅ **Browser GUI** (`gui/`):
  - React + Vite + Tailwind CSS SPA, ~157KB gzipped
  - Job dashboard (auto-refresh 5s), creation form, detail modal
  - SystemStatus panel: orchestrator uptime, job counts, live GPU bars
  - Served from orchestrator static dir

- ✅ **Orchestrator** updates:
  - Static file serving with SPA fallback (`tower-http::ServeDir`)
  - `/v1/system/status` endpoint (orchestrator info, GPU stats via nvidia-smi)
  - Fixed router bug (merge vs nest for API routes)

- ✅ **Startup infrastructure**:
  - `deploy/start_still_waiting.sh`: start/stop/restart/status/install-service/user commands
  - `deploy/orchestrator.service`: systemd unit file
  - Rust added to PATH via `~/.cargo/env` sourcing in .bashrc

- ✅ **Full deployment to NOUGHT**:
  - All files at `/home/whistler/still-waiting/`
  - Orchestrator built: `orchestrator/target/release/agent-orchestrator`
  - Running on port 9999; API and GUI verified working

### Current State

- Orchestrator: RUNNING (PID via `pgrep -f agent-orchestrator`)
- GUI: http://192.168.137.29:9999/ (accessible from LAN)
- API: http://192.168.137.29:9999/v1/ (all endpoints functional)
- Python backend: deployed but NOT auto-integrated with orchestrator yet
- Workers: must be started manually per job

### Critical Knowledge for Continuation

**Paths on NOUGHT**:
- Project: `/home/whistler/still-waiting/`
- Orchestrator binary: `/home/whistler/still-waiting/orchestrator/target/release/agent-orchestrator`
- Python env: `/home/whistler/venv311/bin/activate`
- PyTorch (editable): `/home/whistler/pytorch-kepler`
- GUI dist: `/home/whistler/still-waiting/gui/dist/`
- Rust: `. ~/.cargo/env` before using cargo/rustc

**GPU constraints**:
- GPUs 0,3,4,7 = production llama_wukong (BUSY)
- GPUs 1,2,5,6 = partial (~1.5-2GB free, sub-billion models only)
- Safe test: Qwen2.5-0.5B-Instruct (~1GB F32)

**Worker launch pattern**:
```bash
source /home/whistler/venv311/bin/activate
numactl --cpunodebind=0 --membind=0 \
  CUDA_VISIBLE_DEVICES=1 \
  python3 /home/whistler/still-waiting/python/worker.py <job_id> <config.json>
```

**Deployment note**: base64 pipe upload fails silently for files >50KB; use SFTP `fastPut` instead.

### Next Session Priorities

1. **End-to-end test**: Create job via API/GUI, manually launch worker, verify training loop
2. **Worker auto-spawn**: Orchestrator should launch workers as subprocesses (currently manual)
3. **Worker IPC**: Implement proper heartbeats and job state sync
4. **Distributed training test**: torchrun across GPUs 1,2,5,6 (user-run)
5. **TUI**: Ratatui for SSH monitoring
6. **Conductor**: Implement median-arc evaluation logic

## "In-a-Billion Shots" — Concrete Targets

1. **Patch PyTorch 2.x for sm_37**: ✅ Done — PyTorch 2.4.0a0 built with sm_37
2. **Port FA tile kernel**: If PyTorch FA rejects Kepler, use llama_wukong's fattn.cu
3. **Custom AdamW CUDA op**: Port from llama_wukong's opt-step-adamw.cu
4. **TurboQuant-style training cache**: Quantize activation caches to save VRAM

## API Reference

All endpoints on `http://NOUGHT:9999/v1/`:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/training/jobs` | Create training job |
| GET | `/training/jobs` | List all jobs |
| GET | `/training/jobs/:id` | Job status + checkpoints |
| POST | `/training/jobs/:id/pause` | Pause job |
| POST | `/training/jobs/:id/resume` | Resume job |
| POST | `/training/jobs/:id/checkpoint` | Worker reports checkpoint |
| POST | `/training/jobs/:id/conductor` | Apply surgical edit |
| POST | `/training/jobs/:id/complete` | Mark complete |
| POST | `/training/jobs/:id/fail` | Report failure |
| GET | `/models` | List trained models |
| GET | `/system/status` | System info, GPU stats, restart instructions |

## Build Commands

```bash
# On NOUGHT:
source ~/.cargo/env

# Build orchestrator
cd /home/whistler/still-waiting/orchestrator && cargo build --release

# Build GUI (on Windows, deploy dist/ to NOUGHT)
cd gui && npm run build

# Start orchestrator
./deploy/start_still_waiting.sh start

# Start systemd service (if installed)
sudo systemctl start still-waiting-orchestrator
```

---
*UnobligatedRascal — Ancient hardware, fresh ambition.*
