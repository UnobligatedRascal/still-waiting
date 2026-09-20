# STILL WAITING — Private Model Training on Ancient Hardware

> "We don't settle for right. We make people forget what boxes even were."
> — UnobligatedRascal

Turn NOUGHT (Xeon E5-2697v4 + 128GB ECC + 8x Tesla K80 Kepler) into a lean, private LLM training rig — despite 13-year-old hardware and total ecosystem abandonment.

**Access**: `http://<YOUR_NOUGHT_IP>:9999/` (when orchestrator is running)

---

## Quick Start

### Automated (recommended)

```bash
# Check your hardware compatibility
./scripts/check_hardware.sh

# Full automated setup (installs deps, builds everything)
sudo ./scripts/setup.sh

# Start and open browser
./deploy/start_still_waiting.sh user
```

Access at: `http://<YOUR_IP>:9999/`

### Manual (for more control)

See sections below for step-by-step instructions.

### Options

```bash
# Skip PyTorch build (use existing install or pre-built wheel)
sudo ./scripts/setup.sh --skip-pytorch

# Custom PyTorch version
sudo ./scripts/setup.sh --pytorch-version v2.4.0-rc8

# Setup + auto-start
sudo ./scripts/setup.sh --start
```

### From your browser

Point to: **`http://<YOUR_NOUGHT_IP>:9999/`**

Create a training job with:
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (safe for partial GPUs)
- Target steps: `1000`
- Dataset path: optional JSONL file path

### Run a training worker (manually)

```bash
# Activate Python environment
source /home/whistler/venv311/bin/activate

# Single GPU (e.g., GPU 1)
numactl --cpunodebind=0 --membind=0 \
  CUDA_VISIBLE_DEVICES=1 \
  python3 /home/whistler/still-waiting/python/worker.py <job_id> <config.json>

# Multiple GPUs (DDP, NUMA0)
numactl --cpunodebind=0 --membind=0 \
  torchrun --nproc_per_node=2 \
  python3 /home/whistler/still-waiting/python/worker.py <job_id> <config.json>
```

---

## Architecture

```
┌─────────────────────┐     ┌─────────────────────────────────────────────┐
│   Your Browser      │     │              NOUGHT (<YOUR_IP>)         │
│                     │     │                                             │
│  GUI (SPA at /)     │◄────│── agent-orchestrator (:9999)               │
│  - Job dashboard    │ HTTP│  - API at /v1/                             │
│  - Job creation     │ WS  │  - Static file serving                     │
│  - System status    │     │  - Conductor hooks                         │
│                     │     │  └─────────────────────────────────────────┘
│                     │     │              Python Workers                │
│                     │─────│── transformers_backend.py                  │
│                     │     │  - LoRA fine-tuning                        │
│                     │     │  - Kepler-optimized (F32, gradient accum)  │
└─────────────────────┘     └─────────────────────────────────────────────┘
```

---

## Hardware Reality — NOUGHT

| Component | Spec | Impact |
|-----------|------|--------|
| CPU | 2x Xeon E5-2697v4 (72 cores total) | Dual NUMA: NUMA0→GPU0-3, NUMA1→GPU4-7 |
| RAM | 128GB DDR4 ECC (64GB per NUMA) | Plenty for LoRA up to 13B params |
| GPU | 8x Tesla K80 GK210 Kepler (~11.5GB each, sm_3.7) | ANCIENT. No tensor cores. CUDA 11.8 only. |
| Storage | 225GB root NVMe (~84GB free) | TIGHT. Compress checkpoints. |

### GPU Availability

Production llama_wukong is running — most GPUs are occupied:

| GPUs | Status | Notes |
|------|--------|-------|
| 0,3,4,7 | **BUSY** | Primary production |
| 1,2,5,6 | **PARTIAL** | ~1.5-2GB free — tiny models only |

**Safe test models**: Qwen2.5-0.5B-Instruct (~1GB F32). Anything >3B requires full GPU allocation.

---

## Project Structure

```
still-waiting/
├── README.md                      # this file
├── TODO.md                        # status and next steps
├── AGENTS.md                      # internal notes
├── orchestrator/                  # Rust API server + static file server
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                # entry point, serves GUI + API
│       ├── api.rs                 # /v1/ endpoints
│       ├── state.rs               # jobs, checkpoints, models
│       ├── conductor.rs           # surgical edit hooks
│       └── backend.rs             # worker IPC traits
├── gui/                           # Browser control panel (React+Vite+Tailwind)
│   ├── src/                       # React components
│   ├── dist/                      # built production bundle
│   └── package.json
├── python/                        # Training worker
│   ├── worker.py                  # job executor, NUMA-aware
│   ├── backends/
│   │   └── transformers_backend.py  # Kepler LoRA training loop
│   └── requirements.txt
├── deploy/
│   ├── start_still_waiting.sh     # startup/control script
│   └── orchestrator.service       # systemd unit
└── tui/                           # Ratatui SSH client (planned)
```

---

## API Reference

All endpoints: `http://NOUGHT:9999/v1/`

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/training/jobs` | Create training job |
| GET | `/training/jobs` | List all jobs |
| GET | `/training/jobs/:id` | Job status + checkpoints |
| POST | `/training/jobs/:id/pause` | Pause job |
| POST | `/training/jobs/:id/resume` | Resume job |
| POST | `/training/jobs/:id/checkpoint` | Worker reports checkpoint |
| POST | `/training/jobs/:id/conductor` | Apply surgical edit |
| POST | `/training/jobs/:id/complete` | Mark job complete |
| POST | `/training/jobs/:id/fail` | Report job failure |
| GET | `/models` | List trained models |
| GET | `/system/status` | System info, GPU status, uptime |

### Create Job Request

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "target_steps": 1000,
  "config": {
    "dataset_path": "/data/training.jsonl",
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_length": 512,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "gradient_accumulation_steps": 8
  }
}
```

---

## Training Backend

### Configuration Options

Via job `config` object:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_seq_length` | 512 | Reduced for Kepler VRAM |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling |
| `lora_dropout` | 0.05 | Dropout rate |
| `target_modules` | ["q_proj", "v_proj"] | LoRA target layers |
| `learning_rate` | 2e-4 | AdamW LR |
| `batch_size` | 2 | Per-GPU batch |
| `gradient_accumulation_steps` | 8 | Effective batch = batch_size × accum × GPUs |
| `warmup_ratio` | 0.05 | Linear warmup |
| `dataset_path` | null | JSONL file, directory, or HF dataset name |

### Dataset Format (JSONL)

Each line is a JSON object:

```json
{"text": "Full training text here..."}
```

Or chat-style:

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

### Kepler Optimizations Applied

1. **F32 compute** — FP16 is slow without tensor cores
2. **Legacy cuBLAS only** — cuBLAS Ex APIs crash on Kepler
3. **Gradient accumulation** — Kepler needs larger effective batch sizes
4. **NUMA pinning** — 2.5x speedup (handled by worker.py via numactl)
5. **Periodic cache clearing** — every 64 steps to manage limited VRAM
6. **Gradient clipping** — max_norm=1.0 for training stability

---

## GUI

Browser-based SPA served from orchestrator. No separate install needed.

### Features

- **Dashboard**: Live job status, auto-refresh every 5s
- **Job creation**: Model ref, target steps, dataset path
- **Job detail**: Checkpoint history, loss metrics, pause/resume controls
- **System status**: Orchestrator uptime, job counts, live GPU usage bars

### Development

```bash
cd gui
npm install
npm run dev   # proxies /v1 to NOUGHT:9999
npm run build # outputs to dist/
```

---

## Orchestration

### Start/Stop/Restart

Via startup script (recommended):

```bash
# Start (requires root)
sudo /home/whistler/still-waiting/deploy/start_still_waiting.sh start

# Stop
sudo /home/whistler/still-waiting/deploy/start_still_waiting.sh stop

# Restart
sudo /home/whistler/still-waiting/deploy/start_still_waiting.sh restart

# Status
/home/whistler/still-waiting/deploy/start_still_waiting.sh status

# Run in foreground (no root)
/home/whistler/still-waiting/deploy/start_still_waiting.sh user
```

Via systemd (if installed):

```bash
sudo systemctl start still-waiting-orchestrator
sudo systemctl stop still-waiting-orchestrator
sudo systemctl restart still-waiting-orchestrator
sudo systemctl status still-waiting-orchestrator
sudo systemctl enable still-waiting-orchestrator  # auto-start on boot
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORCH_BIND_ADDR` | 0.0.0.0 | Bind address |
| `ORCH_BIND_PORT` | 9999 | Bind port |
| `ORCH_STATIC_DIR` | gui/dist | Path to GUI dist files |
| `CHECKPOINT_EVERY` | 2048 | Checkpoint interval (worker) |
| `NUMA_NODE` | 0 | Worker NUMA domain |

---

## Known Limitations

- **cuDNN unavailable** — using cuBLAS legacy APIs only
- **No tensor cores** — FP16 is slow; we use F32
- **VRAM limited** — only sub-billion models on partial GPUs
- **Storage tight** — checkpoint compression needed for long runs
- **Worker auto-spawning not implemented** — workers launched manually for now
- **nought-ssh extension** — may need manual credential update (password changed)

---

## What Works on Kepler sm_37

Proven from llama_wukong:

- ✅ PyTorch 2.4.0-rc8 built from source with sm_37
- ✅ Legacy cuBLAS (Sgemm, not SgemmEx)
- ✅ Flash Attention tile kernels (no tensor cores needed)
- ✅ AdamW GPU optimizer
- ✅ NCCL distributed training across 8 GPUs
- ✅ NUMA replication = 2.5x speedup
- ✅ MMQ quantized matmul via DP4A
- ✅ transformers + PEFT LoRA pipeline

---

## Next Steps

See `TODO.md` for current priorities.

---

**Built by UnobligatedRascal. Ancient hardware, fresh ambition. We rebuild what the world abandoned.**
