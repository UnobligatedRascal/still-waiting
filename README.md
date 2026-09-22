# STILL WAITING — Private Model Training on Ancient Hardware

> "We don't settle for right. We make people forget what boxes even were."
> — UnobligatedRascal

Turn your Kepler-era rig into a lean, private LLM training station — despite 13-year-old hardware and total ecosystem abandonment.

**Access**: `http://<YOUR_TRAINING_NODE_IP>:9999/` (when orchestrator is running)

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

Point to: **`http://<YOUR_TRAINING_NODE_IP>:9999/`**

Create a training job with:
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (safe for partial GPUs)
- Target steps: `1000`
- Dataset path: optional JSONL file path

### Run a training worker (manually)

```bash
# Activate Python environment
source <YOUR_VENV_PATH>/bin/activate

# Single GPU
numactl --cpunodebind=0 --membind=0 \
  CUDA_VISIBLE_DEVICES=0 \
  python3 <PROJECT_ROOT>/python/worker.py <job_id> <config.json>

# Multiple GPUs (DDP, same NUMA node)
numactl --cpunodebind=0 --membind=0 \
  torchrun --nproc_per_node=2 \
  python3 <PROJECT_ROOT>/python/worker.py <job_id> <config.json>
```

---

## Architecture

```
┌─────────────────────┐     ┌─────────────────────────────────────────────┐
│   Your Browser      │     │              Training Node                    │
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

## Hardware Reality

Built for and tested on dual-socket Xeon + 8x Tesla K80 (sm_37) — but the code works on any CUDA-capable rig.

| Component | Test Spec | Impact |
|-----------|-----------|--------|
| CPU | 2x Xeon E5-2697v4 (72 cores total) | Dual NUMA domains; pin workers to NUMA nodes |
| RAM | 128GB DDR4 ECC | Plenty for LoRA up to 13B params |
| GPU | 8x Tesla K80 GK210 Kepler (~11.5GB each, sm_3.7) | ANCIENT. No tensor cores. CUDA 11.8 only. |
| Storage | ~200GB NVMe | Manage checkpoints carefully |

### NUMA Awareness

On dual-socket systems, pin workers to NUMA nodes for ~2.5x speedup:

```bash
# NUMA node 0 (e.g., GPUs 0-3)
numactl --cpunodebind=0 --membind=0 CUDA_VISIBLE_DEVICES=0,1 python3 worker.py ...

# NUMA node 1 (e.g., GPUs 4-7)
numactl --cpunodebind=1 --membind=1 CUDA_VISIBLE_DEVICES=4,5 python3 worker.py ...
```

---

## Project Structure

```
still-waiting/
├── README.md                      # this file
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
│   └── orchestrator.service       # systemd unit (example)
└── tui/                           # Ratatui SSH client (planned)
```

---

## API Reference

All endpoints: `http://<YOUR_IP>:9999/v1/`

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
| `model_precision` | "f32" | "f32", "f16_storage", "bnb_nf4", or "custom_nf4" |
| `quant` | null | Alternative: "nf4" for custom NF4 quantization (same as model_precision="custom_nf4") |
| `max_seq_length` | 512 | Reduce for VRAM-constrained GPUs |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling |
| `lora_dropout` | 0.05 | Dropout rate |
| `target_modules` | auto-detected | LoRA target layers (per model architecture) |
| `learning_rate` | 2e-4 | AdamW LR |
| `batch_size` | 2 | Per-GPU batch |
| `gradient_accumulation_steps` | 8 | Effective batch = batch_size × accum × GPUs |
| `warmup_ratio` | 0.05 | Linear warmup |
| `dataset_path` | null | JSONL file, directory, or HF dataset name |

**model_precision modes**:
- `f32`: Full precision. Safe, slowest, most VRAM. Default.
- `f16_storage`: Load weights in FP16 (half VRAM during load), compute in FP32. Recommended for 3B+ models.
- `bnb_nf4`: bitsandbytes QLoRA. ⚠️ Requires bitsandbytes install; currently BROKEN on Kepler sm_37.
- `custom_nf4`: Custom NF4 4-bit quantization for Kepler (⏳ in development). ~7.8x compression vs FP32.

**target_modules**: Auto-detected based on model name. Qwen2.5 → all attention layers. Llama-3 → all attention layers. Override manually if needed.

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
npm run dev   # proxies /v1 to your training node
npm run build # outputs to dist/
```

---

## Orchestration

### Start/Stop/Restart

Via startup script (recommended):

```bash
# Start (requires root)
sudo <PROJECT_ROOT>/deploy/start_still_waiting.sh start

# Stop
sudo <PROJECT_ROOT>/deploy/start_still_waiting.sh stop

# Restart
sudo <PROJECT_ROOT>/deploy/start_still_waiting.sh restart

# Status
<Project_ROOT>/deploy/start_still_waiting.sh status

# Run in foreground (no root)
<Project_ROOT>/deploy/start_still_waiting.sh user
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

- **cuDNN may be unavailable** on very old drivers — cuBLAS legacy APIs work fine
- **No tensor cores** on Kepler — FP16 is slow; we use F32
- **VRAM limited** on K80 — sub-billion models recommended for single GPU
- **Worker auto-spawning not implemented** — workers launched manually for now

---

## What Works on Kepler sm_37

Proven and tested:

- ✅ PyTorch 2.4.0-rc8 built from source with sm_37
- ✅ Legacy cuBLAS (Sgemm, not SgemmEx)
- ✅ Flash Attention tile kernels (no tensor cores needed)
- ✅ AdamW GPU optimizer
- ✅ NCCL distributed training across 8 GPUs
- ✅ NUMA replication = 2.5x speedup
- ✅ MMQ quantized matmul via DP4A
- ✅ transformers + PEFT LoRA pipeline

---

**Built by UnobligatedRascal. Ancient hardware, fresh ambition. We rebuild what the world abandoned.**
