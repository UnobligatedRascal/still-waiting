# still-waiting — AGENTS Notes

> UnobligatedRascal private notes. Technical decisions, lessons learned, next steps.

## Mascot & Branding

**Mascot**: Sonic the Hedgehog in his classic "still waiting" pose — arms crossed, foot tapping.
- Source: Sonic the Hedgehog (1991) Title Screen idle animation
- Meaning: Training on Kepler is slow; Sonic is waiting for it to finish
- Placement: Empty state (no jobs), loading spinners, possibly subtle background element
- Implementation: Sprite sheet or simple CSS animation (keep it lightweight)

**Brand color**: Kepler orange `#e8491d` + Sonic blue accent `#00a8ff`
- Primary actions: Kepler orange
- Highlights/links: Sonic blue
- Subtle Sonic silhouette or speed lines as decorative elements

## Hardware Facts (NOUGHT)

- **CPUs**: 2x Xeon E5-2697v4 (36 cores each, Broadwell), NUMA0 + NUMA1
- **RAM**: 128GB DDR4 ECC (64GB per NUMA domain)
- **GPUs**: 8x Tesla K80 GK210, ~11.5GB each, compute 3.7 (Kepler)
- **Topology**: NUMA0 → GPU 0-3; NUMA1 → GPU 4-7 (PIX-linked pairs)
- **CUDA**: Driver 470.256.02, CUDA 11.8 toolkit
- **Storage**: 256GB OS NVMe + 512GB DATA NVMe (TIGHT for training)
- **Network**: `<NOUGHT_IP>` (e.g. 192.168.137.29), accessible from workstation
- **User**: whistler (password stored separately, update nought.config.json if changed)
- **SSH**: Password auth works

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
- GUI: `http://<NOUGHT_IP>:9999/`
- API: `http://<NOUGHT_IP>:9999/v1/`
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

1. **UI refinement (primary focus)** — Unsloth-level comfort, detailed plan below
2. **End-to-end test**: Create job via API/GUI, manually launch worker, verify training loop
3. **Worker auto-spawn**: Orchestrator should launch workers as subprocesses
4. **Worker IPC**: Proper heartbeats and job state sync
5. **Distributed training test**: torchrun across GPUs 1,2,5,6
6. **TUI**: Ratatui for SSH monitoring
7. **Conductor**: Implement median-arc evaluation logic

---

## UI Refinement Plan — "Unsloth-Level Comfort"

**Goal**: Match Unsloth Desktop/Studio's training UX comfort while keeping stack lean (React 18 + Vite + Tailwind only).
**Research**: Unsloth Studio features include no-code training, live observability (loss/gradient/GPU), data recipes, export workflows, training history.
**Decision**: Stay SPA, single-binary deploy. Add lightweight libs only where they save significant dev time.

### Library Decisions

| Need | Choice | Rationale |
|------|--------|-----------|
| Charts | `lightweight-charts` (TradingView) | ~40KB gzipped, GPU-accelerated canvas, perfect for live loss curves. Recharts is heavier (~150KB). |
| Combobox | Pure React + Tailwind | Model picker is simple enough; avoids headless-ui dependency. |
| Icons | Inline SVG or `lucide-react` | Minimal, no font overhead. |
| Form state | Local React state only | No form library needed for this complexity. |

### Implementation Phases (ordered by impact/effort)

#### Phase 1: Job Creation Comfort

**Components**: `JobForm` rewrite → `QuickJobForm` + `AdvancedSettings`

- **Progressive disclosure**:
  - Default strip: Model picker + Dataset path + "Start with defaults" button
  - Expandable "Advanced" toggle: LoRA rank/α, seq length, batch/accum, LR, target modules
  - Pre-filled Kepler-safe defaults (F32, seq_len=512, batch=2, accum=8)
- **Model picker** (replaces plain text input):
  - Searchable combobox with HF-style fuzzy match
  - Pre-listed safe models: `Qwen/Qwen2.5-0.5B-Instruct`, `TinyLlama/TinyLlama-1.1B`, etc.
  - Badges: "fits partial GPUs", "needs full K80", "will OOM" based on known model sizes
  - Optional VRAM probe: hit `/v1/system/status`, filter models by available GPU memory
- **Live VRAM estimate**:
  - Calculate approx VRAM needed: model_size + optimizer_states(2x) + batch_buffer
  - Show badge: "~2.1GB needed — safe on partial GPUs ✓" or "~14GB needed — requires full GPU"
- **Presets/recipes**:
  - JSON presets loaded into form: "Quick Test", "Kepler Safe", "Full GPU", "Conversational"

**Effort**: Low. Pure frontend, no backend changes.

#### Phase 2: Live Metrics & Job Detail

**Components**: `JobDetail` expansion + `LossChart`

- **Loss sparkline/mini-chart**:
  - `lightweight-charts` AreaChart rendering checkpoint metrics array
  - Auto-scrolls as new checkpoints arrive
  - Tooltip on hover: step, loss, timestamp
- **Job progress bar**:
  - Visual: current_step / target_steps with %
  - ETA estimate: (target_steps - current) / (steps_per_second)
- **Checkpoint timeline** (unique to our stack — lean into it):
  - Table/list of checkpoints with: step, loss, timestamp
  - Actions: "Resume from here", "Snip to this" (conductor commands)
  - Click to view in detail panel
- **Conductor surface**:
  - Text area + structured buttons: "Adjust LR", "Snip to checkpoint", "Inject preference"
  - Makes human-in-the-loop first-class, not hidden behind API calls

**Effort**: Low-Med. `lightweight-charts` integration is straightforward.

#### Phase 3: Real-time Logs

**Components**: `LogPane` (collapsible)

- **Streaming logs**:
  - Backend: Add `/v1/training/jobs/:id/logs` endpoint (SSE or polled JSON)
  - Worker: Stream training logs to file or memory buffer
  - Frontend: Tail-like view, auto-scroll, filter by level
  - Start with polling (every 2s) to avoid WebSocket complexity
- **Visual polish**:
  - Color-coded log levels (INFO green, WARN yellow, ERROR red)
  - Timestamps, step numbers highlighted

**Effort**: Med. Requires backend endpoint + worker log streaming.

#### Phase 4: Visual Polish (low cost, high impact)

- **Status pills**: Icons + consistent styling (running=green pulse, paused=orange, etc.)
- **Skeleton loaders**: Shimmer effect while fetching jobs/status
- **Empty states**: Better messaging when no jobs, no checkpoints, etc.
- **Contrast tweaks**: Tighten dark theme, consistent Kepler accent (#e8491d)
- **Mobile-friendly**: Ensure all panels work on smaller viewports

**Effort**: Low. Pure CSS/Tailwind.

#### Phase 5: Backend Extensions (when needed)

- **Job logs endpoint**: `/v1/training/jobs/:id/logs` (SSE or JSON array)
- **VRAM probe**: Enhance `/v1/system/status` with per-GPU free memory
- **Export trigger**: `/v1/training/jobs/:id/export?fmt=gguf` (calls existing export)
- **Dataset preview**: `/v1/datasets/preview?path=...` returns sample rows

**Effort**: Low-Med. Small Rust endpoints.

### Decision Log

- **No heavy chart lib**: Lightweight-charts over Recharts/Visx for performance on slow hardware.
- **No router**: Single-page SPA works; hash-based routing only if multiple "pages" needed later.
- **No WebSocket initially**: Polling is simpler and sufficient for 5s refresh cadence.
- **No dataset recipes yet**: Unique Unsloth feature but requires significant backend work; phase 2+.
- **Preserve single-binary deploy**: All frontend served from orchestrator; no separate static server.

---

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
