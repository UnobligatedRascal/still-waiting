# still-waiting — Technical Limitations & Roadmap

> UnobligatedRascal — Ancient hardware, fresh ambition.
>
> **Guiding principle**: NOUGHT is our constraint and our advantage. Design for it — use all 92GB VRAM, all 72 CPU cores, all 128GB RAM. Exploit the topology. Don't fight it.

**Last updated**: 2026-09-21

---

## NOUGHT Topology — Know Your Enemy

```
NUMA0 (Socket 0)                    NUMA1 (Socket 1)
┌─────────────────────────┐         ┌─────────────────────────┐
│ Xeon E5-2697v4 (18C/36T)│         │ Xeon E5-2697v4 (18C/36T)│
│ DDR4-2133 ECC (64GB)    │         │ DDR4-2133 ECC (64GB)    │
│                         │         │                         │
│ GPU 0: K80 (11.5GB) ◄───┼──PIX──┐ │ GPU 4: K80 (11.5GB) ◄───┼──PIX──┐
│ GPU 1: K80 (11.5GB) ◄───┼───────┘ │ GPU 5: K80 (11.5GB) ◄───┼───────┘
│ GPU 2: K80 (11.5GB) ◄───┼───────┐ │ GPU 6: K80 (11.5GB) ◄───┼───────┘
│ GPU 3: K80 (11.5GB) ◄───┼──PIX──┘ │ GPU 7: K80 (11.5GB) ◄───┼──PIX──┘
└─────────────────────────┘         └─────────────────────────┘
         NCCL (PCIe)  ──────────────────────  NCCL (PCIe)
```

**Key facts**:
- **sm_37 compute**: No tensor cores, no native FP16 speedup, no BF16
- **cuBLAS legacy only**: `Sgemm`, not `SgemmEx`. No `CUBLAS_STATUS_ARCH_MISMATCH`-prone APIs.
- **NUMA matters**: Cross-socket memory access is ~2.5× slower. Pin workers to NUMA domain.
- **PIX pairs**: GPUs 0-3 and 4-7 are PIX-linked for higher bandwidth within each pair.
- **NCCL works**: But slower than modern NVLink. DDP across 8 GPUs is viable but not blazing.

**Design rule**: Every GPU-local operation should stay GPU-local. Every NUMA-local operation should stay NUMA-local. Cross-socket communication is your enemy — minimize it.

---

## Current Limitations

### 1. No QLoRA (4-bit NF4 quantization) — CONFIRMED BROKEN ON KEPLER

**Status**: Tested bitsandbytes 0.50.2 (latest) on NOUGHT Kepler sm_37 — **runtime failure**.

**Test result**: `Error named symbol not found at line 74 in file /src/csrc/ops.cu`

**Root cause**: bitsandbytes NF4 CUDA kernels use PTX instructions not available on compute capability 3.7. Official minimum is CC 6.0 (Pascal). Older bitsandbytes 0.43.x-0.44.x had Kepler source support but those tags are gone from GitHub, and the NF4 kernels likely still require newer instructions.

**Impact**:
| Model | F32 VRAM per GPU | Fits on single K80 (11.5GB)? |
|-------|------------------|-------------------------------|
| Qwen2.5-0.5B | 2GB | ✅ Yes |
| Qwen2.5-1.5B | 6GB | ✅ Tight |
| Qwen2.5-3B | 12GB | ❌ No |
| Qwen2.5-7B | 28GB | ❌ No |
| Qwen2.5-14B | 56GB | ❌ No |

**Current workaround**: FP16 storage + FP32 compute reduces load time but NOT training VRAM (gradients + optimizer states are FP32 regardless). For 7B+ training, use multi-GPU DDP.

**Path to real QLoRA on Kepler**:
- Write custom NF4-style 4-bit kernels for sm_37 (Phase 2)
- Based on llama_wukong's proven pattern: FA kernels, AdamW kernel all ported
- Scope: ~2-3 weeks of focused CUDA work
- Reward: 7B models on single K80, 14B+ viable

**Priority**: HIGH — but requires custom CUDA implementation.

---

### 2. Worker auto-spawning not implemented

**Status**: Workers must be launched manually via `numactl + torchrun`.

**Impact**: User must know exact command-line syntax, NUMA pinning, and torchrun flags. Error-prone.

**Proposed fix**:
- Orchestrator spawns workers as child processes
- Job config includes `gpus: [0,1,2,3,4,5,6,7]` or `num_gpus: 8`
- Orchestrator computes NUMA split and launches:
  - `numactl --cpunodebind=0 torchrun --nproc_per_node=4 worker.py ...`
  - `numactl --cpunodebind=1 torchrun --nproc_per_node=4 worker.py ...`
- Worker reports PID back to orchestrator for lifecycle management

**Priority**: HIGH — enables end-to-end workflow via UI.

---

### 3. No FSDP / model sharding

**Status**: DDP only (full model replicated on every GPU).

**Impact**: Cannot train models larger than single-GPU VRAM allows (even with QLoRA).

**Proposed fix**:
- Add `torch.distributed.fsdp` support
- Shard model weights across GPUs instead of replicating
- Enables 32B+ training: 72B NF4 = 36GB total, sharded across 8 GPUs = ~4.5GB/GPU

**Tradeoff**: FSDP adds communication overhead. On Kepler with slow NCCL, expect ~30-50% slower per-step.

**Priority**: MEDIUM — needed for 32B-72B models.

---

### 4. No cross-NUMA orchestration

**Status**: Each `torchrun` process is independent. No coordination between NUMA domains.

**Impact**: Running 8 GPUs requires two separate torchrun commands with manual `master_addr/master_port` coordination.

**Proposed fix**:
- Single torchrun invocation across all 8 GPUs with proper `--nnodes=1 --nproc_per_node=8`
- Use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` (all GPUs)
- Pin CPU threads per rank to appropriate NUMA socket using `taskset` or `numactl --physcpubind`
- OR: Use separate torchrun per NUMA domain with orchestrator as master

**Priority**: MEDIUM — usability + correctness.

---

### 5. No dataset recipes / data management

**Status**: Only raw JSONL paths or HF dataset IDs.

**Impact**: Users must format data manually. No preview, no validation, no templating.

**Proposed fix** (future, Unsloth-style):
- Dataset preview endpoint: `/v1/datasets/preview?path=...`
- Built-in templates: "conversational", "code", "instruction"
- Validation: check format, token length distribution, etc.

**Priority**: LOW — nice to have, not blocking.

---

### 6. No checkpoint resuming mid-training

**Status**: Checkpoints are saved but cannot be loaded to resume training.

**Impact**: Interrupted training = start from scratch.

**Proposed fix**:
- `load_checkpoint()` in backend: load LoRA adapter weights + optimizer state
- Job config: `resume_from: "/path/to/step_N"`

**Priority**: MEDIUM — long training jobs need this.

---

### 7. Conductor is a stub

**Status**: `evaluate_and_maybe_edit()` always returns `None`. No median-arc logic.

**Impact**: No human-in-the-loop training guidance. Checkpoints are "set and forget".

**Proposed fix**:
- Implement median progress arc: compare actual loss curve to expected shape
- If loss diverges > threshold → recommend `snip_to:<earlier_checkpoint>`
- UI surface for conductor instructions already exists in JobDetail modal

**Priority**: LOW — unique feature but not core training.

---

## Proposed Implementations (Ordered by Impact)

### P0: FP16 Storage Support (DONE)

**What**: Added `model_precision` config option to `transformers_backend.py`.

**Modes**:
- `f32`: Full FP32 (current default, safe, most VRAM)
- `f16_storage`: Load weights in FP16, cast to FP32 for compute. Saves load time, NOT training VRAM.
- `bnb_nf4`: bitsandbytes NF4 QLoRA (integrated but BROKEN on Kepler — see Limitation #1)

**Tested**: ✅ FP16 storage + LoRA training works on Qwen2.5-0.5B.

**Limitation**: For training, gradients and optimizer states are FP32 regardless, so FP16 storage doesn't reduce training VRAM significantly. For real VRAM savings in training, we'd need FP16 compute too.

**Code**: See `transformers_backend.py::prepare()` — 60 lines added.

---

### P0-alt: Custom NF4 Kernels for Kepler sm_37 ("MAKE it work")

**What**: Write our own 4-bit quantization kernels targeting sm_37, bypassing bitsandbytes entirely.

**Scope**:
1. **NF4-style quantization**: Distribution-aware 4-bit quantization (match QLoRA paper)
2. **Dequantize kernel**: CUDA kernel that dequantizes 4-bit weights to FP32 for compute
3. **4-bit matmul kernel**: Fused dequantize+matmul using legacy cuBLAS Sgemm (no Ex APIs)
4. **Linear4bit layer**: Drop-in replacement for torch.nn.Linear
5. **Gradient path**: Handle gradient computation through quantized weights
6. **Integration**: Plug into `transformers_backend.py` as a new `model_precision="custom_nf4"` mode

**Based on**: llama_wukong's proven pattern — FA tile kernels and AdamW GPU kernel both ported for sm_37.

**Expected result**:
| Model | Custom NF4 VRAM | Fits on single K80? |
|-------|-----------------|---------------------|
| Qwen2.5-7B | ~3.5GB | ✅ Yes |
| Qwen2.5-14B | ~7GB | ✅ Yes (tight) |
| Qwen2.5-32B | ~16GB | ❌ Needs multi-GPU |

**Effort**: ~2-3 weeks of focused CUDA development + testing.

**Risk**: Medium. Kepler's lack of certain instructions (e.g., no native 4-bit ops) means we'd use lookup tables for NF4 dequantization, similar to the original QLoRA implementation.

**Decision**: This is the "MAKE it support Kepler" work. Worth it if we want single-GPU 7B training.

### P1: Worker Auto-Spawn

**What**: Orchestrator launches workers instead of user running CLI commands.

**Design**:
- Job config includes `gpus: [0,1,2,3,4,5,6,7]`
- Orchestrator `launch_worker()` function:
  1. Split GPUs by NUMA: NUMA0=[0-3], NUMA1=[4-7]
  2. Spawn first torchrun with `numactl --cpunodebind=0`
  3. Spawn second torchrun with `numactl --cpunodebind=1`
  4. Store PIDs in job state for cleanup
- Worker heartbeat: POST `/v1/training/jobs/:id/heartbeat` every 60s
- Detect dead workers: restart or mark job failed

**Changes**:
- New module: `orchestrator/src/worker_manager.rs`
- New API: `POST /v1/training/jobs/:id/heartbeat`
- Worker.py: add heartbeat timer

---

### P2: FSDP for Large Models

**What**: Shard model weights across GPUs for 32B+ training.

**When**: After QLoRA is working, when user requests 32B/72B training.

**Design**:
- Job config: `sharding: "fsdp"` enables it
- Backend wraps model with `torch.distributed.fsdp.FullyShardedDataParallel`
- Requires PyTorch ≥ 2.1 (we have 2.4.0a0)

**Risk**: FSDP + bitsandbytes 4-bit has compatibility issues. May need to load in FP16 then shard.

---

### P3: Cross-NUMA torchrun

**What**: Single torchrun command spanning all 8 GPUs with proper NUMA awareness.

**Design**:
```bash
# Instead of two separate torchrun commands:
numactl --interleave=all \
  torchrun --nproc_per_node=8 \
  --standalone \
  worker.py job_id config.json

# Inside worker.py, each rank pins its CPU thread:
rank = int(os.getenv("RANK", "0"))
if rank < 4:
    # NUMA0
    os.system(f"numactl --cpunodebind=0 --exec-only python3 worker.py ...")
else:
    # NUMA1
    os.system(f"numactl --cpunodebind=1 --exec-only python3 worker.py ...")
```

---

### P4: Checkpoint Resume

**What**: `resume_from` config option to load previous checkpoint.

**Design**:
```python
# In prepare(), after loading LoRA adapters:
resume_path = cfg.get("resume_from")
if resume_path:
    state = torch.load(f"{resume_path}/training_state.pt")
    self.model.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    self.scheduler.load_state_dict(state["scheduler"])
    self.step_count = state["step"]
```

---

### P5: VRAM Probe + Model Recommendations

**What**: UI shows which models "fit" based on current GPU free memory.

**Design**:
- `/v1/system/status` already calls `nvidia-smi`
- Parse output to get per-GPU free memory
- Frontend: model picker badges update dynamically:
  - "Fits on any GPU ✓" (0.5B-1.5B)
  - "Fits on K80 with QLoRA ✓" (7B-14B after QLoRA fix)
  - "Needs FSDP across 8 GPUs" (32B+)

---

## Hardware Utilization Rules

**These are non-negotiable for serious training on NOUGHT:**

1. **NUMA pinning is mandatory**: Always use `numactl --cpunodebind=X --membind=X` for GPU workloads. Cross-socket memory access kills performance.

2. **Use ALL GPUs**: 8× K80s in DDP gives ~4-6× effective throughput over single GPU (NCCL overhead, Kepler age). Never waste them unless model is small enough for one.

3. **CPU affinity matters**: Pin Python workers and torchrun to specific cores. Avoid context switching across NUMA domains.

4. **Storage is tight**: 512GB DATA NVMe fills fast with checkpoints. Policy:
   - Keep last 5 checkpoints per job
   - Delete completed job checkpoints after 7 days
   - Use `/data/checkpoints/<job_id>/step_N` structure

5. **RAM is plenty**: 128GB system RAM should be used for:
   - Dataset caching (pre-tokenize to RAM where possible)
   - PyTorch DataLoader workers (2-4 per process is safe)

6. **GPU pairs**: When possible, pair workloads on PIX-linked GPUs (0+3, 1+2, 4+7, 5+6) for higher bandwidth.

---

## Quick Reference: Model → GPU Strategy

| Model | Method | GPUs | NUMA Strategy | Expected VRAM/GPU | Status |
|-------|--------|------|---------------|-------------------|--------|
| 0.5B-1.5B | F32 LoRA | 1 | Any NUMA | 4-11GB | ✅ Works |
| 3B | F32 LoRA | 2 (DDP) | Same NUMA | ~12GB each → ❌ | OOM on K80 |
| 3B | F32 LoRA | 2 (FSDP) | Same NUMA | ~6GB each → ✅ | Needs FSDP impl |
| 7B | F32 LoRA | 4 (DDP) | Split NUMA | ~14GB each → ❌ | OOM on K80 |
| 7B | F32 LoRA | 8 (DDP) | Split NUMA0+1 | ~14GB each → ❌ | OOM on K80 |
| 7B | Custom NF4 LoRA | 1 | Any NUMA | ~5-6GB | ⏳ Phase 2 (custom CUDA) |
| 7B | Custom NF4 LoRA | 8 (DDP) | Split NUMA0+1 | ~5-6GB each | ⏳ Phase 2 |
| 14B | Custom NF4 LoRA | 2 (DDP) | Same NUMA | ~10GB each → ✅ tight | ⏳ Phase 2 |
| 14B | Custom NF4 LoRA | 8 (DDP) | Split NUMA0+1 | ~5-6GB each → ✅ | ⏳ Phase 2 |
| 32B | Custom NF4+FSDP | 8 | Split NUMA0+1 | ~5-6GB each → ✅ | ⏳ Phase 2+
| 72B | Custom NF4+FSDP | 8 | Split NUMA0+1 | ~9-10GB each → ✅ tight | ⏳ Phase 2+ |

**Current reality without custom NF4**:
- Single K80: Max ~0.5B-1.5B models comfortably in F32
- Multi-GPU DDP: 7B+ needs FSDP sharding (not yet implemented)
- FP16 storage doesn't help training VRAM (gradients/optimizer are FP32)

---

*UnobligatedRascal — Ancient hardware, fresh ambition.*
