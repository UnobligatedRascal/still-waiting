# still-waiting — Top-Down / Bottom-Up Code Review

> UnobligatedRascal — Ancient hardware, fresh ambition.
> Review date: 2026-09-25

---

## Executive Summary

Architecture is sound and well-suited to Kepler constraints. NF4 kernel implementation is impressive. Found **4 critical bugs**, **6 performance issues**, and **12 design/concern items**. Most are fixable without major refactoring.

---

## 1. Critical Bugs

### BUG-1: `train_step()` returns after first gradient-accum boundary — skips remaining epochs

**File**: `python/backends/transformers_backend.py`, line ~582

**Problem**: `train_step()` iterates over `self.data_loader`, accumulates gradients, and when `samples_processed % grad_accum_steps == 0`, it **returns immediately** after one optimizer step. This means:
- Only the first batch group of the epoch is trained
- The function is called repeatedly from `worker.py`'s while loop, so it trains batch-by-batch forever
- The epoch/shuffle logic of `DistributedSampler` is bypassed — no epoch boundaries, no sampler reset
- Scheduler steps are correct, but dataset coverage is wrong (never sees all data in an epoch)

**Impact**: Training runs, but dataset iteration is fundamentally broken. May converge poorly or overfit to first batches.

**Fix**: Move the batch loop INTO worker.py, or redesign `train_step()` to accept a batch. The current name "train_step" implies one step, but it contains a full epoch loop that returns early.

**Recommended fix**: Make `train_step()` do exactly one step (one batch's worth of accumulation, OR one optimizer step with explicit batch fetching). Worker loop should iterate over data_loader:

```python
# In backend:
def train_step(self, batch) -> Dict[str, float]:
    """Train on a single batch (with accumulation tracking)."""
    ...

# In worker.py:
for epoch in range(num_epochs):
    for batch in backend.data_loader:
        metrics = backend.train_step(batch)
        current_step += 1
```

### BUG-2: NF4 quantization on CPU may create codebook on wrong device

**File**: `python/kernels/nf4_kepler.py`, `quantize_to_nf4()`

**Problem**: Line `NF4_CODEBOOK = torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32, device=weight.device)` is correct, but this function is called during model loading when weights might be on CPU first. The function works fine on CPU, but the CUDA kernel (`_nf4_linear_cuda`) is called during `forward()` when tensors are on GPU. If `from_linear()` is called on CPU-loaded weights and layer is moved to GPU later via `model.to("cuda")`, the NF4 buffers ARE moved correctly (they're registered buffers). **This is actually fine** — buffers move with `.to()`. Not a real bug, but worth documenting.

**Status**: False alarm — verified safe. Buffers registered via `register_buffer()` move with `.to()`.

### BUG-3: `LossChart.jsx` uses module-level mutable cache — breaks with multiple charts

**File**: `gui/src/components/LossChart.jsx`, line ~146

**Problem**: `_chartCache` is a global variable. If two `LossChart` components exist on the same page (e.g., comparing two jobs), they share the cache. The second chart gets cached data from the first.

**Impact**: Minor — currently only one LossChart is shown (in JobDetail modal), but future multi-chart comparison would be broken.

**Fix**: Use React `useMemo` with proper dependency instead of module-level cache.

### BUG-4: worker.py `train_step` loop never exhausts data_loader — infinite training on first epoch

**File**: `python/worker.py`, `run_job()` loop + `backend.train_step()`

**Problem**: As described in BUG-1, `train_step()` returns after the first accumulator boundary. Worker.py's `while current_step < target_steps` calls `train_step()` repeatedly, which always starts at the beginning of `data_loader`. The iterator is never exhausted, never shuffled, never epoch-reset.

**Impact**: Training never cycles through the dataset. It sees the same first batches repeatedly.

**Fix**: See BUG-1 fix. Need proper epoch/batch iteration.

### BUG-5: Gradient clipping on ALL parameters when NF4 buffers are frozen

**File**: `python/backends/transformers_backend.py`, line ~577

**Problem**: `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)` clips all parameters, but NF4 weights have no gradients (they're buffers, not parameters). This is technically fine since `parameters()` only yields `nn.Parameter`s, not buffers. **Not a real bug** — PyTorch's `.parameters()` correctly excludes buffers.

**Status**: Verified safe.

---

## 2. Performance Issues

### PERF-1: NF4 quantization is O(n×16) in pure Python — slow for 7B+ models

**File**: `python/kernels/nf4_kepler.py`, `quantize_to_nf4()`

**Problem**: The distance computation `(scaled.unsqueeze(2) - NF4_CODEBOOK.unsqueeze(0).unsqueeze(0)).abs()` creates a tensor of shape `(n_blocks, 64, 16)` which for a 7B model with all layers quantized could be ~2.8GB of temporary tensor memory just for quantization. The argmin over 16 values is fine, but the broadcast creates huge intermediates.

**Impact**: Quantization of a 7B model takes extra RAM and time. One-time cost during model load, but noticeable.

**Fix**: Use a vectorized search or binary search on the sorted codebook. Since NF4 codebook is sorted (always), binary search per element is O(n×4) instead of O(n×16). Or use `torch.searchsorted`:

```python
# Faster quantization using searchsorted (codebook is sorted)
# Scale values are in [-1, 1], codebook covers this range
indices = torch.searchsorted(NF4_CODEBOOK, scaled) - 1
indices = torch.clamp(indices, 0, 15)
```

### PERF-2: `dequantize_nf4` CPU fallback uses `torch.stack` + `reshape` — avoidable allocation

**File**: `python/kernels/nf4_kepler.py`, `dequantize_nf4()`

**Problem**: `torch.stack([high, low], dim=-1).reshape(-1)` creates an intermediate tensor. For CPU fallback (used in testing), this is fine. But the pattern can be streamlined with `torch.interleave` or direct indexing.

**Impact**: Minor — CPU path only, used in testing.

### PERF-3: CUDA kernel could be faster with shared memory optimization

**File**: `python/kernels/nf4_dequant.cu`

**Problem**: The kernel loads the codebook into shared memory correctly, but with only 16 floats, this is negligible. The real bottleneck is that we dequantize to FP32 then do `torch.mm` — there's a memory write/read cycle. A truly fused kernel would dequantize directly into the GEMM, but that's significantly more complex.

**Assessment**: Current approach (dequantize → torch.mm) is pragmatic and correct. The fused kernel would need to handle cuBLAS integration manually or use cutlass-style tiling. For Kepler sm_37 without tensor cores, the gain is likely <20% for much more complex code. **Decision: keep current design.**

### PERF-4: No preemption of CUDA extension recompilation — every worker startup rebuilds

**File**: `python/kernels/nf4_cuda.py`

**Problem**: `torch.utils.cpp_extension.load()` JIT-compiles on every import. With `verbose=False` and caching in `/tmp/torch_extensions/`, subsequent runs in the same process reuse the cached .so. But each fresh Python process (each worker) recompiles unless the cache survives.

**Impact**: ~10-30 second delay on first NF4 kernel use per worker process. Annoying but one-time.

**Fix**: Pre-compile the extension once and import the .so directly. Or use `torch.utils.cpp_extension.load()` with `is_stable=True` to cache more aggressively.

### PERF-5: `replace_linear_with_nf4` uses dict lookup for every module — O(n²) parent resolution

**File**: `python/kernels/nf4_kepler.py`, `replace_linear_with_nf4()`

**Problem**: `dict(model.named_modules())[parent_name]` creates a full dict of all modules for EVERY module being checked. This is called inside a loop over all named modules.

**Impact**: Negligible for 0.5B model, but for 7B with thousands of modules, this is slow.

**Fix**: Create the dict ONCE before the loop:

```python
module_dict = dict(model.named_modules())
for name, module in model.named_modules():
    parent = module_dict[parent_name]
```

### PERF-6: Worker sends logs via synchronous `requests.post()` — blocks on slow network

**File**: `python/worker.py`, `send_log()`

**Problem**: Every log POST blocks the training loop until HTTP completes or times out (2 seconds). With log spam, this adds latency. The `timeout=2` helps but doesn't eliminate the issue.

**Impact**: Minor — logs are sent every 16 steps, not every batch. 16 steps might take 10-60 seconds on Kepler, so a 0.1s HTTP call is fine.

---

## 3. Design Decisions — Review

### DEC-1: Custom LoRA wrapper instead of PEFT — CORRECT CHOICE

**Assessment**: Excellent decision. PEFT requires `isinstance(nn.Linear)` check and would force dummy FP32 weights for NF4 layers, wasting the VRAM savings we worked for. Custom `NF4LoRALayer` is cleaner and more VRAM-efficient.

**Verdict**: ✅ Keep. Well-implemented.

### DEC-2: NF4 weights as buffers, not parameters — CORRECT CHOICE

**Assessment**: Using `register_buffer()` for frozen quantized weights is exactly right. Buffers:
- Move with `.to()` and `.cuda()`
- Don't appear in `.parameters()` — optimizer skips them
- Can be saved/loaded via state_dict
- Are not part of gradient computation

**Verdict**: ✅ Keep. Standard QLoRA pattern.

### DEC-3: No DDP worker auto-spawn — PAIN POINT

**Assessment**: Current design forces users to manually run `numactl + torchrun` commands. This is error-prone and breaks the "GUI → training" dream.

**Verdict**: ⚠️ Fix needed. High priority.

**Recommendation**: Orchestrator should spawn workers as subprocesses. See limitations-roadmap.md P1. Design is documented; implementation just hasn't happened.

### DEC-4: Checkpoint format — `save_pretrained()` is wrong for NF4

**File**: `python/backends/transformers_backend.py`, `save_checkpoint()`

**Problem**: `model_to_save.save_pretrained(path)` uses HuggingFace's default serialization, which saves the LinearNF4 state_dict as-is. This works but:
1. Doesn't use safetensors by default (security/size concern)
2. Saves the whole model config every time — redundant
3. NF4 packed format + scales are saved correctly, but a standard HF load won't know how to reconstruct LinearNF4 without our custom class

**Verdict**: ⚠️ Needs fix.

**Recommendation**: 
- Save with `safe_serialization=True`
- Include a marker file indicating NF4 format
- Save optimizer state separately (for resume)
- Document custom load procedure

### DEC-5: No scheduler persistence across checkpoints

**Problem**: Checkpoints don't save optimizer or scheduler state. Resuming would restart AdamW moments from scratch.

**Verdict**: ⚠️ Needs fix for long-running training.

### DEC-6: F32 compute on Kepler — CORRECT CHOICE

**Assessment**: FP16 on Kepler is actually SLOWER than FP32 because there are no tensor cores and the FP16 path uses software emulation or suboptimal instructions. F32 + cuBLAS legacy is the right call.

**Verdict**: ✅ Correct. Documented and defended.

### DEC-7: Single-binary deploy (Rust orchestrator serves GUI) — CORRECT CHOICE

**Assessment**: Excellent. Single artifact, single port, zero Docker. Matches the "ancient hardware, zero fuss" ethos.

**Verdict**: ✅ Keep.

### DEC-8: No database — in-memory state only

**File**: `orchestrator/src/state.rs`

**Problem**: All jobs, logs, and checkpoints are stored in memory. Orchestrator restart = all state lost.

**Verdict**: ⚠️ Acceptable for now (checkpoints on disk are the source of truth), but should add persistence.

**Recommendation**: Minimal JSON file persistence for job metadata. Don't persist logs (too large).

### DEC-9: LossChart module-level cache — BAD PATTERN

**Verdict**: ⚠️ Fix (see BUG-3). React state management should be component-local.

### DEC-10: `generate_nf4_codebook()` exists but isn't used — DEAD CODE

**File**: `python/kernels/nf4_kepler.py`

**Problem**: The codebook generation function produces values that differ from the official bitsandbytes codebook. The comment even says "differences are expected." But the official codebook is used everywhere. The generation function is dead code.

**Verdict**: ⚠️ Either fix to match official values, or delete. Don't leave misleading code.

**Recommendation**: Delete `generate_nf4_codebook()` and friends, or add it as a validation utility that asserts match against official codebook.

### DEC-11: Log storage cap of 5000 per job — TOO LOW for long training

**File**: `orchestrator/src/state.rs`, `add_log()`

**Problem**: 5000 log entries with 16-step logging = ~80,000 steps of logs retained. That's actually reasonable for most training runs. Not a bug, but worth noting.

**Verdict**: ✅ Acceptable.

### DEC-12: Conductor as stub — WORTH BUILDING

**Assessment**: The conductor concept (human-in-the-loop training guidance) is unique and valuable. Currently it's just a stub that returns `None`.

**Verdict**: Keep the hook. Implement later when core training is validated.

---

## 4. Code Quality Issues

### QC-1: `train_step()` name is misleading

The method iterates over the entire data_loader and returns after first step. Name implies one step; behavior is one batch group from an epoch loop. Rename or restructure.

### QC-2: Unused imports in transformers_backend.py

`math` and `random` are imported but never used. `Tuple` in typing is unused. Clean up.

### QC-3: No type hints on Rust API handlers

The Rust code is well-structured but API handlers lack doc comments on request/response schemas. Add inline docs or OpenAPI spec.

### QC-4: Hardcoded checkpoint path `/data/checkpoints/`

**File**: `python/worker.py`, line ~186

`ckpt_path = f"/data/checkpoints/{job_id}/step_{current_step}"` assumes `/data/` exists. Should be configurable via env var.

### QC-5: JobForm sends `target_modules` as comma-separated string, backend expects list

**File**: `gui/src/components/JobForm.jsx` vs `transformers_backend.py`

GUI sends `"q_proj,v_proj,k_proj"` as string. Backend does `cfg.get("target_modules")` and expects list. This will fail.

**Fix**: Either split on comma in backend or join as array in frontend.

### QC-6: Missing error handling in worker.py main()

Line ~256 has a dangling tuple syntax: `send_log(job_id, "INFO", f"Worker started: LOCAL_RANK={local_rank}, WORLD_SIZE={get_world_size()}, NUMA={NUMA_NODE}"), {`

The trailing `), {` is invalid Python syntax.

### QC-7: No graceful shutdown in orchestrator

The orchestrator doesn't handle SIGTERM/SIGINT for clean shutdown (flush logs, save state, kill workers).

### QC-8: LossChart `useMemoizedChartData` uses `useMemoizedChartData` as function name but isn't a hook

Naming convention violation — it's not a custom hook but named as if it is. Rename to `getChartData` or similar.

### QC-9: NF4 test suite doesn't test DDP path

All tests are single-GPU. DDP behavior with LinearNF4 is untested.

---

## 5. Security/Reliability

### SEC-1: No authentication on orchestrator API

All endpoints are open. On a trusted LAN this is fine, but worth noting.

### SEC-2: No input validation on job creation

`CreateJobRequest` accepts any string for `model`, `target_steps`, etc. Malicious input could cause resource exhaustion.

### SEC-3: Base64 deployment uploads fail silently for >50KB

Documented in AGENTS.md, but the deployment tooling has this silent failure mode. Use SFTP instead (already noted).

---

## 6. Fixes Applied During Review

| ID | Issue | Fixed | Notes |
|----|-------|-------|-------|
| BUG-6 | Syntax error in worker.py line 256 | ✅ | Trailing `), {` removed |
| BUG-3 | LossChart module-level cache | ✅ | Replaced with `useMemo` |
| PERF-5 | O(n²) parent lookup in replace_linear_with_nf4 | ✅ | Pre-compute module dict |
| DEC-5 | Scheduler/optimizer not saved in checkpoint | ✅ | Added optimizer/scheduler save to save_checkpoint |
| DEC-4 | Checkpoint format not safetensors by default | ✅ | Changed save_checkpoint to use safe_serialization |
| QC-6 | JobForm target_modules string vs list mismatch | ✅ | Backend now splits on comma |
| QC-4 | Hardcoded /data/checkpoints path | ✅ | Made configurable via CHECKPOINT_DIR env var |
| DEC-10 | Dead code: generate_nf4_codebook + unused helpers | ✅ | Removed; kept only OFFICIAL_NF4_CODEBOOK |
| QC-8 | LossChart `useMemoizedChartData` misleading name | ✅ | Removed entirely, inlined into useMemo |
| NF4 TODO | Stale TODO list in nf4_kepler.py | ✅ | Marked integration as done |
| PERF-1 | NF4 quantization O(n×16) broadcast | ✅ | Replaced with O(n×4) searchsorted |
| QC-2 | Unused imports in transformers_backend.py | ✅ | Removed math, random, Tuple |
| BUG-1/BUG-4 | train_step() returns after first grad-accum boundary | ✅ | Renamed train_step() → train_epoch(), full epoch iteration, step_callback for monitoring, worker loops over epochs |

---

## 7. Prioritized Action Items

### P0 (Block NF4 7B test)
1. ✅ Fix worker.py syntax error
2. ✅ Fix LossChart cache
3. ✅ Fix target_modules string/list mismatch
4. ✅ Fix train_step()/data_loader iteration logic (BUG-1, BUG-4)
5. ✅ Add safetensors + optimizer state to checkpoints
6. Pre-compile NF4 extension or cache aggressively

### P1 (Before production use)
7. Implement worker auto-spawn from orchestrator
8. Add checkpoint resume capability
9. Add orchestrator state persistence (JSON)
10. Add graceful shutdown handling
11. Fix NF4 codebook dead code (delete or validate)
12. Test NF4 + DDP path

### P2 (Nice to have)
13. OpenAPI spec for orchestrator API
14. Input validation on job creation
15. Log rotation/archival (5000 cap is per-job in-memory)
16. Authentication/authorization layer
17. GPU availability-aware job scheduling

---

## 8. What We Did Right (Keep Doing)

1. **NF4 custom kernels**: Correct approach for Kepler. No compromises on VRAM savings.
2. **Buffers for frozen weights**: Standard, correct, clean.
3. **Custom LoRA wrapper**: Avoids PEFT's nn.Linear requirement and dummy weight overhead.
4. **F32 compute**: Right call for Kepler (FP16 is slower without tensor cores).
5. **Single-binary deploy**: Rust orchestrator serving GUI = zero friction.
6. **NUMA awareness**: Documented and enforced. Critical for this topology.
7. **Documentation**: AGENTS.md, limitations-roadmap.md, README are thorough.
8. **Test suite**: test_nf4.py covers the full NF4 stack.

---

## 9. Future Recommendations

### If we're serious about 7B+ on single K80:
- CPU-offloaded AdamW optimizer states (saves 2× model weight VRAM)
- Activation checkpointing (recompute activations instead of storing — trade time for VRAM)
- Gradient accumulation with very large effective batch (already supported, just tune)

### For 32B+:
- FSDP is mandatory (already documented in limitations-roadmap.md P2)
- Consider ZeRO-3 style optimization

### For usability:
- Worker auto-spawn is THE missing piece. Everything else works; this is the friction point.
- One-click "start training from GUI" = orchestrator spawns worker subprocess.

---

*UnobligatedRascal — We rebuild what the world abandoned.*
