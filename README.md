 # STILL WAITING — Private Model Training on Ancient Hardware

 > "We don't settle for right. We make people forget what boxes even were."
 > — UnobligatedRascal

 ## Purpose

 Turn NOUGHT (Xeon E5-2697v4 + 128GB ECC + 8x Tesla K80 Kepler) into a lean, mean, private LLM training
 rig despite:
 - Hardware age (Broadwell CPUs, 13-year-old Kepler GPUs)
 - Total software ecosystem abandonment of Kepler
 - Headless operation over SSH/LAN/Internet

 **Philosophy**: We've already revived llama.cpp on this hardware with custom matmul pipelines and
 NUMA-aware tensor splitting that doubled performance. We're applying the same playbook to training.
 Whatever a "production team" wouldn't do — patching modern frameworks, building from source for
 deprecated architectures, spending weeks on "one-in-a-billion" optimizations — we do without
 hesitation.

 ## Architecture

 ```
 ┌─────────────────────┐     ┌─────────────────────────────────────────────┐
 │   Your Workstation  │     │              NOUGHT (192.168.137.29)         │
 │                     │     │                                             │
 │  ┌──────────────┐   │     │  ┌────────────────────────────────────────┐ │
 │  │ GUI Client   │◄──┼─────┼──│ agent-orchestrator (Rust, :9999)      │ │
 │  │ (Tauri/Svelte)│   │ HTTP│  │ - OpenAI-compatible API               │ │
 │  └──────────────┘   │ WS  │  │ - Job/Checkpoint registry             │ │
 │                     │     │  │ - Conductor (surgical edit logic)     │ │
 │  ┌──────────────┐   │     │  └──────────────┬───────────────────────┘ │
 │  │ TUI Client   │◀──┼─────┼─────────────────┼───────────────────────── │
 │  │ (Ratatui)    │   │ SSH │                │                           │
 │  └──────────────┘   │     │  ┌─────────────┴───────────────────────┐   │
 │                     │─────│──│ Python Training Workers             │   │
 │                     │─────│──│ - Transformers/PEFT LoRA (Kepler)   │   │
 │                     │     │  └─────────────────────────────────────┘   │
 └─────────────────────┘     └─────────────────────────────────────────────┘
 ```

 ## Hardware Reality Check — NOUGHT

 | Component | Spec | Implications |
 |-----------|------|--------------|
 | CPU | 2x Xeon E5-2697v4 (36 cores each, 72 total) | Dual NUMA: NUMA0→GPU0-3, NUMA1→GPU4-7 |
 | RAM | 128GB DDR4 ECC (64GB per NUMA domain) | Plenty for LoRA up to 13B params |
 | GPU | 8x Tesla K80 GK210 Kepler (~11.5GB each, sm_3.7) | ANCIENT. No tensor cores. CUDA 11.8 toolkit.
 |
 | Storage | 225GB root NVMe (~84GB free) | TIGHT. Compress checkpoints. |
 | Network | LAN/WAN | GUI over HTTP+WebSocket; SSH for TUI |

### Current State
- Production llama_wukong running (using ~8-9GB per GPU)
- ~1-2GB free per GPU for training experiments
- Python 3.11.2, Rust 1.98.1 installed
- PyTorch 2.4.0a0+sm_37 built from source and working
- transformers 4.40.0 + peft 0.7.0 installed
- LoRA training validated on Qwen2.5-0.5B-Instruct

 ## Proven: What Works on Kepler sm_37 (from llama_wukong)

 We've already fought these battles. Here's what we know works:

 ### cuBLAS
 - **Legacy APIs only**: `cublasSgemm`, `cublasSgemmStridedBatched`, `cublasSgemmBatched` work
 - **Ex APIs crash**: `cublasGemmEx`, `cublasGemmStridedBatchedEx` → `CUBLAS_STATUS_ARCH_MISMATCH`
 - **No tensor ops**: `CUBLAS_DEFAULT_MATH`, not `CUBLAS_TF32_TENSOR_OP_MATH`

 ### Flash Attention
 - **Tile/vector kernels work** on sm_37 (no tensor cores needed, just slower)
 - Supported head sizes: 40, 64, 72, 80, 96, 112, 128, 192, 256, 320, 512, 576
 - Source: `llama_wukong/ggml-cuda/fattn.cu` — portable to PyTorch if needed

 ### Optimizer
 - **AdamW GPU kernel proven**: `llama_wukong/ggml-cuda/opt-step-adamw.cu` runs on Kepler
 - GPU-side optimizer saves PCIe bandwidth vs CPU fallback

 ### Distributed Training
 - **NCCL works** across 8 GPUs with tensor splitting
 - **NUMA replication = 2.5x speedup**: Pin workers to NUMA domains with `numactl --cpunodebind=X
 --membind=X`
 - NUMA0 = GPU 0-3, NUMA1 = GPU 4-7 (PIX-linked pairs)

 ### Quantization
 - **MMQ (quantized matmul) works**: Kepler's DP4A instructions handle integer matmul
 - **TurboQuant KV cache**: 2.5-4.25 bpw compression proven in inference; applicable to training caches

 ## PyTorch: Building for the "Impossible"

 Official PyTorch 2.0+ dropped Kepler support. We're building 2.4.0-rc8 from source with sm_37
 re-enabled.

 ### Build Command (PROVEN WORKING)
 ```bash
 cd /home/whistler/pytorch-kepler && \
 export CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11" && \
 export PATH=/usr/local/cuda-11.8/bin:$PATH && \
 export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH && \
 export TORCH_CUDA_ARCH_LIST="3.7" && \
 export MAX_JOBS=36 && \
 export USE_CUDA=1 && \
 export CUDA_HOME=/usr/local/cuda-11.8 && \
 export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11" && \
 export CUDAFLAGS="-ccbin /usr/bin/g++-11" && \
 export CUDAHOSTCXX=/usr/bin/g++-11 && \
 export CXX=/usr/bin/g++-11 && \
 export CC=/usr/bin/gcc-11 && \
 python3 setup.py develop
 ```

 ### Alternative: Prebuilt Wheel (xiaoran007)
 - PyTorch 2.4.0-rc8 wheel for sm_35/sm_37 available at https://github.com/xiaoran007/Pytorch-for-Kepler
 - Built for Python 3.9 only; requires separate Python install

 ### Fallback: PyTorch 1.13/1.14
 - Last official releases with sm_37 support
 - Missing modern features but proven to work

 ## Training Strategy

 ### Model Targets
 - **Immediate**: Sub-billion models (0.5B-1B) for validation with limited VRAM
 - **Primary**: 7B-13B LoRA fine-tuning distributed across 8 GPUs (full VRAM available)

 ### Technical Approach
 1. **F32 compute**: FP16 is slow on Kepler without tensor cores
 2. **LoRA adapters**: Only train adapter weights; keep base model frozen
 3. **DDP distribution**: torch.distributed across 8 GPUs
 4. **Gradient checkpointing**: Trade compute for memory
 5. **Checkpoint every 2048 steps**: Dense checkpoints enable surgical edits

 ### NOT Compatible (save time)
 - Unsloth (requires Ampere+)
 - QLoRA 4-bit (unstable on Kepler)
 - Any tensor-core-only operations

 ## Project Structure

 ```
 still-waiting/
 ├── README.md                  # this file
 ├── orchestrator/              # Rust API server
 │   ├── Cargo.toml
 │   └── src/
 │       ├── main.rs            # Entry point, binds :9999 (8000 taken)
 │       ├── api.rs             # OpenAI-style /v1/ endpoints
 │       ├── state.rs           # Jobs, checkpoints, models registry
 │       ├── conductor.rs       # Surgical edit hooks
 │       └── backend.rs         # Python worker IPC traits
 ├── tui/                       # Ratatui SSH client (scaffold)
 ├── gui/                       # Tauri + SvelteKit control panel (planned)
 ├── python/                    # Training worker runtime
 │   ├── worker.py              # Job executor, NUMA-aware
 │   ├── backends/
 │   │   └── transformers_backend.py  # Kepler LoRA training
 │   └── requirements.txt       # PyTorch + Transformers stack
 ├── deploy/                    # NOUGHT install scripts
 └── AGENTS.md                  # Our internal notes
 ```

 ## API Reference

 All endpoints on `http://NOUGHT:9999/v1/`

 | Method | Path | Purpose |
 |--------|------|---------|
 | POST | `/training/jobs` | Create training job |
 | GET | `/training/jobs` | List all jobs |
 | GET | `/training/jobs/:id` | Job status + checkpoints |
 | POST | `/training/jobs/:id/pause|resume` | Control |
 | POST | `/training/jobs/:id/checkpoint` | Worker reports checkpoint |
 | POST | `/training/jobs/:id/conductor` | Apply surgical edit |

 Tested and working:
 ```bash
 # Create job
 curl -X POST http://localhost:9999/v1/training/jobs \
   -H "Content-Type: application/json" \
   -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "target_steps": 100}'

 # List jobs
 curl http://localhost:9999/v1/training/jobs
 ```

 ## Design Decisions

 1. **No Docker** — Direct CUDA access, native debugging, shared filesystem
 2. **OpenAI-compatible API** — Standard surface, trivial client integration
 3. **Conductor hook** — Built-in place for median-arc evaluation and human reinforcement
 4. **Free-form job config** — Pass any hyperparams without changing Rust core
 5. **Checkpoint-first** — 2048-step cadence; surgical edits require dense history
 6. **Build from source when needed** — We've done it for llama.cpp; we'll do it for PyTorch

## Session Summary — 2026-09-19

### Completed
- ✅ Full project scaffold (Rust orchestrator, Python worker, TUI, deploy scripts)
- ✅ Orchestrator built and tested on NOUGHT (port 9999, all endpoints working)
- ✅ PyTorch 2.4.0-rc8 built from source with sm_37 support
- ✅ CUDA compute validated on Kepler (matmul, attention, backprop all working)
- ✅ transformers 4.40.0 + peft 0.7.0 installed
- ✅ LoRA training pipeline validated on Qwen2.5-0.5B-Instruct (forward+backward OK)
- ✅ All knowledge captured from llama_wukong (kepler-fixes.patch, fattn kernels, NUMA patterns)

### Next Session Priorities
1. Implement full training loop in transformers_backend.py
2. Deploy python worker code to NOUGHT
3. Test distributed training across multiple GPUs
4. Build TUI for SSH monitoring
5. Design GUI wireframes

### Critical Knowledge for Continuation
- PyTorch 2.4.0a0 installed editable from `/home/whistler/pytorch-kepler` in venv311
- venv311 path: `/home/whistler/venv311/bin/activate`
- Orchestrator runs on port 9999 (8000 in use by production)
- NUMA topology: NUMA0→GPU0-3, NUMA1→GPU4-7; use `numactl --cpunodebind=X --membind=X`
- GCC-11 is required host compiler (`/usr/bin/gcc-11`)
- CUDA 11.8 at `/usr/local/cuda-11.8`
- cuDNN not available (version=None) — cuBLAS legacy APIs only

 ---
 **Built by UnobligatedRascal. Ancient hardware, fresh ambition. We rebuild what the world abandoned.**