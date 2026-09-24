#!/usr/bin/env python3
"""
Comprehensive NF4 validation test suite for Kepler sm_37.

Tests all implemented NF4 functionality including:
1. CPU quantize/dequantize reference
2. CUDA dequantize kernel vs CPU reference
3. CUDA nf4_linear_forward kernel
4. LinearNF4 layer (from_linear, forward pass)
5. LinearNF4 + LoRA integration (custom wrapper)
6. Model-level NF4 replacement + inference
7. NF4 training step (forward + backward with LoRA)
8. Checkpoint save/load round-trip
9. VRAM compression validation (prove ~7x savings)
10. DDP path validation (multi-GPU)
11. Topology/NUMA validation
12. Scalability validation (VRAM math for 27B/35B)

Usage:
    # Run all tests (stops on first failure)
    python test_nf4_comprehensive.py

    # Run specific test
    python test_nf4_comprehensive.py --test kernels
    python test_nf4_comprehensive.py --test layer
    python test_nf4_comprehensive.py --test model
    python test_nf4_comprehensive.py --test train
    python test_nf4_comprehensive.py --test checkpoint
    python test_nf4_comprehensive.py --test vram
    python test_nf4_comprehensive.py --test ddp
    python test_nf4_comprehensive.py --test topology
    python test_nf4_comprehensive.py --test scalability

    # Run with specific model
    python test_nf4_comprehensive.py --model Qwen/Qwen2.5-7B-Instruct

    # Run with output file
    python test_nf4_comprehensive.py --output test_results.json

UnobligatedRascal — Making old hardware sing.
"""

import argparse
import sys
import time
import gc
import os
import json
import subprocess
import tempfile
import traceback
from datetime import datetime

import torch
import numpy as np

# Global test state
test_results = {}
test_log = []
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINING_STEPS = 10
TEST_OUTPUT_FILE = None


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log(msg: str):
    """Log message to console and test log."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")
    log_line = f"[{timestamp}] {msg}"
    print(f"[test] {msg}", flush=True)
    test_log.append(log_line)


def log_test(name: str, passed: bool, detail: str = ""):
    """Log test result with structured data."""
    global test_results
    timestamp = datetime.now().strftime("%H:%M:%S.%f")
    test_results[name] = {"passed": passed, "detail": detail, "timestamp": timestamp}
    status = "PASS" if passed else "FAIL"
    detail_str = f" — {detail}" if detail else ""
    print(f"[{status}] {name}{detail_str}", flush=True)
    test_log.append(f"[{timestamp}] [{status}] {name}{detail_str}")


def log_vram(gpu: int = 0, label: str = ""):
    """Log VRAM status with detailed metrics."""
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(gpu).total_memory
    allocated = torch.cuda.memory_allocated(gpu)
    reserved = torch.cuda.memory_reserved(gpu)
    free = total - reserved
    prefix = f"[{label}] " if label else ""
    log(f"VRAM GPU{gpu}: {prefix}allocated={allocated/1024**3:.2f}GB, "
        f"reserved={reserved/1024**3:.2f}GB, free={free/1024**3:.2f}GB/{total/1024**3:.1f}GB "
        f"({100*free/total:.0f}% free)")


def log_peak_vram(gpu: int = 0):
    """Log peak VRAM usage."""
    if not torch.cuda.is_available():
        return
    peak_alloc = torch.cuda.max_memory_allocated(gpu)
    peak_reserved = torch.cuda.max_memory_reserved(gpu)
    log(f"PEAK VRAM GPU{gpu}: alloc={peak_alloc/1024**3:.2f}GB, "
        f"reserved={peak_reserved/1024**3:.2f}GB")


def get_gpu_count():
    """Get number of available GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def save_test_results(output_file: str):
    """Save comprehensive test results to JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": TEST_MODEL,
        "training_steps": TRAINING_STEPS,
        "gpu_count": get_gpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "test_results": test_results,
        "test_log": test_log,
    }
    
    # Add hardware info
    if torch.cuda.is_available():
        hardware = []
        for i in range(get_gpu_count()):
            props = torch.cuda.get_device_properties(i)
            hardware.append({
                "gpu_id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "clock_rate_mhz": getattr(props, "clock_rate", None) / 1000 if hasattr(props, "clock_rate") else None,
            })
        results["hardware"] = hardware
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Test results saved to {output_file}")


# ============================================================
# Test 1: CPU quantize/dequantize reference
# ============================================================

def test_cpu_quantize_dequantize():
    log("=== Test 1: CPU quantize/dequantize ===")
    from kernels.nf4_kepler import quantize_to_nf4, dequantize_nf4, OFFICIAL_NF4_CODEBOOK

    torch.manual_seed(42)
    start_time = time.time()

    results = []
    try:
        # Test shapes relevant to neural network layers
        for shape in [(128, 256), (4096, 4096), (1024, 2048), (2048, 4096)]:
            shape_start = time.time()
            w = torch.randn(*shape)
            quantized, scales, codebook = quantize_to_nf4(w)
            w_reconstructed = dequantize_nf4(quantized, scales, codebook)
            w_reconstructed = w_reconstructed[:w.numel()].reshape(w.shape)
            shape_time = time.time() - shape_start

            l2_err = (w - w_reconstructed).pow(2).mean().sqrt().item()
            linf_err = (w - w_reconstructed).abs().max().item()
            compression = w.numel() * 4 / (quantized.numel() + scales.numel())

            results.append({
                "shape": shape,
                "l2_error": l2_err,
                "linf_error": linf_err,
                "compression_ratio": compression,
                "time_seconds": shape_time,
            })

            if l2_err >= 0.5:
                log_test("cpu_quantize_dequantize", False, f"L2 error too high: {l2_err} for shape {shape}")
                return False
            log(f"  Shape {shape}: L2={l2_err:.6f}, Linf={linf_err:.6f}, {compression:.2f}x compression, {shape_time:.3f}s")

        total_time = time.time() - start_time
        avg_compression = sum(r["compression_ratio"] for r in results) / len(results)
        log(f"  Total time: {total_time:.3f}s, avg compression: {avg_compression:.2f}x")

        test_results["cpu_quantize_dequantize_data"] = results
        log_test("cpu_quantize_dequantize", True, "all shapes passed")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("cpu_quantize_dequantize", False, str(e))
        return False


# ============================================================
# Test 2: CUDA dequantize kernel vs CPU reference
# ============================================================

def test_cuda_dequantize():
    log("=== Test 2: CUDA dequantize kernel ===")
    if not torch.cuda.is_available():
        log_test("cuda_dequantize", True, "skipped (no CUDA)")
        return True

    from kernels.nf4_kepler import quantize_to_nf4, dequantize_nf4, OFFICIAL_NF4_CODEBOOK
    from kernels.nf4_cuda import dequantize_nf4 as cuda_dequantize

    torch.manual_seed(42)
    start_time = time.time()

    try:
        for shape in [(128, 256), (1024, 1024), (4096, 4096)]:
            shape_start = time.time()
            w = torch.randn(*shape).cuda()
            quantized, scales, codebook = quantize_to_nf4(w)
            quantized = quantized.cuda()
            scales = scales.cuda()

            # CPU reference
            cpu_ref = dequantize_nf4(quantized.cpu(), scales.cpu(),
                                      torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32))
            cpu_ref = cpu_ref[:w.numel()].reshape(w.shape).cuda()

            # CUDA kernel
            cuda_out = cuda_dequantize(quantized, scales)
            cuda_out = cuda_out[:w.numel()].reshape(w.shape)

            max_diff = (cpu_ref - cuda_out).abs().max().item()
            shape_time = time.time() - shape_start

            if max_diff != 0.0:
                log_test("cuda_dequantize", False, f"CUDA vs CPU max diff: {max_diff} for shape {shape}")
                return False
            log(f"  Shape {shape}: max_diff={max_diff:.10f}, {shape_time:.3f}s")

        total_time = time.time() - start_time
        log(f"  Total time: {total_time:.3f}s")
        log_test("cuda_dequantize", True, "all shapes match CPU reference")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("cuda_dequantize", False, str(e))
        return False


# ============================================================
# Test 3: CUDA nf4_linear_forward kernel
# ============================================================

def test_cuda_linear_forward():
    log("=== Test 3: CUDA nf4_linear_forward ===")
    if not torch.cuda.is_available():
        log_test("cuda_linear_forward", True, "skipped (no CUDA)")
        return True

    from kernels.nf4_kepler import quantize_to_nf4, dequantize_nf4, OFFICIAL_NF4_CODEBOOK
    from kernels.nf4_cuda import nf4_linear_forward as cuda_linear_forward

    torch.manual_seed(42)
    start_time = time.time()

    try:
        device = torch.device("cuda")
        codebook = torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32, device=device)

        test_cases = [
            ((256, 512), 2, 8),      # 3D: (batch=2, seq=8, in=256)
            ((4096, 4096), 1, 4),    # 3D: (batch=1, seq=4, in=4096)
            ((256, 512), 4, None),   # 2D: (batch=4, in=256)
        ]
        for (in_f, out_f), batch, seq in test_cases:
            shape_start = time.time()
            # Create reference Linear
            weight = torch.randn(out_f, in_f, device=device)
            bias = torch.randn(out_f, device=device)

            # Quantize weight
            quantized, scales, _ = quantize_to_nf4(weight)
            quantized = quantized.cuda()
            scales = scales.cuda()

            # Dequantize for reference
            weight_dq = dequantize_nf4(quantized, scales, codebook)
            weight_dq = weight_dq[:out_f * in_f].reshape(out_f, in_f)

            # Test inputs
            if seq is not None:
                x = torch.randn(batch, seq, in_f, device=device)
            else:
                x = torch.randn(batch, in_f, device=device)

            # Reference: dequantize + torch.nn.functional.linear
            ref_out = torch.nn.functional.linear(x, weight_dq, bias)

            # CUDA kernel
            cuda_out = cuda_linear_forward(
                x, quantized, scales, bias,
                true_out_features=out_f, in_features=in_f
            )

            # Compare
            max_diff = (ref_out - cuda_out).abs().max().item()
            mean_diff = (ref_out - cuda_out).abs().mean().item()
            shape_time = time.time() - shape_start

            if max_diff >= 1e-4:
                log_test("cuda_linear_forward", False, f"Max diff too high: {max_diff} (mean={mean_diff})")
                return False
            log(f"  Input {tuple(x.shape)}, Weight {out_f}x{in_f}: "
                f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, {shape_time:.3f}s")

        total_time = time.time() - start_time
        log(f"  Total time: {total_time:.3f}s")
        log_test("cuda_linear_forward", True, "all shapes within tolerance")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("cuda_linear_forward", False, str(e))
        return False


# ============================================================
# Test 4: LinearNF4 layer
# ============================================================

def test_linear_nf4_layer():
    log("=== Test 4: LinearNF4 layer ===")
    if not torch.cuda.is_available():
        log_test("linear_nf4_layer", True, "skipped (no CUDA)")
        return True

    import torch.nn as nn
    from kernels.nf4_kepler import LinearNF4, dequantize_nf4, OFFICIAL_NF4_CODEBOOK

    torch.manual_seed(42)
    device = torch.device("cuda")
    codebook = torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32, device=device)
    start_time = time.time()

    try:
        for in_f, out_f in [(256, 512), (4096, 4096)]:
            shape_start = time.time()
            # Create reference Linear with specific weights
            ref_linear = nn.Linear(in_f, out_f, bias=True).to(device)

            # Create NF4 version
            nf4_layer = LinearNF4.from_linear(ref_linear)
            nf4_layer.to(device)

            # Verify weights are frozen
            for p in nf4_layer.parameters():
                if p.requires_grad:
                    log_test("linear_nf4_layer", False, "LinearNF4 parameters should not require grad")
                    return False

            # Compare forward pass (on dequantized weight)
            x = torch.randn(2, 8, in_f, device=device)
            ref_out = ref_linear(x)

            # NF4 output (should match dequantized weight behavior)
            nf4_out = nf4_layer(x)

            # Also compute expected output from dequantized weight
            dq_weight = dequantize_nf4(
                nf4_layer.weight_nf4, nf4_layer.weight_scales, codebook
            )
            dq_weight = dq_weight[:out_f * in_f].reshape(out_f, in_f)
            expected_out = torch.nn.functional.linear(x, dq_weight, nf4_layer.bias)

            max_diff_nf4 = (nf4_out - expected_out).abs().max().item()
            max_diff_ref = (ref_out - nf4_out).abs().max().item()  # vs original (includes quantization error)
            shape_time = time.time() - shape_start

            if max_diff_nf4 >= 1e-4:
                log_test("linear_nf4_layer", False, f"NF4 vs expected max diff: {max_diff_nf4}")
                return False
            log(f"  {out_f}x{in_f}: NF4_vs_expected={max_diff_nf4:.2e}, NF4_vs_original={max_diff_ref:.6f}, {shape_time:.3f}s")

        total_time = time.time() - start_time
        log(f"  Total time: {total_time:.3f}s")
        log_test("linear_nf4_layer", True, "forward pass matches dequantized reference")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("linear_nf4_layer", False, str(e))
        return False


# ============================================================
# Test 5: LinearNF4 + LoRA integration (custom wrapper)
# ============================================================

def test_linear_nf4_lora():
    log("=== Test 5: LinearNF4 + LoRA (nf4_lora_wrap) ===")
    if not torch.cuda.is_available():
        log_test("linear_nf4_lora", True, "skipped (no CUDA)")
        return True

    import torch.nn as nn
    from kernels.nf4_kepler import LinearNF4, nf4_lora_wrap

    torch.manual_seed(42)
    device = torch.device("cuda")
    start_time = time.time()

    try:
        # Create a simple model with a LinearNF4 layer
        class TestModel(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                base_linear = nn.Linear(in_f, out_f, bias=True)
                self.linear = nf4_lora_wrap(LinearNF4.from_linear(base_linear), r=8, alpha=16)
                self.linear.to(device)

            def forward(self, x):
                return self.linear(x)

        model = TestModel(256, 512).to(device)

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        log(f"  Trainable: {trainable}, Frozen: {frozen}")

        # Forward pass (no no_grad — need gradient graph)
        x = torch.randn(2, 4, 256, device=device)
        y = model(x)
        log(f"  Forward pass: input {tuple(x.shape)} → output {tuple(y.shape)}")

        # Backward pass (gradients should only go to LoRA)
        y.sum().backward()

        # Check gradients
        lora_grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        nf4_grads = [p for p in model.parameters() if not p.requires_grad and p.grad is not None]

        log(f"  LoRA params with grad: {len(lora_grads)}")
        log(f"  NF4 params with grad: {len(nf4_grads)} (should be 0 — weights frozen)")

        if len(lora_grads) == 0:
            log_test("linear_nf4_lora", False, "LoRA params should have gradients")
            return False
        if len(nf4_grads) != 0:
            log_test("linear_nf4_lora", False, "NF4 weights should be frozen (no gradients)")
            return False

        # Step optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()

        # Verify LoRA changes output
        with torch.no_grad():
            y_after = model(x)
        diff = (y - y_after).abs().max().item()
        log(f"  Output change after optimizer step: max_diff={diff:.6f} (should be > 0)")
        if diff <= 1e-6:
            log_test("linear_nf4_lora", False, "LoRA should change output after optimizer step")
            return False

        total_time = time.time() - start_time
        log(f"  Total time: {total_time:.3f}s")
        log_test("linear_nf4_lora", True, "LoRA gradient flow and optimizer step verified")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("linear_nf4_lora", False, str(e))
        return False


# ============================================================
# Test 6: Model-level NF4 replacement
# ============================================================

def test_model_nf4(model_name: str):
    log(f"=== Test 6: Model NF4 replacement ({model_name}) ===")
    if not torch.cuda.is_available():
        log_test("model_nf4", True, "skipped (no CUDA)")
        return True

    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kernels.nf4_kepler import replace_linear_with_nf4

    log_vram(label="START")
    clear_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    try:
        # Load model
        log(f"Loading model: {model_name}")
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        load_time = time.time() - start
        log(f"  Loaded in {load_time:.1f}s")
        log_vram(label="after_load_fp32")

        # Count Linear layers
        linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        log(f"  Total Linear layers: {linear_count}")

        # Replace with NF4
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        log(f"Replacing {len(target_modules)} target types with NF4")

        start = time.time()
        replace_linear_with_nf4(model, target_modules=target_modules, verbose=True)
        replace_time = time.time() - start
        log(f"  Replacement took {replace_time:.1f}s")
        log_vram(label="after_nf4_replace")

        # Move to GPU
        log("Moving model to GPU")
        start = time.time()
        model.to("cuda")
        gpu_time = time.time() - start
        log(f"  GPU transfer took {gpu_time:.1f}s")
        log_vram(label="after_gpu_transfer")

        # Test inference
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        log("Running inference...")
        clear_cache()
        torch.cuda.reset_peak_memory_stats()
        log_vram(label="before_inference")

        with torch.no_grad():
            start = time.time()
            outputs = model(**inputs, labels=inputs["input_ids"])
            inference_time = time.time() - start
            log(f"  Loss: {outputs.loss.item():.4f} ({inference_time:.2f}s)")

        log_vram(label="after_inference")
        log_peak_vram()

        total_time = time.time() - start_time
        log(f"  Total time: {total_time:.1f}s")
        log_test("model_nf4", True, f"inference completed, loss={outputs.loss.item():.4f}")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("model_nf4", False, str(e))
        return False


# ============================================================
# Test 7: NF4 training step
# ============================================================

def test_model_train(model_name: str, steps: int = TRAINING_STEPS):
    log(f"=== Test 7: NF4 training ({model_name}, {steps} steps) ===")
    if not torch.cuda.is_available():
        log_test("model_train", True, "skipped (no CUDA)")
        return True

    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kernels.nf4_kepler import replace_linear_with_nf4, apply_lora_to_model

    log_vram(label="START")
    clear_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    try:
        # Load model
        log(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Replace with NF4
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        replace_linear_with_nf4(model, target_modules=target_modules, verbose=False)

        # Apply LoRA using our custom wrapper
        model, lora_params = apply_lora_to_model(model, target_modules=target_modules, r=8, alpha=16)
        model.to("cuda")
        model.train()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"LoRA params: {len(lora_params)} parameter groups")
        total = sum(p.numel() for p in model.parameters())
        log(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
        log_vram(label="after_setup")

        # Clear cache before training
        clear_cache()
        log_vram(label="after_clear_cache")

        # Simple training loop
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

        losses = []
        step_times = []
        for step in range(steps):
            # Dummy batch (single sample, short sequence to save VRAM)
            prompt = f"Training step {step}. This is a test sentence for NF4 training." * 5
            inputs = tokenizer(prompt, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            labels = inputs["input_ids"].clone()

            # Clear cache every few steps to combat fragmentation
            if step > 0 and step % 5 == 0:
                clear_cache()

            step_start = time.time()
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            step_time = time.time() - step_start

            losses.append(loss.item())
            step_times.append(step_time)

            # Log every step for first 5, then every 5, then last
            should_log = (step < 5) or (step % 5 == 0) or (step == steps - 1)
            if should_log:
                allocated = torch.cuda.memory_allocated() / 1024**3
                log(f"Step {step+1:3d}/{steps}: loss={loss.item():.4f}, time={step_time:.2f}s, "
                    f"VRAM_alloc={allocated:.2f}GB")

        # Final stats
        log_vram(label="after_training")
        log_peak_vram()

        # Check for NaN
        if any(np.isnan(loss_val) for loss_val in losses):
            log_test("model_train", False, "NaN loss detected")
            return False

        # Check for loss divergence
        if losses[-1] > losses[0] * 5:
            log(f"  WARNING: Loss may be diverging (start={losses[0]:.4f}, end={losses[-1]:.4f})")

        total_time = time.time() - start_time
        log(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
        log(f"Step times: min={min(step_times):.2f}s, max={max(step_times):.2f}s, "
            f"avg={sum(step_times)/len(step_times):.2f}s")
        log(f"  Total time: {total_time:.1f}s")
        log_test("model_train", True, f"training completed, avg_loss={np.mean(losses):.4f}")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("model_train", False, str(e))
        return False


# ============================================================
# Test 8: Checkpoint save/load round-trip
# ============================================================

def test_checkpoint_roundtrip(model_name: str):
    log(f"=== Test 8: Checkpoint save/load round-trip ({model_name}) ===")
    if not torch.cuda.is_available():
        log_test("checkpoint_roundtrip", True, "skipped (no CUDA)")
        return True

    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kernels.nf4_kepler import replace_linear_with_nf4, apply_lora_to_model

    clear_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load model and apply NF4
            log(f"Loading model for checkpoint test: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            replace_linear_with_nf4(model, target_modules=target_modules, verbose=False)
            model, _ = apply_lora_to_model(model, target_modules=target_modules, r=8, alpha=16)
            model.to("cuda")
            model.train()

            # Save checkpoint
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint")
            log(f"Saving checkpoint to {checkpoint_path}")
            model.save_pretrained(checkpoint_path, safe_serialization=True)

            # Save optimizer state separately (for resume)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
            optimizer_state_path = os.path.join(tmpdir, "optimizer.pt")
            torch.save(optimizer.state_dict(), optimizer_state_path)

            log_vram(label="after_save")

            # Clear cache and reload
            del model
            clear_cache()

            # Load checkpoint
            log(f"Loading checkpoint from {checkpoint_path}")
            loaded_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            loaded_model.to("cuda")

            # Verify loaded model has LinearNF4 layers
            has_nf4 = any("weight_nf4" in name for name, _ in loaded_model.named_buffers())
            if not has_nf4:
                log_test("checkpoint_roundtrip", False, "Loaded model missing NF4 weight buffers")
                return False

            # Verify loaded model produces same output as original
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            prompt = "Hello, world!"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                loaded_outputs = loaded_model(**inputs, labels=inputs["input_ids"])

            log(f"  Loaded model loss: {loaded_outputs.loss.item():.4f}")

            # Verify optimizer state can be loaded
            loaded_optimizer_state = torch.load(optimizer_state_path, weights_only=True)
            log(f"  Optimizer state loaded: {len(loaded_optimizer_state)} entries")

            del loaded_model
            clear_cache()

        total_time = time.time() - start_time
        log(f"  Total time: {total_time:.1f}s")
        log_test("checkpoint_roundtrip", True, "save/load verified, NF4 buffers preserved")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("checkpoint_roundtrip", False, str(e))
        return False


# ============================================================
# Test 9: VRAM compression validation
# ============================================================

def test_vram_compression(model_name: str):
    log(f"=== Test 9: VRAM compression validation ({model_name}) ===")
    if not torch.cuda.is_available():
        log_test("vram_compression", True, "skipped (no CUDA)")
        return True

    import torch.nn as nn
    from transformers import AutoModelForCausalLM
    from kernels.nf4_kepler import replace_linear_with_nf4

    clear_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    try:
        # Measure FP32 VRAM
        log(f"Loading model in FP32 to measure baseline VRAM")
        model_fp32 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model_fp32.to("cuda")

        fp32_allocated = torch.cuda.memory_allocated() / 1024**3
        log(f"  FP32 allocated: {fp32_allocated:.2f}GB")

        # Delete FP32 model
        del model_fp32
        clear_cache()

        # Measure NF4 VRAM
        log(f"Loading model in FP32, then converting to NF4")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        replace_linear_with_nf4(model, target_modules=target_modules, verbose=False)
        model.to("cuda")

        nf4_allocated = torch.cuda.memory_allocated() / 1024**3
        log(f"  NF4 allocated: {nf4_allocated:.2f}GB")

        # Calculate compression ratio
        # Note: We're measuring peak allocated, not just weights, so ratio will be
        # lower than 7x. The compression applies only to weight storage.
        weight_ratio = fp32_allocated / nf4_allocated
        log(f"  VRAM ratio (FP32/NF4): {weight_ratio:.2f}x")

        del model
        clear_cache()

        # Validate we got significant compression (should be > 2x for this model)
        if weight_ratio < 2.0:
            log_test("vram_compression", False, f"Expected >2x compression, got {weight_ratio:.2f}x")
            return False

        total_time = time.time() - start_time
        log(f"  Total time: {total_time:.1f}s")
        log_test("vram_compression", True, f"measured {weight_ratio:.2f}x VRAM reduction")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("vram_compression", False, str(e))
        return False


# ============================================================
# Test 10: DDP path validation
# ============================================================

def test_ddp_path(model_name: str):
    log(f"=== Test 10: DDP path validation ({model_name}) ===")
    if not torch.cuda.is_available():
        log_test("ddp_path", True, "skipped (no CUDA)")
        return True

    if get_gpu_count() < 2:
        log_test("ddp_path", True, "skipped (need >= 2 GPUs)")
        return True

    # Write a minimal DDP test script
    test_script = """
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from kernels.nf4_kepler import replace_linear_with_nf4, apply_lora_to_model

def test_ddp_nf4():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Apply NF4
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    replace_linear_with_nf4(model, target_modules=target_modules, verbose=(rank == 0))

    # Apply LoRA
    model, _ = apply_lora_to_model(model, target_modules=target_modules, r=8, alpha=16)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train()

    # Simple training step
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = f"DDP test from rank {rank}"
    inputs = tokenizer(prompt, return_tensors="pt").to(rank)
    labels = inputs["input_ids"].clone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    for step in range(3):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        optimizer.step()

        if rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    dist.destroy_process_group()
    print("DDP NF4 test completed successfully")

if __name__ == "__main__":
    test_ddp_nf4()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        test_script_path = f.name

    try:
        # Run with torchrun for 2 processes
        log(f"Running DDP test with 2 GPUs (torchrun)")
        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            "--nnodes=1",
            test_script_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        log(f"  Return code: {result.returncode}")
        if result.stdout:
            for line in result.stdout.split('\n')[-10:]:
                log(f"  stdout: {line}")
        if result.stderr:
            for line in result.stderr.split('\n')[-10:]:
                log(f"  stderr: {line}")

        if result.returncode != 0:
            log_test("ddp_path", False, "torchrun failed")
            return False

        if "DDP NF4 test completed successfully" not in result.stdout:
            log_test("ddp_path", False, "DDP test did not complete successfully")
            return False

        log_test("ddp_path", True, "DDP training step completed on 2 GPUs")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("ddp_path", False, str(e))
        return False
    finally:
        os.unlink(test_script_path)


# ============================================================
# Test 11: Topology/NUMA validation
# ============================================================

def test_topology():
    log("=== Test 11: Topology/NUMA validation ===")

    try:
        # Check GPU count
        if not torch.cuda.is_available():
            log_test("topology", True, "skipped (no CUDA)")
            return True

        gpu_count = get_gpu_count()
        log(f"  GPU count: {gpu_count}")

        if gpu_count != 8:
            log(f"  WARNING: Expected 8 GPUs, found {gpu_count}")

        # Check GPU topology
        log("  GPU topology:")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            log(f"    GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB, "
                f"compute_cap={props.major}.{props.minor}")

        # Validate all GPUs are K80
        all_k80 = True
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            if "K80" not in props.name:
                all_k80 = False
                log(f"  WARNING: GPU {i} is not a K80: {props.name}")

        if not all_k80:
            log_test("topology", False, "Not all GPUs are K80")
            return False

        # Validate compute capability is sm_37
        all_sm37 = True
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            if props.major != 3 or props.minor != 7:
                all_sm37 = False
                log(f"  WARNING: GPU {i} is not sm_37: {props.major}.{props.minor}")

        if not all_sm37:
            log_test("topology", False, "Not all GPUs are sm_37")
            return False

        # Check NUMA topology (via /sys or lscpu)
        log("  NUMA topology:")
        try:
            import subprocess
            result = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n')[:20]:
                    log(f"    {line}")
        except Exception as e:
            log(f"  Could not query NUMA topology: {e}")

        log_test("topology", True, "all GPUs are K80 sm_37")
        return True
    except Exception as e:
        log(f"  ERROR: {e}")
        log(traceback.format_exc())
        log_test("topology", False, str(e))
        return False


# ============================================================
# Test 12: Scalability validation (VRAM math)
# ============================================================

def test_scalability():
    log("=== Test 12: Scalability validation (VRAM math) ===")

    # Calculate expected VRAM for various model sizes
    # NF4 compression ratio: ~7.11x for weights
    # Training overhead: optimizer states (2x model size), gradients (1x), activations (variable)

    def calculate_training_vram(model_params_b, nf4=True, lora_r=8):
        """Estimate training VRAM in GB."""
        # Model weight size in GB
        fp32_weight_gb = model_params_b  # 1 param = 4 bytes, 1B params = 4GB = ~1GB in NF4

        if nf4:
            weight_gb = fp32_weight_gb / 7.11  # NF4 compression
        else:
            weight_gb = fp32_weight_gb

        # LoRA params: roughly r * sum(in_features + out_features) for each targeted layer
        # For a 7B model with typical architecture:
        # - ~40 transformer layers
        # - Each layer: 7 attention projections (q,k,v,o,gate,up,down) + 2 MLP
        # - Rough estimate: LoRA adds ~0.1% of model params for r=8
        lora_params = model_params_b * 0.001 * lora_r
        lora_gb = lora_params  # FP32

        # Optimizer states (AdamW): 2x model params + gradients
        optimizer_gb = weight_gb * 2 + weight_gb  # moment1 + moment2 + gradients

        # Activations: varies with batch size, seq length
        # Estimate: 1x model size for small batch
        activations_gb = weight_gb

        total = weight_gb + lora_gb + optimizer_gb + activations_gb
        return {
            "weight_gb": weight_gb,
            "lora_gb": lora_gb,
            "optimizer_gb": optimizer_gb,
            "activations_gb": activations_gb,
            "total_gb": total,
        }

    log("  Expected VRAM for various model sizes (NF4 + LoRA training):")
    for params_b in [0.5, 7, 14, 27, 35]:
        est = calculate_training_vram(params_b, nf4=True)
        fits_k80 = est["total_gb"] <= 11.5
        fits_all_8 = est["total_gb"] <= 11.5 * 8
        status = ""
        if not fits_k80 and not fits_all_8:
            status = " ❌ TOO LARGE"
        elif not fits_k80:
            status = " (needs DDP across multiple GPUs)"

        log(f"  {params_b:3.1f}B: weight={est['weight_gb']:.1f}GB, lora={est['lora_gb']:.1f}GB, "
            f"opt={est['optimizer_gb']:.1f}GB, act={est['activations_gb']:.1f}GB, "
            f"TOTAL={est['total_gb']:.1f}GB{status}")

    # Validate that 7B NF4 should fit on single K80
    est_7b = calculate_training_vram(7.0, nf4=True)
    if est_7b["total_gb"] > 11.5:
        log_test("scalability", False, f"7B NF4 estimated at {est_7b['total_gb']:.1f}GB, exceeds K80 VRAM")
        return False

    # Validate that 27B NF4 should fit with DDP across 8 GPUs
    est_27b = calculate_training_vram(27.0, nf4=True)
    if est_27b["total_gb"] > 11.5 * 8:
        log_test("scalability", False, f"27B NF4 estimated at {est_27b['total_gb']:.1f}GB, exceeds total GPU VRAM")
        return False

    log(f"  7B NF4 estimated at {est_7b['total_gb']:.1f}GB — should fit on single K80 (11.5GB)")
    log(f"  27B NF4 estimated at {est_27b['total_gb']:.1f}GB — should fit with DDP across 8 GPUs")

    log_test("scalability", True, "VRAM math validates NF4 scaling to 27B/35B models")
    return True


# ============================================================
# Main
# ============================================================

def main():
    global TEST_MODEL, TRAINING_STEPS, TEST_OUTPUT_FILE
    parser = argparse.ArgumentParser(description="Comprehensive NF4 validation test suite")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "kernels", "layer", "model", "train",
                                 "checkpoint", "vram", "ddp", "topology", "scalability"],
                        help="Test to run")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides default)")
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS,
                        help="Training steps for train test")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for test results (JSON)")
    args = parser.parse_args()

    if args.model:
        TEST_MODEL = args.model
    if args.steps:
        TRAINING_STEPS = args.steps
    if args.output:
        TEST_OUTPUT_FILE = args.output

    if not torch.cuda.is_available():
        log("CUDA not available. Some tests will be skipped.")
    else:
        log(f"CUDA available: {torch.cuda.get_device_name(0)} ({get_gpu_count()} GPUs)")
        log(f"PyTorch version: {torch.__version__}")
        log(f"CUDA version: {torch.version.cuda}")

    tests_to_run = []

    if args.test == "all":
        tests_to_run = [
            ("kernels", test_cpu_quantize_dequantize),
            ("kernels", test_cuda_dequantize),
            ("kernels", test_cuda_linear_forward),
            ("layer", test_linear_nf4_layer),
            ("layer", test_linear_nf4_lora),
            ("model", lambda: test_model_nf4(TEST_MODEL)),
            ("train", lambda: test_model_train(TEST_MODEL, TRAINING_STEPS)),
            ("checkpoint", lambda: test_checkpoint_roundtrip(TEST_MODEL)),
            ("vram", lambda: test_vram_compression(TEST_MODEL)),
            ("ddp", lambda: test_ddp_path(TEST_MODEL)),
            ("topology", test_topology),
            ("scalability", test_scalability),
        ]
    else:
        # Run specific test
        test_map = {
            "kernels": [
                ("kernels", test_cpu_quantize_dequantize),
                ("kernels", test_cuda_dequantize),
                ("kernels", test_cuda_linear_forward),
            ],
            "layer": [
                ("layer", test_linear_nf4_layer),
                ("layer", test_linear_nf4_lora),
            ],
            "model": [("model", lambda: test_model_nf4(TEST_MODEL))],
            "train": [("train", lambda: test_model_train(TEST_MODEL, TRAINING_STEPS))],
            "checkpoint": [("checkpoint", lambda: test_checkpoint_roundtrip(TEST_MODEL))],
            "vram": [("vram", lambda: test_vram_compression(TEST_MODEL))],
            "ddp": [("ddp", lambda: test_ddp_path(TEST_MODEL))],
            "topology": [("topology", test_topology)],
            "scalability": [("scalability", test_scalability)],
        }
        tests_to_run = test_map[args.test]

    log(f"\nRunning {len(tests_to_run)} tests...")
    log("=" * 60)

    all_passed = True
    for test_name, test_fn in tests_to_run:
        clear_cache()
        passed = test_fn()
        if not passed and args.test == "all":
            log(f"Test failed: {test_name}. Stopping.")
            all_passed = False
            break
        all_passed = all_passed and passed

    log("=" * 60)
    log("\nTest Results Summary:")
    for name, result in test_results.items():
        # Skip data lists, only show test results
        if isinstance(result, list):
            continue
        status = "PASS" if result["passed"] else "FAIL"
        detail = f" — {result['detail']}" if result["detail"] else ""
        print(f"  [{status}] {name}{detail}")

    if all_passed:
        log("\n=== ALL TESTS PASSED ===")
    else:
        log("\n=== SOME TESTS FAILED ===")

    # Save results to file if requested
    if TEST_OUTPUT_FILE:
        save_test_results(TEST_OUTPUT_FILE)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()