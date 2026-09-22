#!/usr/bin/env python3
"""
Comprehensive NF4 validation test suite for Kepler sm_37.

Tests:
1. CPU quantize/dequantize reference
2. CUDA dequantize kernel vs CPU reference
3. CUDA nf4_linear_forward kernel
4. LinearNF4 layer (from_linear, forward pass)
5. LinearNF4 + LoRA integration (PEFT)
6. Model-level NF4 replacement (TinyLlama or Qwen2.5-0.5B)
7. NF4 training step (forward + backward with LoRA)

Usage:
    # Test 1-3 (kernels only):
    python test_nf4.py --test kernels

    # Test 4-5 (layer + LoRA):
    python test_nf4.py --test layer

    # Test 6 (small model replacement):
    python test_nf4.py --test model --model Qwen/Qwen2.5-0.5B-Instruct

    # Test 7 (training step on small model):
    python test_nf4.py --test train --model Qwen/Qwen2.5-0.5B-Instruct --steps 10

    # Full suite (kernels + layer + small model):
    python test_nf4.py --test full --model Qwen/Qwen2.5-0.5B-Instruct

    # 7B test (requires full GPU):
    python test_nf4.py --test model --model Qwen/Qwen2.5-7B-Instruct

UnobligatedRascal — Making old hardware sing.
"""

import argparse
import sys
import time
import gc

import torch
import numpy as np


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log(msg: str):
    print(f"[test] {msg}", flush=True)


def log_vram(gpu: int = 0):
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(gpu).total_memory
    allocated = torch.cuda.memory_allocated(gpu)
    free = total - torch.cuda.memory_reserved(gpu)
    log(f"VRAM GPU{gpu}: {allocated/1024**3:.2f}GB / {total/1024**3:.1f}GB "
        f"({free/1024**3:.2f}GB free)")


# ============================================================
# Test 1: CPU quantize/dequantize reference
# ============================================================

def test_cpu_quantize_dequantize():
    log("=== Test 1: CPU quantize/dequantize ===")
    from kernels.nf4_kepler import quantize_to_nf4, dequantize_nf4, OFFICIAL_NF4_CODEBOOK

    torch.manual_seed(42)

    # Test shapes relevant to neural network layers (typically multiples of 64 or large)
    for shape in [(128, 256), (4096, 4096), (1024, 2048), (2048, 4096)]:
        w = torch.randn(*shape)
        quantized, scales, codebook = quantize_to_nf4(w)
        w_reconstructed = dequantize_nf4(quantized, scales, codebook)
        w_reconstructed = w_reconstructed[:w.numel()].reshape(w.shape)

        l2_err = (w - w_reconstructed).pow(2).mean().sqrt().item()
        linf_err = (w - w_reconstructed).abs().max().item()
        compression = w.numel() * 4 / (quantized.numel() + scales.numel())

        assert l2_err < 0.5, f"L2 error too high: {l2_err}"
        log(f"  Shape {shape}: L2={l2_err:.6f}, Linf={linf_err:.6f}, {compression:.2f}x compression")

    log("  PASSED")
    return True


# ============================================================
# Test 2: CUDA dequantize kernel vs CPU reference
# ============================================================

def test_cuda_dequantize():
    log("=== Test 2: CUDA dequantize kernel ===")
    if not torch.cuda.is_available():
        log("  SKIPPED (no CUDA)")
        return True

    from kernels.nf4_kepler import quantize_to_nf4, dequantize_nf4, OFFICIAL_NF4_CODEBOOK
    from kernels.nf4_cuda import dequantize_nf4 as cuda_dequantize

    torch.manual_seed(42)

    for shape in [(128, 256), (1024, 1024), (4096, 4096)]:
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
        assert max_diff == 0.0, f"CUDA vs CPU max diff: {max_diff}"
        log(f"  Shape {shape}: max_diff={max_diff:.10f}")

    log("  PASSED")
    return True


# ============================================================
# Test 3: CUDA nf4_linear_forward kernel
# ============================================================

def test_cuda_linear_forward():
    log("=== Test 3: CUDA nf4_linear_forward ===")
    if not torch.cuda.is_available():
        log("  SKIPPED (no CUDA)")
        return True

    from kernels.nf4_kepler import quantize_to_nf4, dequantize_nf4, OFFICIAL_NF4_CODEBOOK
    from kernels.nf4_cuda import nf4_linear_forward as cuda_linear_forward

    torch.manual_seed(42)

    device = torch.device("cuda")
    codebook = torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32, device=device)

    test_cases = [
        ((256, 512), 2, 8),      # 3D: (batch=2, seq=8, in=256)
        ((4096, 4096), 1, 4),    # 3D: (batch=1, seq=4, in=4096)
        ((256, 512), 4, None),   # 2D: (batch=4, in=256)
    ]
    for (in_f, out_f), batch, seq in test_cases:
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

        # Allow small numerical difference (quantization error in dequantize path is identical,
        # but cuBLAS vs direct mm might differ slightly)
        assert max_diff < 1e-4, f"Max diff too high: {max_diff} (mean={mean_diff})"
        log(f"  Input {tuple(x.shape)}, Weight {out_f}x{in_f}: "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    log("  PASSED")
    return True


# ============================================================
# Test 4: LinearNF4 layer
# ============================================================

def test_linear_nf4_layer():
    log("=== Test 4: LinearNF4 layer ===")
    if not torch.cuda.is_available():
        log("  SKIPPED (no CUDA)")
        return True

    import torch.nn as nn
    from kernels.nf4_kepler import LinearNF4, dequantize_nf4, OFFICIAL_NF4_CODEBOOK

    torch.manual_seed(42)
    device = torch.device("cuda")
    codebook = torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32, device=device)

    for in_f, out_f in [(256, 512), (4096, 4096)]:
        # Create reference Linear with specific weights
        ref_linear = nn.Linear(in_f, out_f, bias=True).to(device)
        with torch.no_grad():
            ref_weight = ref_linear.weight.clone()

        # Create NF4 version
        nf4_layer = LinearNF4.from_linear(ref_linear)
        nf4_layer.to(device)

        # Verify weights are frozen
        for p in nf4_layer.parameters():
            assert not p.requires_grad, "LinearNF4 parameters should not require grad"

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

        assert max_diff_nf4 < 1e-4, f"NF4 vs expected max diff: {max_diff_nf4}"
        log(f"  {out_f}x{in_f}: NF4_vs_expected={max_diff_nf4:.2e}, NF4_vs_original={max_diff_ref:.6f}")

    log("  PASSED")
    return True


# ============================================================
# Test 5: LinearNF4 + LoRA integration (custom wrapper)
# ============================================================

def test_linear_nf4_lora():
    log("=== Test 5: LinearNF4 + LoRA (nf4_lora_wrap) ===")
    if not torch.cuda.is_available():
        log("  SKIPPED (no CUDA)")
        return True

    import torch.nn as nn
    from kernels.nf4_kepler import LinearNF4, nf4_lora_wrap

    torch.manual_seed(42)
    device = torch.device("cuda")

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

    assert len(lora_grads) > 0, "LoRA params should have gradients"
    assert len(nf4_grads) == 0, "NF4 weights should be frozen (no gradients)"

    # Step optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()

    # Verify LoRA changes output
    with torch.no_grad():
        y_after = model(x)
    diff = (y - y_after).abs().max().item()
    log(f"  Output change after optimizer step: max_diff={diff:.6f} (should be > 0)")
    assert diff > 1e-6, "LoRA should change output after optimizer step"

    log("  PASSED")
    return True


# ============================================================
# Test 6: Model-level NF4 replacement
# ============================================================

def test_model_nf4(model_name: str):
    log(f"=== Test 6: Model NF4 replacement ({model_name}) ===")
    if not torch.cuda.is_available():
        log("  SKIPPED (no CUDA)")
        return True

    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kernels.nf4_kepler import replace_linear_with_nf4, LinearNF4, get_vram_info

    log_vram()
    clear_cache()

    # Load model
    log(f"Loading model: {model_name}")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    log(f"  Loaded in {time.time() - start:.1f}s")
    log_vram()

    # Count Linear layers
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    log(f"  Total Linear layers: {linear_count}")

    # Replace with NF4
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    log(f"Replacing {len(target_modules)} target types with NF4")

    start = time.time()
    replace_linear_with_nf4(model, target_modules=target_modules, verbose=True)
    log(f"  Replacement took {time.time() - start:.1f}s")
    log_vram()

    # Move to GPU
    log("Moving model to GPU")
    start = time.time()
    model.to("cuda")
    log(f"  GPU transfer took {time.time() - start:.1f}s")
    log_vram()

    # Test inference
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    log("Running inference...")
    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs, labels=inputs["input_ids"])
        log(f"  Loss: {outputs.loss.item():.4f} ({time.time() - start:.2f}s)")

    log_vram()
    log("  PASSED")
    return True


# ============================================================
# Test 7: NF4 training step
# ============================================================

def test_model_train(model_name: str, steps: int = 10):
    log(f"=== Test 7: NF4 training ({model_name}, {steps} steps) ===")
    if not torch.cuda.is_available():
        log("  SKIPPED (no CUDA)")
        return True

    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kernels.nf4_kepler import replace_linear_with_nf4, apply_lora_to_model, log_vram

    clear_cache()

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
    log_vram()

    # Clear cache before training
    clear_cache()

    # Simple training loop
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    losses = []
    for step in range(steps):
        # Dummy batch (single sample, short sequence to save VRAM)
        prompt = f"Training step {step}. This is a test sentence for NF4 training." * 5
        inputs = tokenizer(prompt, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        labels = inputs["input_ids"].clone()

        # Clear cache every few steps
        if step > 0 and step % 5 == 0:
            clear_cache()

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 2 == 0 or step == steps - 1:
            log(f"Step {step+1}/{steps}: loss={loss.item():.4f}")

    if steps % 10 == 0:
        log_vram()

    # Check for NaN
    if any(np.isnan(loss_val) for loss_val in losses):
        log("  FAILED: NaN loss detected")
        return False

    log(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
    log("  PASSED")
    return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NF4 validation test suite")
    parser.add_argument("--test", type=str, default="kernels",
                        choices=["kernels", "layer", "model", "train", "full"],
                        help="Test to run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model name for model/train tests")
    parser.add_argument("--steps", type=int, default=10,
                        help="Training steps for train test")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        log("CUDA not available. Some tests will be skipped.")
    else:
        log(f"CUDA available: {torch.cuda.get_device_name(0)}")

    all_passed = True

    if args.test in ("kernels", "full"):
        all_passed &= test_cpu_quantize_dequantize()
        clear_cache()
        all_passed &= test_cuda_dequantize()
        clear_cache()
        all_passed &= test_cuda_linear_forward()
        clear_cache()

    if args.test in ("layer", "full"):
        all_passed &= test_linear_nf4_layer()
        clear_cache()
        all_passed &= test_linear_nf4_lora()
        clear_cache()

    if args.test == "model":
        all_passed &= test_model_nf4(args.model)

    if args.test == "train":
        all_passed &= test_model_train(args.model, args.steps)

    clear_cache()

    if all_passed:
        log("\n=== ALL TESTS PASSED ===")
        sys.exit(0)
    else:
        log("\n=== SOME TESTS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
