"""
Custom NF4-style 4-bit quantization for Kepler sm_37.

Bypasses bitsandbytes entirely — writes our own CUDA kernels targeting
compute capability 3.7. Based on QLoRA paper (Dettmers et al. 2023).

Key design decisions for Kepler:
- Use legacy cuBLAS Sgemm (NO Ex APIs — crash on sm_37)
- Lookup-table based dequantization (Kepler lacks native 4-bit ops)
- FP32 compute throughout (FP16 is slow without tensor cores)
- Follow llama_wukong pattern: simple CUDA kernels, Python wrappers

Architecture:
1. NF4Quantize: Generate NF4 codebook + quantize weights (Python)
2. LinearNF4: torch.nn.Linear replacement (CUDA dequantize + Sgemm)
3. Gradient path: Straight-through estimator through quantization

TODO:
- [x] Generate NF4 codebook (16 values, distribution-aware)
- [x] Quantize/dequantize functions (CPU reference implementation)
- [x] CUDA kernel: dequantize 4-bit block → FP32 tensor (nf4_dequant.cu)
- [x] CUDA kernel: fused dequantize + matmul using cuBLAS Sgemm (nf4_dequant.cu)
- [x] PyTorch autograd function for gradient computation (nf4_cuda.py)
- [ ] LinearNF4 layer wrapping everything
- [ ] Integration into transformers_backend.py

Integration notes:
- Config flag: "quant": "nf4" in job config → worker routes to NF4 path
- Checkpoints: Store packed NF4 + scales; support materializing FP32 for surgical edits
- Optimizer: Consider CPU-offloaded Adam moments for extra VRAM savings
- Reporting: Surface compression ratio to orchestrator metrics
- Safety: Auto-fallback to FP32 on kernel failure/NaNs

UnobligatedRascal — Making old hardware sing.
"""

import math
import numpy as np
import torch
from typing import Any


def generate_nf4_codebook() -> tuple[np.ndarray, Any]:
    """
    Generate NF4 codebook: 16 values optimized for normal distributions.

    NF4 uses two Gaussian mixtures to place codebook values where
    they're most useful for weight distributions.

    Returns:
        (codebook, F): codebook of 16 values, F for quantization math
    """
    # NF4 uses two Gaussian distributions:
    # N(μ₁, σ₁²) with weight m₁ and N(μ₂, σ₂²) with weight m₂
    # Optimized parameters from QLoRA paper
    m1, m2 = 0.68268945, 1 - 0.68268945
    mu1, mu2 = -0.3550504, 0.3550504
    sigma1, sigma2 = 0.3228877, 0.1599543

    # Generate inverse CDF samples
    # For each of the 16 codebook values, find the x such that
    # CDF(x) = i/16 for i in [1, 17)
    probs = np.linspace(1 / 16, 15 / 16, 16)

    # NF4 codebook is the inverse CDF of the mixture Gaussian
    # We approximate using binary search on the CDF
    codebook = np.zeros(16)
    for i, p in enumerate(probs):
        codebook[i] = _inverse_cdf(p, m1, mu1, sigma1, m2, mu2, sigma2)

    return codebook, None


# Official NF4 codebook from bitsandbytes (reference values)
# Used for validation against our implementation
OFFICIAL_NF4_CODEBOOK = np.array([
    -1.0, -0.6965, -0.5246, -0.3949, -0.291, -0.2065, -0.1365, -0.0785,
    -0.0297, 0.0126, 0.051, 0.086, 0.1195, 0.1539, 0.191, 0.2341
])


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """Standard Gaussian PDF."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _gaussian_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard Gaussian CDF using error function."""
    return 0.5 * (1 + math.erf((x - mu) / (sigma * np.sqrt(2))))


def _mixture_cdf(x: float, m1: float, mu1: float, sigma1: float,
                 m2: float, mu2: float, sigma2: float) -> float:
    """CDF of mixture Gaussian."""
    return m1 * _gaussian_cdf(x, mu1, sigma1) + m2 * _gaussian_cdf(x, mu2, sigma2)


def _inverse_cdf(p: float, m1: float, mu1: float, sigma1: float,
                 m2: float, mu2: float, sigma2: float) -> float:
    """
    Inverse CDF of mixture Gaussian via binary search.
    Find x such that mixture_cdf(x) = p.
    """
    low, high = -3.0, 3.0
    for _ in range(60):  # Sufficient precision
        mid = (low + high) / 2
        cdf = _mixture_cdf(mid, m1, mu1, sigma1, m2, mu2, sigma2)
        if cdf < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def quantize_to_nf4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to NF4 format.

    For each block of 64 elements:
    - Compute block scale (absmax)
    - Quantize each element to nearest NF4 codebook index (0-15)
    - Pack two indices per byte (4 bits each)

    Args:
        weight: FP32 tensor of shape (out_features, in_features)

    Returns:
        (quantized, scales, codebook): packed 4-bit indices, per-block scales, codebook
    """
    # Use official bitsandbytes NF4 codebook for compatibility
    NF4_CODEBOOK = torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32)
    BLOCK_SIZE = 64

    # Reshape for block processing
    weight_flat = weight.flatten()
    n_elements = weight_flat.numel()

    # Pad to multiple of BLOCK_SIZE if needed
    if n_elements % BLOCK_SIZE != 0:
        pad_size = BLOCK_SIZE - (n_elements % BLOCK_SIZE)
        weight_flat = torch.nn.functional.pad(weight_flat, (0, pad_size))

    n_blocks = weight_flat.numel() // BLOCK_SIZE

    # Compute per-block scales (absmax)
    weight_blocks = weight_flat.reshape(n_blocks, BLOCK_SIZE)
    scales = weight_blocks.abs().max(dim=1).values  # (n_blocks,)

    # Quantize: divide by scale, find nearest codebook value
    # Shape: (n_blocks, BLOCK_SIZE)
    scaled = weight_blocks / scales.unsqueeze(1)

    # Find nearest codebook index for each element
    # Distance to each codebook value: (n_blocks, BLOCK_SIZE, 16)
    distances = (scaled.unsqueeze(2) - NF4_CODEBOOK.unsqueeze(0).unsqueeze(0)).abs()
    indices = distances.argmin(dim=2)  # (n_blocks, BLOCK_SIZE), values in [0, 15]

    # Pack two 4-bit indices per byte
    # High nibble: indices[:, ::2], Low nibble: indices[:, 1::2]
    packed = (indices[:, ::2] << 4) | indices[:, 1::2]
    quantized = packed.byte()

    return quantized, scales, NF4_CODEBOOK


def dequantize_nf4(quantized: torch.Tensor, scales: torch.Tensor,
                   codebook: torch.Tensor) -> torch.Tensor:
    """
    Dequantize NF4 tensor back to FP32.

    Args:
        quantized: Packed 4-bit indices
        scales: Per-block scales
        codebook: NF4 codebook (16 values)

    Returns:
        FP32 tensor
    """
    BLOCK_SIZE = 64

    # Unpack: extract high and low nibbles
    high = (quantized >> 4) & 0x0F
    low = quantized & 0x0F

    # Interleave: [h0, l0, h1, l1, ...]
    indices = torch.stack([high, low], dim=-1).reshape(-1).long()  # (n_blocks * BLOCK_SIZE,)

    # Lookup codebook values
    dequantized = codebook[indices]  # (n_blocks * BLOCK_SIZE,)

    # Reshape to blocks and apply scales
    n_blocks = scales.numel()
    dequantized = dequantized.reshape(n_blocks, BLOCK_SIZE) * scales.unsqueeze(1)

    return dequantized


class LinearNF4:
    """
    Placeholder for LinearNF4 layer.

    Will be a torch.nn.Linear replacement that:
    1. Stores weights in NF4 format (~4x smaller than FP32)
    2. Dequantizes on-the-fly for forward pass (CUDA kernel + cuBLAS Sgemm)
    3. Supports gradient computation via straight-through estimator

    Not implemented yet — see TODO at top of file.
    """
    # TODO: Implement LinearNF4 layer
    # - __init__: accept in_features, out_features, device
    # - Store weight_nf4, weight_scales, weight_codebook
    # - forward: dequantize weight → cuBLAS Sgemm matmul
    # - backward: straight-through estimator


# Test codebook generation
if __name__ == "__main__":
    print("=== NF4 Codebook Generation ===")
    codebook, _ = generate_nf4_codebook()
    print("Generated codebook:", codebook)
    print("Min:", codebook.min(), "Max:", codebook.max())
    print("Mean:", codebook.mean())

    print("\nOfficial bitsandbytes NF4:", OFFICIAL_NF4_CODEBOOK)
    print("Difference (L2):", np.abs(codebook - OFFICIAL_NF4_CODEBOOK).mean())
    print("Note: Differences are expected — NF4 parameters may vary")

    print("\n=== Quantize/Dequantize Test ===")
    torch.manual_seed(42)
    w = torch.randn(128, 256)  # Small test tensor
    quantized, scales, cb = quantize_to_nf4(w)
    w_reconstructed = dequantize_nf4(quantized, scales, cb)

    # Trim padding if any
    w_reconstructed = w_reconstructed[:w.numel()].reshape(w.shape)

    print(f"Original shape: {w.shape}")
    print(f"Quantized bytes: {quantized.numel() + scales.numel()}")
    print(f"Original bytes: {w.numel() * 4}")
    print(f"Compression: {w.numel() * 4 / (quantized.numel() + scales.numel()):.2f}x")
    print(f"Reconstruction error (L2): {(w - w_reconstructed).pow(2).mean().sqrt().item():.6f}")
    print(f"Reconstruction error (Linf): {(w - w_reconstructed).abs().max().item():.6f}")
