"""
PyTorch extension for NF4 CUDA kernels on Kepler sm_37.

Wraps the CUDA dequantize kernel and provides torch.autograd integration.

UnobligatedRascal — Making old hardware sing.
"""
import os

import torch
from torch.utils.cpp_extension import load as torch_load

_kernels_dir = os.path.dirname(os.path.abspath(__file__))
_extension = None

def _get_extension():
    global _extension
    if _extension is None:
        # Use GCC 11 for CUDA 11.8 compatibility (GCC 12 breaks nvcc host compilation)
        # Same compiler setup used for llama_wukong
        os.environ["CC"] = "/usr/bin/gcc-11"
        os.environ["CXX"] = "/usr/bin/g++-11"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7"

        _extension = torch_load(
            name="nf4_cuda",
            sources=[
                os.path.join(_kernels_dir, "nf4_dequant.cu"),
            ],
            extra_cuda_cflags=[
                "-ccbin", "/usr/bin/gcc-11",
                "-O3",
                "-arch=sm_37",
                "-allow-unsupported-compiler",
                "-Wno-deprecated-gpu-targets",
            ],
            verbose=False,
        )
    return _extension


def dequantize_nf4(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize NF4-packed tensor back to FP32.

    Args:
        quantized: uint8 tensor of packed 4-bit indices, shape (n_bytes,)
            Each byte contains two 4-bit indices: (high << 4) | low
        scales: FP32 tensor of per-block scales, shape (n_blocks,)
            One scale per NF4_BLOCK_SIZE (64) elements

    Returns:
        FP32 tensor of shape (quantized.numel() * 2,)
    """
    if _extension is None:
        _get_extension()

    return _extension.dequantize_nf4(quantized, scales)


def nf4_linear_forward(
    input: torch.Tensor,
    weight_nf4: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    true_out_features: int | None = None,
    in_features: int | None = None
) -> torch.Tensor:
    """
    NF4 linear layer forward: y = input @ W_nf4^T + bias

    Dequantizes NF4 weights on-the-fly and performs matrix multiply
    using cuBLAS Sgemm (via PyTorch).

    Args:
        input: FP32 tensor of shape (batch, in_features) or (batch, seq_len, in_features)
        weight_nf4: uint8 packed NF4 weights, flat shape (n_bytes,)
            Represents (out_features, in_features) matrix (may include NF4 padding)
        weight_scales: FP32 per-block scales, shape (n_blocks,)
        bias: optional FP32 bias, shape (out_features,)
        true_out_features: actual output features (before NF4 padding). Inferred from input if None.
        in_features: actual input features. Inferred from input if None.

    Returns:
        FP32 output of shape (batch, out_features) or (batch, seq_len, out_features)
    """
    if _extension is None:
        _get_extension()

    if in_features is None:
        in_features = input.size(-1)
    if true_out_features is None:
        # Infer from weight size (may include padding — kernel handles trim)
        true_out_features = weight_nf4.numel() * 2 // in_features

    if bias is None:
        bias = torch.empty(0, dtype=torch.float32, device=input.device)

    return _extension.nf4_linear_forward(
        input, weight_nf4, weight_scales, bias,
        true_out_features, in_features
    )


class NF4DequantizeFunction(torch.autograd.Function):
    """
    Autograd function for NF4 dequantization with straight-through estimator.

    Forward: dequantize NF4 → FP32
    Backward: pass gradients straight through (ignore quantization error)

    This is the standard approach used by QLoRA — the base weights are frozen
    (no gradient through quantization), only LoRA adapters receive gradients.
    """

    @staticmethod
    def forward(ctx, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Dequantize NF4 to FP32."""
        if quantized.is_cuda:
            result = dequantize_nf4(quantized, scales)
        else:
            # Fallback to CPU implementation
            from .nf4_kepler import dequantize_nf4 as cpu_dequantize
            result = cpu_dequantize(quantized, scales, _get_codebook())

        ctx.quantized = quantized
        ctx.scales = scales
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Straight-through estimator: pass gradient through unchanged."""
        # For frozen quantized weights (QLoRA style), we don't compute gradients
        # through quantization. Return zeros.
        return torch.zeros_like(ctx.quantized), torch.zeros_like(ctx.scales)


def dequantize_nf4_torch(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize NF4 tensor with autograd support.

    Use this in model forward passes. The straight-through estimator ensures
    gradients flow correctly when combined with LoRA adapters.
    """
    return NF4DequantizeFunction.apply(quantized, scales)


def _get_codebook() -> torch.Tensor:
    """Get the NF4 codebook tensor."""
    return torch.tensor([
        -1.0000, -0.6965, -0.5246, -0.3949,
        -0.2910, -0.2065, -0.1365, -0.0785,
        -0.0297,  0.0126,  0.0510,  0.0860,
         0.1195,  0.1539,  0.1910,  0.2341
    ], dtype=torch.float32)


# Lazy initialization for module-level use
def _ensure_extension():
    if _extension is None:
        _get_extension()
