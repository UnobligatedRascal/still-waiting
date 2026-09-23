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
- [x] CUDA kernel: dequantize + matmul using cuBLAS Sgemm via torch.mm (nf4_dequant.cu)
- [x] PyTorch autograd function for gradient computation (nf4_cuda.py)
- [x] LinearNF4 layer (torch.nn.Module, from_linear factory, CUDA forward)
- [x] Layer replacement pass (replace_linear_with_nf4, target_modules filter)
- [x] Integration into transformers_backend.py
- [ ] NF4 checkpoint save/load (packed weights + scales format)

Integration notes:
- Config flag: "quant": "nf4" in job config → worker routes to NF4 path
- Config flag: model_precision="custom_nf4" → load FP32 → NF4 replace → LoRA wrap
- Checkpoints: Store packed NF4 + scales; support materializing FP32 for surgical edits
- Optimizer: Consider CPU-offloaded Adam moments for extra VRAM savings
- Reporting: Surface compression ratio to orchestrator metrics
- Safety: Auto-fallback to FP32 on kernel failure/NaNs

UnobligatedRascal — Making old hardware sing.
"""

import numpy as np
import torch
import torch.nn as nn

from .nf4_cuda import nf4_linear_forward as _nf4_linear_cuda


# Official NF4 codebook from bitsandbytes (reference values).
# This is THE codebook — all quantization/dequantization uses these values.
OFFICIAL_NF4_CODEBOOK = np.array([
    -1.0, -0.6965, -0.5246, -0.3949, -0.291, -0.2065, -0.1365, -0.0785,
    -0.0297, 0.0126, 0.051, 0.086, 0.1195, 0.1539, 0.191, 0.2341
])


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
    NF4_CODEBOOK = torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32, device=weight.device)
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

    # Find nearest codebook index using searchsorted (codebook is sorted).
    # searchsorted returns insertion point; we check neighbors to find closest.
    # This is O(n×4) instead of O(n×16) broadcast — much faster for large weights.
    insert_points = torch.searchsorted(NF4_CODEBOOK, scaled)
    # Check left neighbor
    left_indices = torch.clamp(insert_points - 1, 0, 15)
    left_values = NF4_CODEBOOK[left_indices]
    # Check right neighbor (insert point)
    right_indices = torch.clamp(insert_points, 0, 15)
    right_values = NF4_CODEBOOK[right_indices]
    # Pick whichever is closer
    left_dist = (scaled - left_values).abs()
    right_dist = (scaled - right_values).abs()
    indices = torch.where(left_dist <= right_dist, left_indices, right_indices)

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
        quantized: Packed 4-bit indices (any shape, flattened internally)
        scales: Per-block scales
        codebook: NF4 codebook (16 values)

    Returns:
        FP32 tensor, flattened (caller must reshape to original shape)
        Length = quantized.numel() * 2 (may include padding)
    """
    BLOCK_SIZE = 64

    # Flatten quantized for uniform processing
    quantized_flat = quantized.flatten()

    # Unpack: extract high and low nibbles
    high = (quantized_flat >> 4) & 0x0F
    low = quantized_flat & 0x0F

    # Interleave: [h0, l0, h1, l1, ...]
    indices = torch.stack([high, low], dim=-1).reshape(-1).long()  # (n_blocks * BLOCK_SIZE,)

    # Lookup codebook values
    dequantized = codebook[indices]  # (n_blocks * BLOCK_SIZE,)

    # Reshape to blocks and apply scales
    n_blocks = scales.numel()
    dequantized = dequantized.reshape(n_blocks, BLOCK_SIZE) * scales.unsqueeze(1)

    # Return flattened — caller must slice to original size and reshape
    return dequantized.flatten()


class LinearNF4(nn.Module):
    """
    NF4-quantized linear layer for Kepler sm_37.

    Replaces torch.nn.Linear with a NF4-quantized version that:
    1. Stores weights in NF4 format (~4x smaller than FP32)
    2. Dequantizes on-the-fly for forward pass via CUDA kernel + cuBLAS Sgemm
    3. Weights are frozen (requires_grad=False) — designed for QLoRA-style training

    Note: Does NOT inherit from nn.Linear to avoid VRAM overhead of dummy weights.
    Uses nf4_lora_wrap() instead of PEFT for LoRA integration.

    Usage:
        # From existing Linear:
        linear = nn.Linear(4096, 4096, bias=True)
        # ... load pretrained weights ...
        nf4_layer = LinearNF4.from_linear(linear)

        # Apply LoRA:
        nf4_layer = nf4_lora_wrap(nf4_layer, r=8, alpha=16)

    Memory savings:
        - FP32 Linear(4096, 4096): ~64MB weights
        - NF4 Linear(4096, 4096): ~16MB weights + ~4KB scales
        - Compression: ~4x

    UnobligatedRascal — Making old hardware sing.
    """

    __constants__ = ['in_features', 'out_features']

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_nf4: torch.Tensor,
        weight_scales: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # NF4-quantized weight: uint8 packed tensor
        self.register_buffer('weight_nf4', weight_nf4)
        # Per-block scales: FP32
        self.register_buffer('weight_scales', weight_scales)
        # Bias (if any): FP32
        self.register_buffer('bias', bias if bias is not None else torch.empty(0))

        # All buffers are frozen — no gradients through quantized weights
        # This is critical for QLoRA: only LoRA adapters are trainable

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        verbose: bool = False
    ) -> 'LinearNF4':
        """
        Convert an existing nn.Linear layer to LinearNF4.

        Quantizes the Linear's weight tensor to NF4 format.
        Preserves bias if present.

        Args:
            linear: Source nn.Linear layer with pretrained weights
            verbose: If True, print quantization stats

        Returns:
            LinearNF4 layer with equivalent behavior (quantized)
        """
        device = linear.weight.device
        weight = linear.weight.data.float()  # Ensure FP32
        bias = linear.bias.data.float() if linear.bias is not None else None

        # Quantize weight
        quantized, scales, codebook = quantize_to_nf4(weight)

        # Move to device
        quantized = quantized.to(device)
        scales = scales.to(device)
        if bias is not None:
            bias = bias.to(device)

        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight_nf4=quantized,
            weight_scales=scales,
            bias=bias,
        )

        if verbose:
            orig_size = linear.weight.numel() * 4  # FP32 bytes
            nf4_size = quantized.numel() + scales.numel() * 4  # uint8 + FP32 scales
            bias_size = bias.numel() * 4 if bias is not None else 0
            print(f"  LinearNF4: {linear.in_features}x{linear.out_features}")
            print(f"    Original: {orig_size / 1024:.1f}KB → NF4: {(nf4_size + bias_size) / 1024:.1f}KB "
                  f"({orig_size / (nf4_size + bias_size):.2f}x compression)")

        return layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dequantize NF4 weights → matmul → add bias.

        Weights are frozen — gradients flow only to input (for LoRA adapters).

        Args:
            input: FP32 tensor of shape (batch, in_features) or (batch, seq_len, in_features)

        Returns:
            FP32 output of shape (batch, out_features) or (batch, seq_len, out_features)
        """
        # Ensure input is FP32 (Kepler: F32 compute)
        if input.dtype != torch.float32:
            input = input.float()

        # Move NF4 buffers to same device as input (handles DDP/device transfers)
        weight_nf4 = self.weight_nf4.to(input.device)
        weight_scales = self.weight_scales.to(input.device)

        if input.is_cuda:
            # CUDA path: dequantize + matmul via custom kernel
            output = _nf4_linear_cuda(
                input=input,
                weight_nf4=weight_nf4,
                weight_scales=weight_scales,
                bias=self.bias,
                true_out_features=self.out_features,
                in_features=self.in_features,
            )
        else:
            # CPU fallback: use Python reference implementation
            codebook = torch.tensor(OFFICIAL_NF4_CODEBOOK, dtype=torch.float32, device=input.device)
            weight_flat = dequantize_nf4(weight_nf4, weight_scales, codebook)

            # Trim padding if present and reshape
            total_elements = self.out_features * self.in_features
            weight_fp32 = weight_flat[:total_elements].reshape(self.out_features, self.in_features)

            # Standard linear: y = x @ W^T + b
            output = torch.nn.functional.linear(input, weight_fp32, self.bias)

        return output

    def extra_repr(self) -> str:
        nf4_bytes = self.weight_nf4.numel()
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"nf4_bytes={nf4_bytes}, bias={self.bias.numel() > 0}"


def replace_linear_with_nf4(
    model: nn.Module,
    target_modules: list[str] | None = None,
    verbose: bool = True
) -> nn.Module:
    """
    Replace selected nn.Linear layers in a model with LinearNF4.

    Walks the model recursively and replaces nn.Linear modules whose names
    contain any of the target module patterns (e.g., 'q_proj', 'k_proj').

    Args:
        model: PyTorch model to modify
        target_modules: List of name patterns to match. If None, replace ALL Linear layers.
        verbose: If True, print progress and stats

    Returns:
        Modified model (in-place)

    Example:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
        replace_linear_with_nf4(
            model,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]
        )
    """
    if target_modules is None:
        target_modules = [""]  # Match all

    replaced_count = 0
    skipped_count = 0
    total_original_bytes = 0
    total_nf4_bytes = 0

    def _should_replace(module_name: str) -> bool:
        """Check if module name matches any target pattern."""
        if not target_modules:
            return False
        # Match if any target pattern appears in the module name (suffix match for common patterns)
        for pattern in target_modules:
            if pattern == "":
                return True  # Empty pattern matches all
            if pattern in module_name:
                return True
        return False

    def _replace_module(parent: nn.Module, name: str, child: nn.Module) -> None:
        nonlocal replaced_count, skipped_count, total_original_bytes, total_nf4_bytes

        if not isinstance(child, nn.Linear):
            return

        if not _should_replace(name):
            return

        # Replace
        nf4_layer = LinearNF4.from_linear(child, verbose=(verbose and replaced_count < 10))

        total_original_bytes += child.weight.numel() * 4
        total_nf4_bytes += nf4_layer.weight_nf4.numel() + nf4_layer.weight_scales.numel() * 4

        setattr(parent, name, nf4_layer)
        replaced_count += 1

    # Pre-compute module dict for O(1) parent lookups
    module_dict = dict(model.named_modules())

    # Walk model and replace
    for name, module in model.named_modules():
        # Find parent and child name
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent = module_dict[parent_name]
        else:
            parent = model
            child_name = name

        _replace_module(parent, child_name, module)

    # Report summary
    if verbose:
        print("\nNF4 Replacement Summary:")
        print(f"  Replaced: {replaced_count} Linear layers")
        print(f"  Original weights: {total_original_bytes / 1024**2:.1f}MB")
        print(f"  NF4 weights: {total_nf4_bytes / 1024**2:.1f}MB")
        print(f"  Compression: {total_original_bytes / total_nf4_bytes:.2f}x")
        if total_original_bytes > 0:
            print(f"  VRAM saved: {(total_original_bytes - total_nf4_bytes) / 1024**2:.1f}MB")

    return model


def nf4_lora_wrap(
    layer: LinearNF4,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Wrap a LinearNF4 layer with LoRA adapters.

    Implements standard LoRA (Hu et al. 2021): freezes base weights, learns
    low-rank decomposition ΔW = B @ A where A: (r, in_features), B: (out_features, r).

    Forward: y = base(x) + scaling * B @ A @ dropout(x)

    Args:
        layer: LinearNF4 layer to wrap
        r: LoRA rank (typically 4-64)
        alpha: Scaling factor (typically = r or 2r)
        dropout: Dropout before LoRA path

    Returns:
        nn.Module wrapping the layer with LoRA adapters
    """
    return NF4LoRALayer(layer, r=r, alpha=alpha, dropout=dropout)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str] | None = None,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    freeze_base: bool = True,
) -> tuple[nn.Module, list[str]]:
    """
    Apply LoRA adapters to all LinearNF4 layers in a model.

    Args:
        model: Model with LinearNF4 layers
        target_modules: Name patterns to match (only LoRA-wrap matching layers)
        r: LoRA rank
        alpha: LoRA scaling
        dropout: LoRA dropout
        freeze_base: If True, freeze all non-LoRA parameters (standard QLoRA behavior)

    Returns:
        (modified_model, list_of_lora_param_names)
    """
    if target_modules is None:
        target_modules = [""]

    lora_param_names = []

    def _should_apply(module_name: str) -> bool:
        if not target_modules:
            return False
        for pattern in target_modules:
            if pattern == "" or pattern in module_name:
                return True
        return False

    for name, module in list(model.named_modules()):
        if isinstance(module, LinearNF4) and _should_apply(name):
            parent_name, child_name = name.rsplit('.', 1) if '.' in name else (None, name)
            parent = model if parent_name is None else dict(model.named_modules())[parent_name]

            wrapped = nf4_lora_wrap(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, wrapped)

            # Track LoRA parameter names
            for pname, _ in wrapped.named_parameters():
                if pname.startswith('lora_'):
                    lora_param_names.append(f"{name}.{pname}")

    # Freeze all non-LoRA parameters (standard QLoRA: only LoRA adapters are trained)
    if freeze_base:
        for name, param in model.named_parameters():
            if not any(name.startswith(ln) for ln in lora_param_names):
                param.requires_grad = False

    return model, lora_param_names


class NF4LoRALayer(nn.Module):
    """
    LoRA wrapper for LinearNF4 layers.

    Implements: y = base_layer(x) + (alpha/r) * B @ A @ dropout(x)

    Base layer weights are frozen. Only A and B are trainable.
    B is zero-initialized so LoRA starts as identity (no change to base output).
    """

    def __init__(
        self,
        base_layer: LinearNF4,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze base layer
        for p in base_layer.parameters():
            p.requires_grad = False

        # LoRA matrices: ΔW = B @ A
        # A: (r, in_features), random init
        # B: (out_features, r), zero init
        self.lora_A = nn.Parameter(
            torch.randn(r, base_layer.in_features) * (1 / base_layer.in_features)
        )
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))

        # Optional dropout before LoRA
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base NF4 path (frozen, no gradients through quantized weights)
        result = self.base_layer(x).detach()

        # LoRA path: scaling * B @ A @ dropout(x)
        # Handle arbitrary input shapes: x is (..., in_features)
        lora_input = x
        if self.dropout is not None:
            lora_input = self.dropout(lora_input)

        # Reshape to 2D for matmul: (..., in_features) → (total, in_features)
        original_shape = lora_input.shape
        lora_input_2d = lora_input.reshape(-1, self.base_layer.in_features)

        # A @ x^T gives (r, total), then B @ (...) gives (out_features, total)
        # Transpose back: (total, out_features)
        lora_update_2d = (self.lora_B @ self.lora_A @ lora_input_2d.t()).t()

        # Reshape back to original shape
        lora_update = lora_update_2d.reshape(*original_shape[:-1], self.base_layer.out_features)

        return result + self.scaling * lora_update

    def __repr__(self) -> str:
        return f"NF4LoRA({self.base_layer}, r={self.r}, alpha={self.alpha})"


# VRAM monitoring utilities

def get_vram_info(device: int | torch.device | None = None) -> dict:
    """Get VRAM usage info for a device."""
    if not torch.cuda.is_available():
        return {"available": False}
    if device is None:
        device = 0
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total - reserved
    return {
        "device": device,
        "name": props.name,
        "total_gb": total / 1024**3,
        "allocated_gb": allocated / 1024**3,
        "reserved_gb": reserved / 1024**3,
        "free_gb": free / 1024**3,
        "free_pct": 100 * free / total,
    }


def log_vram(msg: str = "", device: int | None = 0) -> None:
    """Print VRAM status with optional message."""
    info = get_vram_info(device)
    prefix = f"[{msg}] " if msg else ""
    print(f"{prefix}VRAM GPU{device}: {info['allocated_gb']:.2f}GB / {info['total_gb']:.1f}GB "
          f"({info['free_gb']:.2f}GB free)")


# Quick self-test
if __name__ == "__main__":
    print("=== NF4 Quantize/Dequantize Test ===")
    print(f"Using official bitsandbytes NF4 codebook: {OFFICIAL_NF4_CODEBOOK}")

    torch.manual_seed(42)
    w = torch.randn(128, 256)  # Small test tensor
    quantized, scales, cb = quantize_to_nf4(w)
    w_reconstructed = dequantize_nf4(quantized, scales, cb)

    # Trim padding if any
    w_reconstructed = w_reconstructed[:w.numel()].reshape(w.shape)

    print(f"\nOriginal shape: {w.shape}")
    print(f"Quantized bytes: {quantized.numel() + scales.numel()}")
    print(f"Original bytes: {w.numel() * 4}")
    print(f"Compression: {w.numel() * 4 / (quantized.numel() + scales.numel()):.2f}x")
    print(f"Reconstruction error (L2): {(w - w_reconstructed).pow(2).mean().sqrt().item():.6f}")
    print(f"Reconstruction error (Linf): {(w - w_reconstructed).abs().max().item():.6f}")
