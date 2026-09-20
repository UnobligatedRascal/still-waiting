# PyTorch for Kepler (sm_37) — Build Guide

Official PyTorch 2.0+ dropped Kepler support. This guide builds PyTorch 2.4.0-rc8 from source
with `TORCH_CUDA_ARCH_LIST="3.7"` re-enabled.

Tested on: Debian 12, CUDA 11.8, gcc-11, 8x Tesla K80 (sm_37)

## Prerequisites

```bash
# Install build tools
sudo apt update && sudo apt install -y \
    build-essential cmake g++-11 gcc-11 \
    python3 python3-dev python3-pip python3-venv \
    libopenblas-dev liblapack-dev

# Install CUDA 11.8 toolkit
# Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
# For Debian 12 x86_64:
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
```

## Build Command

```bash
# Create venv (recommended)
python3 -m venv ~/venv-torch && source ~/venv-torch/bin/activate
pip install numpy typing_extensions setuptools wheel

# Clone PyTorch
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.4.0-rc8

# Build
export CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11"
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="3.7"
export MAX_JOBS=36
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda-11.8
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export CUDAFLAGS="-ccbin /usr/bin/g++-11"
export CUDAHOSTCXX=/usr/bin/g++-11
export CXX=/usr/bin/g++-11
export CC=/usr/bin/gcc-11
export CMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"

python3 setup.py develop
```

## Build Notes

- **gcc-11 required**: Newer GCC may have issues with Kepler CUDA code paths
- **TORCH_CUDA_ARCH_LIST="3.7"**: This is the critical flag that re-enables Kepler
- **Build time**: ~3-4 hours on dual Xeon E5-2697v4
- **Disk space**: ~15-20GB needed for build artifacts

## Verify

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"Compute capability: sm_{props.major}{props.minor}")

# Test compute
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
z = x @ y
print(f"Matmul result shape: {z.shape}")
```

Expected output:
```
PyTorch: 2.4.0a0+sm_37
CUDA available: True
Device: Tesla K80
Compute capability: sm_37
Matmul result shape: torch.Size([1024, 1024])
```

## Pre-built Wheel

A PyTorch 2.4.0-rc8 wheel for sm_35/sm_37 is available at:
https://github.com/xiaoran007/Pytorch-for-Kepler

Built for Python 3.9 only. If your system uses a different Python version,
build from source using this guide.

## Known Limitations on Kepler

- **cuBLAS Ex APIs crash**: `cublasGemmEx`, `cublasGemmStridedBatchedEx` → `CUBLAS_STATUS_ARCH_MISMATCH`
- **No tensor cores**: Use F32 compute; FP16 is slow
- **cuDNN may not be available**: cuBLAS legacy APIs work fine

---
UnobligatedRascal
