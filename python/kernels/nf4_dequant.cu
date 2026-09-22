/*
NF4 CUDA kernels for Kepler sm_37.
PyTorch C++ extension with dequantize and fused matmul.

Based on QLoRA paper (Dettmers et al. 2023).
Uses legacy cuBLAS Sgemm (NO Ex APIs — crash on sm_37).

UnobligatedRascal — Making old hardware sing.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define NF4_BLOCK_SIZE 64

// CUDA dequantize kernel (internal)
template<int NUM_PER_THREAD>
__global__ void dequantize_nf4_kernel(
    const unsigned char* quantized,
    const float* scales,
    float* out,
    const int n_bytes
) {
    __shared__ float shared_cb[16];

    if (threadIdx.x == 0) shared_cb[0] = -1.0000f;
    if (threadIdx.x == 1) shared_cb[1] = -0.6965f;
    if (threadIdx.x == 2) shared_cb[2] = -0.5246f;
    if (threadIdx.x == 3) shared_cb[3] = -0.3949f;
    if (threadIdx.x == 4) shared_cb[4] = -0.2910f;
    if (threadIdx.x == 5) shared_cb[5] = -0.2065f;
    if (threadIdx.x == 6) shared_cb[6] = -0.1365f;
    if (threadIdx.x == 7) shared_cb[7] = -0.0785f;
    if (threadIdx.x == 8) shared_cb[8] = -0.0297f;
    if (threadIdx.x == 9) shared_cb[9] = 0.0126f;
    if (threadIdx.x == 10) shared_cb[10] = 0.0510f;
    if (threadIdx.x == 11) shared_cb[11] = 0.0860f;
    if (threadIdx.x == 12) shared_cb[12] = 0.1195f;
    if (threadIdx.x == 13) shared_cb[13] = 0.1539f;
    if (threadIdx.x == 14) shared_cb[14] = 0.1910f;
    if (threadIdx.x == 15) shared_cb[15] = 0.2341f;

    __syncthreads();

    const int total_threads = gridDim.x * blockDim.x;
    const int base_byte = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = base_byte; b < n_bytes; b += total_threads) {
        const unsigned char byte = quantized[b];

        const unsigned char high_idx = byte >> 4;
        const unsigned char low_idx = byte & 0x0F;

        const int out_idx0 = b * 2;
        const int out_idx1 = b * 2 + 1;
        const int block_idx0 = out_idx0 / NF4_BLOCK_SIZE;
        const int block_idx1 = out_idx1 / NF4_BLOCK_SIZE;

        out[out_idx0] = shared_cb[high_idx] * scales[block_idx0];
        out[out_idx1] = shared_cb[low_idx] * scales[block_idx1];
    }
}

// Launch dequantize kernel
void launch_dequantize(
    const unsigned char* d_quantized,
    const float* d_scales,
    float* d_out,
    const int n_bytes,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n_bytes + block_size - 1) / block_size;

    dequantize_nf4_kernel<1><<<grid_size, block_size, 0, stream>>>(
        d_quantized, d_scales, d_out, n_bytes);
}

/**
 * Simple dequantize function for testing.
 */
torch::Tensor dequantize_nf4(
    const torch::Tensor& quantized,
    const torch::Tensor& scales
) {
    TORCH_CHECK(quantized.is_cuda(), "quantized must be on CUDA");
    TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
    TORCH_CHECK(quantized.dtype() == torch::kUInt8, "quantized must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");

    const int n_bytes = quantized.numel();
    const int n_elements = n_bytes * 2;

    auto out = torch::empty({n_elements},
        torch::TensorOptions().dtype(torch::kFloat32).device(quantized.device()));

    cudaStream_t stream = 0;
    launch_dequantize(
        quantized.data_ptr<unsigned char>(),
        scales.data_ptr<float>(),
        out.data_ptr<float>(),
        n_bytes,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA dequantize error: ", cudaGetErrorString(err));
    }

    return out;
}

/**
 * NF4 linear forward: y = x @ W_nf4^T + bias
 *
 * Args:
 *   input: (batch_size, seq_len, in_features) or (batch_size, in_features)
 *   weight_nf4: packed NF4 weights, shape (out_features, in_features) flattened
 *   weight_scales: per-block scales, shape (n_blocks,)
 *   bias: optional bias, shape (out_features,)
 *
 * Returns:
 *   output: (batch_size, seq_len, out_features) or (batch_size, out_features)
 */
torch::Tensor nf4_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight_nf4,
    const torch::Tensor& weight_scales,
    const torch::Tensor& bias
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(weight_nf4.is_cuda(), "weight_nf4 must be on CUDA");
    TORCH_CHECK(weight_scales.is_cuda(), "weight_scales must be on CUDA");
    TORCH_CHECK(weight_nf4.dtype() == torch::kUInt8, "weight_nf4 must be uint8");
    TORCH_CHECK(weight_scales.dtype() == torch::kFloat32, "weight_scales must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32 (Kepler: F32 compute)");

    const int batch_size = input.size(0);
    const int in_features = input.size(-1);
    const int out_features = weight_nf4.numel() * 2 / in_features; // NF4: 2 values per byte

    // Handle both 2D (batch, in_features) and 3D (batch, seq_len, in_features)
    bool is_3d = input.dim() == 3;
    const int total_m = is_3d ? batch_size * input.size(1) : batch_size;

    // Step 1: Dequantize weights to temporary buffer
    // Weight shape: (out_features, in_features)
    const int weight_elements = out_features * in_features;
    const int weight_bytes = weight_elements / 2;

    at::Tensor weight_fp32 = torch::empty({out_features, in_features},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    cudaStream_t stream = 0;
    launch_dequantize(
        weight_nf4.data_ptr<unsigned char>(),
        weight_scales.data_ptr<float>(),
        weight_fp32.data_ptr<float>(),
        weight_bytes,
        stream);

    // Step 2: Reshape input to (total_m, in_features) for matmul
    at::Tensor input_2d = is_3d ? input.view({total_m, in_features}) : input;

    // Step 3: y = input @ W^T using PyTorch (which uses cuBLAS internally)
    at::Tensor output_2d = torch::mm(input_2d, weight_fp32.t());

    // Step 4: Add bias if present
    if (bias.defined() && bias.numel() > 0) {
        output_2d = output_2d + bias;
    }

    // Step 5: Reshape back to original dimensions
    if (is_3d) {
        return output_2d.view({batch_size, input.size(1), out_features});
    } else {
        return output_2d;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_nf4", &dequantize_nf4,
          "NF4 dequantize kernel (CUDA)");
    m.def("nf4_linear_forward", &nf4_linear_forward,
          "NF4 linear layer forward (dequantize + matmul)",
          py::arg("input"),
          py::arg("weight_nf4"),
          py::arg("weight_scales"),
          py::arg("bias"));
}
