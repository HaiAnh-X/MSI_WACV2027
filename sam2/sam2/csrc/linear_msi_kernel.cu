// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// Fast Fused Tucker Contraction & Linear Operator CUDA Kernel for SAM2 (Linear_MSI)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

namespace linear_msi {

template <typename T>
struct AccType {
    using type = float;
};

template <>
struct AccType<double> {
    using type = double;
};

// Fast templated kernel for Linear_MSI forward pass
// Contraction: x = einsum("abcd,ea,fb,gc,hd->efgh", S, U0, Uc, U2, U3) + bias
// Permute & reshape: (B, C_out, H, W) -> (B, H*W, C_out)
// Fused pos addition: out = x + pos
template <typename scalar_t>
__global__ void linear_msi_forward_kernel(
    const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ U0,
    const scalar_t* __restrict__ Uc,
    const scalar_t* __restrict__ U2,
    const scalar_t* __restrict__ U3,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ pos,
    scalar_t* __restrict__ out,
    const int B,
    const int C_out,
    const int H,
    const int W,
    const int r0,
    const int r1,
    const int r2,
    const int r3,
    const bool has_bias,
    const bool has_pos
) {
    using acc_t = typename AccType<scalar_t>::type;
    const int N = H * W;

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (b >= B || n >= N || c >= C_out) return;

    const int i = n / W; // height index
    const int j = n % W; // width index

    acc_t val = static_cast<acc_t>(0);

    for (int d = 0; d < r1; ++d) {
        acc_t u_c_val = static_cast<acc_t>(Uc[c * r1 + d]);
        acc_t spatial_factor = static_cast<acc_t>(0);

        for (int e = 0; e < r2; ++e) {
            acc_t u2_val = static_cast<acc_t>(U2[i * r2 + e]);

            for (int f = 0; f < r3; ++f) {
                acc_t u3_val = static_cast<acc_t>(U3[j * r3 + f]);
                acc_t s_u0_sum = static_cast<acc_t>(0);

                for (int a = 0; a < r0; ++a) {
                    acc_t u0_val = static_cast<acc_t>(U0[b * r0 + a]);
                    int s_idx = ((a * r1 + d) * r2 + e) * r3 + f;
                    acc_t s_val = static_cast<acc_t>(S[s_idx]);
                    s_u0_sum += u0_val * s_val;
                }

                spatial_factor += u2_val * u3_val * s_u0_sum;
            }
        }

        val += u_c_val * spatial_factor;
    }

    if (has_bias) {
        val += static_cast<acc_t>(bias[c]);
    }

    const int out_idx = (b * N + n) * C_out + c;
    if (has_pos) {
        val += static_cast<acc_t>(pos[out_idx]);
    }

    out[out_idx] = static_cast<scalar_t>(val);
}

// C++ launcher function called from PyTorch binding
torch::Tensor linear_msi_cuda_forward(
    const torch::Tensor& S,
    const torch::Tensor& U0,
    const torch::Tensor& weight,
    const torch::Tensor& U1,
    const torch::Tensor& U2,
    const torch::Tensor& U3,
    const c10::optional<torch::Tensor>& bias_opt,
    const c10::optional<torch::Tensor>& pos_opt
) {
    TORCH_CHECK(S.is_cuda(), "S must be a CUDA tensor");
    TORCH_CHECK(U0.is_cuda(), "U0 must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(U1.is_cuda(), "U1 must be a CUDA tensor");
    TORCH_CHECK(U2.is_cuda(), "U2 must be a CUDA tensor");
    TORCH_CHECK(U3.is_cuda(), "U3 must be a CUDA tensor");

    const int r0 = S.size(0);
    const int r1 = S.size(1);
    const int r2 = S.size(2);
    const int r3 = S.size(3);

    const int B = U0.size(0);
    const int C_out = weight.size(0);
    const int H = U2.size(0);
    const int W = U3.size(0);
    const int N = H * W;

    // Compute Uc = weight @ U1 -> shape (C_out, r1)
    torch::Tensor Uc = torch::matmul(weight, U1);

    auto out_options = torch::TensorOptions().dtype(S.dtype()).device(S.device());
    torch::Tensor out = torch::empty({B, N, C_out}, out_options);

    const bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
    const bool has_pos = pos_opt.has_value() && pos_opt.value().defined();

    const torch::Tensor bias = has_bias ? bias_opt.value() : torch::Tensor();
    const torch::Tensor pos = has_pos ? pos_opt.value() : torch::Tensor();

    dim3 block(16, 16, 1);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (C_out + block.y - 1) / block.y,
        B
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        S.scalar_type(),
        "linear_msi_forward_cuda",
        ([&] {
            linear_msi_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
                S.data_ptr<scalar_t>(),
                U0.data_ptr<scalar_t>(),
                Uc.data_ptr<scalar_t>(),
                U2.data_ptr<scalar_t>(),
                U3.data_ptr<scalar_t>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                has_pos ? pos.data_ptr<scalar_t>() : nullptr,
                out.data_ptr<scalar_t>(),
                B, C_out, H, W,
                r0, r1, r2, r3,
                has_bias, has_pos
            );
        })
    );

    return out;
}

} // namespace linear_msi
