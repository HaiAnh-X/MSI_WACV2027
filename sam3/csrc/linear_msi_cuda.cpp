// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// PyTorch C++ Pybind11 binding for Linear_MSI CUDA custom operator

#include <torch/extension.h>

namespace linear_msi {
torch::Tensor linear_msi_cuda_forward(
    const torch::Tensor& S,
    const torch::Tensor& U0,
    const torch::Tensor& weight,
    const torch::Tensor& U1,
    const torch::Tensor& U2,
    const torch::Tensor& U3,
    const c10::optional<torch::Tensor>& bias_opt,
    const c10::optional<torch::Tensor>& pos_opt
);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_msi::linear_msi_cuda_forward, "Linear_MSI CUDA forward pass");
}
