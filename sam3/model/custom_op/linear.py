import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Optional
from torch import Tensor

_CUDA_MSI_EXT = None

def _get_cuda_msi_ext():
    global _CUDA_MSI_EXT
    if _CUDA_MSI_EXT is not None:
        return _CUDA_MSI_EXT
    
    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_dir = os.path.abspath(os.path.join(curr_dir, "..", "..", "csrc"))
        cu_file = os.path.join(csrc_dir, "linear_msi_kernel.cu")
        cpp_file = os.path.join(csrc_dir, "linear_msi_cuda.cpp")
        
        if os.path.exists(cu_file) and os.path.exists(cpp_file):
            _CUDA_MSI_EXT = load(
                name="linear_msi_cuda",
                sources=[cpp_file, cu_file],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=False
            )
            return _CUDA_MSI_EXT
    except Exception as e:
        print(f"[MSI CUDA load error]: {e}")
        _CUDA_MSI_EXT = None
    return None

class Linear_MSI(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = fan_in ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, memory, pos=None):
        if isinstance(memory, dict):
            S, U = memory["S"], memory["U_list"]
            B = U[0].shape[0]

            # Try custom CUDA kernel path if on GPU
            if S.is_cuda:
                cuda_ext = _get_cuda_msi_ext()
                if cuda_ext is not None:
                    try:
                        pos_tensor = pos
                        if pos_tensor is not None:
                            if pos_tensor.dim() == 3 and pos_tensor.shape[0] != B:
                                pos_tensor = pos_tensor.permute(1, 0, 2)
                            if pos_tensor.shape[-1] != self.out_features:
                                p = F.linear(pos_tensor.reshape(-1, pos_tensor.shape[-1]), self.weight, None)
                                pos_tensor = p.reshape(B, -1, self.out_features)

                        return cuda_ext.forward(
                            S, U[0], self.weight, U[1], U[2], U[3],
                            self.bias, pos_tensor
                        )
                    except Exception:
                        pass # Fall through to PyTorch einsum reference implementation

            # Standard PyTorch einsum reference implementation
            Uc = torch.matmul(self.weight, U[1])
            x = torch.einsum("abcd,ea,fb->efcd", S, U[0], Uc)
            x = torch.einsum("efcd,gc,hd->efgh", x, U[2], U[3])
            B, C, H, W = x.shape
            if self.bias is not None:
                x = x + self.bias.view(1, C, 1, 1)
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

            if pos is not None:
                if pos.dim() == 3 and pos.shape[0] != B:
                    pos = pos.permute(1, 0, 2)
                if pos.shape[-1] == C:
                    x = x + pos
                else:
                    p = F.linear(pos.reshape(-1, pos.shape[-1]), self.weight, None)
                    x = x + p.reshape(B, -1, C)
            return x

        elif isinstance(memory, torch.Tensor):
            C_in = memory.shape[-1]
            if C_in == self.in_features:
                out = F.linear(memory, self.weight, self.bias)
            elif C_in == self.out_features:
                out = memory
                if self.bias is not None:
                    out = out + self.bias
            else:
                raise RuntimeError(f"Channel mismatch: {C_in} vs in_features={self.in_features}")

            if pos is not None:
                if pos.shape == out.shape:
                    out = out + pos
                elif pos.shape[-1] == out.shape[-1]:
                    p = F.linear(pos, self.weight, None)
                    out = out + p
                else:
                    p = F.linear(pos.reshape(-1, pos.shape[-1]), self.weight, None)
                    out = out + p.reshape(out.shape)
            return out
        
        raise TypeError(f"Unsupported type: {type(memory)}")

def wrap_linear_with_msi(model):
    for name, module in model.named_modules():
        if ("memory_attention" in name or "transformer" in name or "encoder" in name) and ("k_proj" in name or "v_proj" in name):
            if isinstance(module, torch.nn.Linear):
                old_w = module.weight.data.clone()
                old_b = module.bias.data.clone() if module.bias is not None else None
                
                new_layer = Linear_MSI(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=(old_b is not None),
                    device=old_w.device,
                    dtype=old_w.dtype
                )
                new_layer.weight.data.copy_(old_w)
                if old_b is not None:
                    new_layer.bias.data.copy_(old_b)
                
                parts = name.split('.')
                parent = model.get_submodule('.'.join(parts[:-1]))
                setattr(parent, parts[-1], new_layer)
    return model
