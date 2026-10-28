
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, memory, pos=None):
        if isinstance(memory, dict):
            S, U = memory["S"], memory["U_list"]
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
                    p = F.linear(pos.reshape(-1, pos.shape[-1]), self.weight, None)
                    x = x + p.reshape(B, -1, C)
            return x

        elif isinstance(memory, torch.Tensor):
            d0, d1, d2 = memory.shape
            if d0 > d1:
                memory = memory.permute(1, 0, 2)
            
            B, N, C_in = memory.shape
            
            if C_in == self.in_features:
                out = F.linear(memory.reshape(-1, C_in), self.weight, self.bias)
                out = out.reshape(B, N, self.out_features)
            elif C_in == self.out_features:
                out = memory
                if self.bias is not None:
                    out = out + self.bias.view(1, 1, -1)
            else:
                raise RuntimeError(f"Channel mismatch: {C_in}")

            if pos is not None:
                if pos.dim() == 3 and pos.shape[0] != B:
                    pos = pos.permute(1, 0, 2)
                    p = F.linear(pos.reshape(-1, pos.shape[-1]), self.weight, None)
                    out = out + p.reshape(B, N, self.out_features)
            return out
        
        raise TypeError(f"Unsupported type: {type(memory)}")

