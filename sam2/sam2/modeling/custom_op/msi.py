import torch as th
import numpy as np
import logging
import os


def truncated_svd_4_mode_var(X, var=0.99, return_rank=False, return_full_rank=False):
    original_dtype = X.dtype
    device = X.device
    
    if X.dtype in [th.bfloat16, th.float16]:
        X = X.to(th.float32)
    
    if X.dim() == 1:
        X = X.unsqueeze(0)
        
    U, S, Vt = th.linalg.svd(X, full_matrices=False)
    
    total_variance = th.sum(S**2)
    
    if total_variance == 0:
        k = 1
    else:
        explained_variance = th.cumsum(S**2, dim=0) / total_variance
        k_tensor = th.searchsorted(explained_variance, var)
        k = k_tensor.item() + 1
    
    k = min(k, S.shape[0])
    k = max(1, k)

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    if return_rank:
        return U_k, S_k, Vt_k, k
    elif return_full_rank:
        var_ratio = (S**2) / total_variance if total_variance > 0 else th.zeros_like(S)
        return U, S, Vt, var_ratio
    else:
        return U_k, S_k, Vt_k

def Gram_Schmidt(matrix):
    new_matrix = matrix.clone()
    original_type = new_matrix.dtype
    new_matrix, _ = th.linalg.qr(new_matrix.to(dtype=th.float32))
    new_matrix = new_matrix.to(dtype=original_type)
    return new_matrix

def set_random(shape, random_seed=233, device=None):
    th.manual_seed(np.random.RandomState(random_seed).randint(1_000_000_000))
    random_tensor = th.randn(shape, device=device)
    return random_tensor

def unfolding(n, A):
    shape = A.shape
    sizelist = list(range(len(shape)))
    sizelist[n], sizelist[0] = 0, n
    return A.permute(sizelist).reshape(shape[n], -1)

def find_U(unfolded_tensor, previous_U, reuse_U=False, rank=1, device=None):
    if device is None:
        device = unfolded_tensor.device
    n, m = unfolded_tensor.shape
    rank = min(m, n, rank)

    if reuse_U:
        V = th.matmul(unfolded_tensor.t(), previous_U.to(device))
    else:
        V = set_random((m, rank), device=device)

    U = th.matmul(unfolded_tensor, V)
    U = Gram_Schmidt(U)
    return U.detach()

def find_U_mode_n(n, A, rank, reuse_U, previous_U, is_llm=False):
    unfolded_A = unfolding(n, A)
    if is_llm:
        if reuse_U:
            return find_U(unfolded_A, previous_U, reuse_U=reuse_U, rank=rank, device=A.device)
        else:
            u, _, _, k = truncated_svd_4_mode_var(unfolded_A, var=rank, return_rank=True)
            return u
    else:
        return find_U(unfolded_A, previous_U, reuse_U=reuse_U, rank=rank, device=A.device)

def _calculate_hosvd_params(shape, ranks):
    core_params = 1
    for r in ranks:
        core_params *= r
    
    factor_params = 0
    for i, dim in enumerate(shape):
        factor_params += dim * ranks[i]
        
    s_params = sum(ranks)
    
    return core_params + factor_params + s_params

def calculate_ranks_from_compression(tensor, target_variance, layer_name="UnknownLayer"):
    
    B, C, H, W = tensor.shape
    dims = 4
    original_shape = [B, C, H, W]
    original_params = tensor.numel()

    base_ranks = []
    for mode in range(dims):
        permute_order = [mode] + [i for i in range(dims) if i != mode]
        X = tensor.permute(permute_order).reshape(original_shape[mode], -1).contiguous()
        _, _, _, k = truncated_svd_4_mode_var(X, var=target_variance, return_rank=True)
        
        base_ranks.append(k)
    
    final_params = _calculate_hosvd_params(original_shape, base_ranks)
    actual_cr = (1 -  (final_params / original_params if original_params > 0 else 0.0)) * 100
    
 
    return base_ranks

def hosvd_power_iteration(A, previous_Ulist, reuse_U, rank):
    S = A.clone()
    u_list = []
    # print("rank:", rank)

    if type(rank) != list: 
        rank = [rank] * A.dim()
        
    for i in range(A.dim()):
        if reuse_U and previous_Ulist is not None and i < len(previous_Ulist):
            previous_U = previous_Ulist[i]
        else:
            previous_U = None
        
        u = find_U_mode_n(n=i, A=A, rank=rank[i], reuse_U=reuse_U, previous_U=previous_U)
        S = th.tensordot(S, u, dims=([0], [0]))
        u_list.append(u)
    return S, u_list

def restore_hosvd(S, u_list):
    restored_tensor = S.clone()
    for u in u_list:
        restored_tensor = th.tensordot(restored_tensor, u.t(), dims=([0], [0]))
    return restored_tensor