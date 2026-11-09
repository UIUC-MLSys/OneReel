"""Varlen attention on FlashAttention-2 or FlashAttention-3 (flash_attn_interface, Hopper)."""
from __future__ import annotations

import numpy as np
import torch

try:
    import flash_attn_interface as _fa3
except ImportError:
    _fa3 = None
from flash_attn import flash_attn_func as _fa2_func, flash_attn_varlen_func as _fa2_varlen

BACKEND = 'fa2'          # 'fa2' | 'fa3' | 'auto' (fa3 when importable)


def backend() -> str:
    if BACKEND == 'auto':
        return 'fa3' if _fa3 is not None else 'fa2'
    if BACKEND == 'fa3' and _fa3 is None:
        raise ImportError('FlashAttention-3 (flash_attn_interface) is not installed')
    return BACKEND


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def cu_seqlens(lengths, device='cuda') -> torch.Tensor:
    return torch.tensor([0] + np.cumsum(lengths).tolist(), dtype=torch.int32, device=device)


def dense(q, k, v):
    """q, k, v [B, S, H, D]"""
    if backend() == 'fa3':
        return _first(_fa3.flash_attn_func(q, k, v))
    return _fa2_func(q, k, v)


def varlen(q, k, v, cu_q, cu_k, max_q, max_k):
    """packed q [Nq, H, D], k / v [Nk, H, D]"""
    if backend() == 'fa3':
        return _first(_fa3.flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k))
    return _fa2_varlen(q, k, v, cu_q, cu_k, max_q, max_k)


def varlen_lse(q, k, v, cu_q, cu_k, max_q, max_k):
    """varlen with the softmax log-sum-exp: (out [Nq, H, D], lse [H, Nq] fp32)"""
    if backend() == 'fa3':
        r = _fa3._flash_attn_forward(q, k, v, None, None, None, None, cu_q, cu_k, None, None, None, max_q, max_k, None,
                                     None, None, None, None, None, None, None, None, q.shape[-1] ** -0.5, False)
        return r[0], r[1]
    o, lse, _ = _fa2_varlen(q, k, v, cu_q, cu_k, max_q, max_k, return_attn_probs=True)
    return o, lse


def packed_lse(segments):
    """as packed, with each segment's log-sum-exp: list of (out, lse [H, n])"""
    lq = [s[0].shape[0] for s in segments]
    lk = [s[1].shape[0] for s in segments]
    o, lse = varlen_lse(torch.cat([s[0] for s in segments]), torch.cat([s[1] for s in segments]),
                        torch.cat([s[2] for s in segments]), cu_seqlens(lq), cu_seqlens(lk), max(lq), max(lk))
    return list(zip(o.split(lq), lse.split(lq, dim=1)))


def packed(segments):
    """segments: list of (q, k, v) sequences -> list of outputs, one varlen call"""
    lq = [s[0].shape[0] for s in segments]
    lk = [s[1].shape[0] for s in segments]
    o = varlen(torch.cat([s[0] for s in segments]), torch.cat([s[1] for s in segments]), torch.cat([s[2] for s in segments]),
               cu_seqlens(lq), cu_seqlens(lk), max(lq), max(lk))
    return o.split(lq)


def overlapping_ranges(q, k, v, q_ranges, k_ranges):
    """attention of q rows q_ranges[j] to k / v rows k_ranges[j] without copying k / v: FA3 reads the overlapping
    contiguous key ranges in place (cu_seqlens_k as offsets, seqused_k as lengths); FA2 runs one call per range.
    q_ranges must tile q in order."""
    if backend() == 'fa3':
        lq = [b - a for a, b in q_ranges]
        lk = [b - a for a, b in k_ranges]
        off = torch.tensor([a for a, _ in k_ranges] + [k.shape[0]], dtype=torch.int32, device=q.device)
        used = torch.tensor(lk, dtype=torch.int32, device=q.device)
        return _first(_fa3.flash_attn_varlen_func(q, k, v, cu_seqlens(lq, q.device), off, max(lq), max(lk), seqused_k=used))
    return torch.cat([_fa2_func(q[None, a:b], k[None, c:d], v[None, c:d])[0] for (a, b), (c, d) in zip(q_ranges, k_ranges)])
