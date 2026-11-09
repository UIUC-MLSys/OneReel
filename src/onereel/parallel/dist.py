"""Sequence-parallel process group: xfuser's (one SP group over all ranks) when xfuser is installed, else the default
torch.distributed group (the MiniMax-H3 environment). World 1 needs no process group."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

_ON = False
_XF = False


def init() -> int:
    """initialise the distributed environment from torchrun's variables; returns the world size"""
    global _ON, _XF
    n = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    if n == 1 or _ON:
        return n
    try:
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    except ImportError:
        dist.init_process_group('nccl', device_id=torch.device('cuda', torch.cuda.current_device()))
    else:
        init_distributed_environment(rank=int(os.environ['RANK']), world_size=n)
        initialize_model_parallel(sequence_parallel_degree=n, ulysses_degree=n)
        _XF = True
    _ON = True
    return n


def world() -> int:
    if not _ON:
        return 1
    if not _XF:
        return dist.get_world_size()
    from xfuser.core.distributed import get_sequence_parallel_world_size
    return get_sequence_parallel_world_size()


def rank() -> int:
    if not _ON:
        return 0
    if not _XF:
        return dist.get_rank()
    from xfuser.core.distributed import get_sequence_parallel_rank
    return get_sequence_parallel_rank()


def group():
    """torch process group of the sequence-parallel ranks"""
    if not _XF:
        return dist.group.WORLD
    from xfuser.core.distributed import get_sp_group
    return get_sp_group().device_group


def peer_rank(r: int) -> int:
    """global rank of sequence-parallel rank r (for point-to-point ops)"""
    if not _XF:
        return r
    from xfuser.core.distributed import get_sp_group
    return get_sp_group().ranks[r]


def all_reduce(x: torch.Tensor):
    if world() > 1:
        dist.all_reduce(x, group=group())


def all_gather_object(obj):
    if world() == 1:
        return [obj]
    out = [None] * world()
    dist.all_gather_object(out, obj, group=group())
    return out


def barrier():
    if world() > 1:
        dist.barrier(group=group())
