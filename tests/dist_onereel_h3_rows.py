"""MiniMax-H3 packed-row parallelism against one process (run on P GPUs, H3 environment):
  torchrun --nproc_per_node 2 tests/dist_onereel_h3_rows.py
A shuffled packed [text | audio | video] sequence; every rank runs the H3 attention processor on its rows (gather or
Ulysses path, halo or bounded memory) and the gathered result must match the single-process computation."""
import torch

from onereel import attention as A
from onereel.memory import BoundaryMemory, MemoryAttention, make_plan
from onereel.models import minimax_h3 as M
from onereel.parallel import dist as D

D.init()
r, n = D.rank(), D.world()
H, Dh, per, F = 4, 64, 6, 14
C = H * Dh
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()


class Attn(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        lin = lambda: torch.nn.Linear(C, C, bias=False, dtype=torch.bfloat16)
        self.to_q, self.to_k, self.to_v = lin(), lin(), lin()
        self.to_out = torch.nn.ModuleList([lin(), torch.nn.Identity()])
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g).to(p.dtype) / C ** 0.5
        self.norm_q = self.norm_k = torch.nn.Identity()
        self.fused_projections, self.heads = False, H


for memory in (0, 8):
    for mode in ('scene', 'ulysses'):
        g = torch.Generator().manual_seed(7)
        counts = [3, 4, 5, 7]                    # odd N: the Ulysses split pads
        ctx = M.Context(halo=0 if memory else 2, lengths=[3, 5, 2, 4], memory=memory, mode=mode)
        ctx.counts = counts
        Nt, Na, Nv = sum(counts), 2 * 2 * F, per * F
        N = Nt + Na + Nv
        perm = torch.randperm(N, generator=g).cuda()
        text, audio, video = perm[:Nt], perm[Nt:Nt + Na], perm[Nt + Na:]
        pl = M.build_plan(ctx, video, audio, text, F)
        attn = Attn(g).cuda()
        x = torch.randn(1, N, C, generator=g).to('cuda', torch.bfloat16)
        q, k, v = (f(x)[0].unflatten(-1, (H, -1)) for f in (attn.to_q, attn.to_k, attn.to_v))
        if memory:
            plan = make_plan([(j, a, b, True) for j, (a, b) in enumerate(zip(pl['scene_q'], pl['scene_k']))], pl['cand'],
                             pl['cand_owner'], 'cuda')
            o = MemoryAttention(BoundaryMemory(memory))(0, q, k, v, plan)
        else:
            o = torch.zeros_like(q)
            o[pl['q']] = A.varlen(q[pl['q']], k[pl['k']], v[pl['k']], pl['cu_q'], pl['cu_k'], pl['max_q'], pl['max_k'])
        ref = attn.to_out[0](o.flatten(1, 2)[None])

        ctx.plan = pl
        ctx.par = M._parallel(ctx, pl, N, H)
        if memory:
            ctx._mem = MemoryAttention(BoundaryMemory(memory))
        out = M.Processor(ctx, 0)(attn, x[:, ctx.par['local']])
        mine = ctx.par['rows'][r]
        e = rel(out[0, :len(mine)], ref[0, mine])
        assert ctx.run_mode == mode and e < 1e-2, (memory, mode, ctx.run_mode, r, e)
        if r == 0:
            print(f'PASS memory={memory} mode={mode} rows/rank={[len(x) for x in ctx.par["rows"]]} '
                  f'V_G={ctx.path["V_G"]} V_U={ctx.path["V_U"]:.0f} rel {e:.1e}', flush=True)
D.barrier()
