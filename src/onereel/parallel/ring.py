"""Ring attention over a scene layout: USP's Ring adapted to scene attention (baseline of the communication study).

Every scene's video tokens are split into n near-equal contiguous chunks and rank r holds chunk r of every scene, so
the attention work is balanced whatever the scene lengths. The video rows are reordered so that each rank's rows form
one contiguous range of the permuted sequence (perm: permuted row -> original token; the models shard, apply RoPE and
gather in that order). Per layer the K/V of the local rows travel around the ring in n - 1 point-to-point steps, the
next transfer posted before the visiting block is computed; at every step each local query row attends the rows of the
visiting block that lie in its scene's key set (own frames, halo and memory frames, as in the scene path), and the
partial outputs are merged by their log-sum-exp. Text queries (joint-attention models) read their scene's own frames
and their sentence, so every rank computes them in full. The traffic equals native Ring Attention: every rank sends its
K/V block to every other rank, 2 (n - 1) / n of the video K/V rows per rank and layer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from onereel import attention as A
from onereel.parallel import dist as D


class RingPlan:
    def __init__(self, layout, n: int, T: int, device='cuda'):
        self.layout, self.n, self.T = layout, n, T
        bounds = [np.linspace(lo * T, hi * T, n + 1).round().astype(int) for lo, hi in layout.ranges]
        self.chunks = [[(j, int(b[r]), int(b[r + 1])) for j, b in enumerate(bounds)] for r in range(n)]
        self.perm = torch.cat([torch.arange(lo, hi) for r in range(n) for _, lo, hi in self.chunks[r]]).to(device)
        self.inv = torch.empty_like(self.perm)
        self.inv[self.perm] = torch.arange(len(self.perm), device=device)
        sizes = [sum(hi - lo for _, lo, hi in self.chunks[r]) for r in range(n)]
        c = np.cumsum([0] + sizes)
        self.own = [(int(c[r]), int(c[r + 1])) for r in range(n)]
        self.local = []                          # per rank: (scene, local lo, local hi) of its rows, in scene order
        for r in range(n):
            off, rows = 0, []
            for j, lo, hi in self.chunks[r]:
                rows.append((j, off, off + hi - lo))
                off += hi - lo
            self.local.append(rows)
        self._keys = {}

    def volume(self) -> float:
        """K/V rows each rank sends per layer (Eq. 2 units, D cancels): its block to each of the n - 1 others"""
        return 2 * max(b - a for a, b in self.own) * (self.n - 1)

    def keys(self, src: int, tag) -> dict:
        """scene -> rows of rank src's block in that scene's key set (None when there are none); tag identifies the
        layout's current extra frames (frame memory changes them every step)"""
        key = (src, tag)
        if key not in self._keys:
            self._keys = {k: v for k, v in self._keys.items() if k[1] == tag}
            lay, T = self.layout, self.T
            frame = torch.cat([torch.arange(lo, hi) for _, lo, hi in self.chunks[src]]) // T
            out = {}
            for j in range(lay.n_scenes):
                sel = torch.isin(frame, torch.tensor(lay.key_frames(j))).nonzero().flatten()
                out[j] = sel.to(self.perm.device) if len(sel) else None
            self._keys[key] = out
        return self._keys[key]


def _merge(o, lse, o_s, lse_s):
    """o [N, H, D] fp32 and lse [H, N] updated in place with a partial result (o_s, lse_s)"""
    m = torch.maximum(lse, lse_s)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w1, w2 = torch.exp(lse - m), torch.exp(lse_s - m)
    den = w1 + w2
    o.mul_((w1 / den).transpose(0, 1)[..., None]).add_(o_s.float() * (w2 / den).transpose(0, 1)[..., None])
    lse.copy_(m + torch.log(den))


def attention(plan: RingPlan, q, k, v, eq=None, ek=None, ev=None, text_ranges=None, tag=None):
    """q / k / v [local rows, H, D] in the permuted order; eq / ek / ev [text, H, D] replicated (or None).
    Returns (video out, text out or None)"""
    n, r = D.world(), D.rank()
    H = q.shape[1]
    o = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.full((H, q.shape[0]), float('-inf'), device=q.device)
    if eq is not None:
        ot = torch.zeros(eq.shape, dtype=torch.float32, device=q.device)
        lse_t = torch.full((H, eq.shape[0]), float('-inf'), device=q.device)
    cur, src = torch.stack([k, v]).contiguous(), r
    for s in range(n):
        works = None
        if s < n - 1:
            nxt = (src - 1) % n
            buf = cur.new_empty((2, plan.own[nxt][1] - plan.own[nxt][0]) + tuple(cur.shape[2:]))
            works = dist.batch_isend_irecv([dist.P2POp(dist.isend, cur, (r + 1) % n, group=D.group()),
                                            dist.P2POp(dist.irecv, buf, (r - 1) % n, group=D.group())])
        keys, segs, meta = plan.keys(src, tag), [], []
        for j, lo, hi in plan.local[r]:
            idx = keys[j]
            kj = [cur[0].index_select(0, idx)] if idx is not None else []
            vj = [cur[1].index_select(0, idx)] if idx is not None else []
            if eq is not None and s == 0:
                t0, t1 = text_ranges[j]
                kj, vj = kj + [ek[t0:t1]], vj + [ev[t0:t1]]
            if kj and hi > lo:
                segs.append((q[lo:hi], torch.cat(kj), torch.cat(vj)))
                meta.append(('v', lo, hi))
        if eq is not None:
            for j, lo, hi in plan.local[src]:
                t0, t1 = text_ranges[j]
                kj, vj = [cur[0][lo:hi]], [cur[1][lo:hi]]
                if s == 0:
                    kj, vj = kj + [ek[t0:t1]], vj + [ev[t0:t1]]
                segs.append((eq[t0:t1], torch.cat(kj), torch.cat(vj)))
                meta.append(('t', t0, t1))
        if segs:
            for (kind, lo, hi), (os_, ls_) in zip(meta, A.packed_lse(segs)):
                if kind == 'v':
                    _merge(o[lo:hi], lse[:, lo:hi], os_, ls_)
                else:
                    _merge(ot[lo:hi], lse_t[:, lo:hi], os_, ls_)
        if works is not None:
            for w in works:
                w.wait()
            cur, src = buf, nxt
    return o.to(q.dtype), (ot.to(eq.dtype) if eq is not None else None)
