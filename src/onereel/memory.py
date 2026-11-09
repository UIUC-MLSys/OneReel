"""Bounded cross-scene memory (paper §3.2, Appendix E).

Candidates are the first and last latent frames of every scene (a frame shared by both counts once); text and audio are
never candidates. Per layer, CFG branch and head, the L2-normalised, position-encoded candidate keys are clustered into
at most M groups (k-means: up to ITERS_INIT iterations the first time, ITERS_UPDATE warm-started iterations at later
steps from the previous centroids). Each non-empty cluster keeps the original candidate nearest its centroid (ties: the
lowest token index); its key / value come from the current layer and step. A scene's queries then read their complete
local context plus the representatives of the other scenes (representatives of their own scene are already local), all
in one softmax.

Representatives differ per head, so attention is packed head-major: every (head, query piece) pair is one
variable-length sequence of a single head, and the whole layer is one varlen call. Queries and keys / values are
gathered from [rows, heads] buffers (row * H + head). A piece names its query rows, its local key rows (rows of the
key / value buffer) and whether it reads the memory; `frame_plan` builds the pieces of the frame-major models (Hunyuan,
Wan), MiniMax-H3 builds its own over the packed [text | audio | video] rows. On the gather path every rank receives
all boundary candidates and clusters them itself; the clustering is deterministic, so the ranks agree on the
representatives without exchanging them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from onereel import attention as A
from onereel.layout import SceneLayout

ITERS_INIT, ITERS_UPDATE = 50, 2


def candidate_frames(layout: SceneLayout) -> tuple[list[int], list[int]]:
    """boundary frames (first and last frame of every scene, each frame once) and their scene"""
    frames, owner = [], []
    for j, (lo, hi) in enumerate(layout.ranges):
        for f in dict.fromkeys((lo, hi - 1)):
            frames.append(f)
            owner.append(j)
    return frames, owner


def candidates(layout: SceneLayout, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """video rows (frame-major, all tokens) of the boundary frames and their scene"""
    frames, owner = candidate_frames(layout)
    rows = torch.cat([torch.arange(f * T, (f + 1) * T, device=device) for f in frames])
    scene = torch.tensor(owner, device=device).repeat_interleave(T)
    return rows, scene


class BoundaryMemory:
    def __init__(self, M: int = 128):
        self.M = M
        self.centroids = {}                 # (layer, branch) -> [H, M, D] float32

    def reset(self):
        self.centroids = {}

    @torch.no_grad()
    def select(self, key, kc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """kc [B, H, D] candidate keys -> (rep [H, M] candidate positions, valid [H, M]); one representative per non-empty
        cluster, padded with valid = False"""
        x = F.normalize(kc.float().transpose(0, 1), dim=-1)                       # [H, B, D]
        H, B, D = x.shape
        M = min(self.M, B)
        c = self.centroids.get(key)
        if c is None or c.shape != (H, M, D):
            c = x[:, torch.linspace(0, B - 1, M, device=x.device).round().long()].clone()
            iters = ITERS_INIT
        else:
            iters = ITERS_UPDATE
        xx = (x * x).sum(-1, keepdim=True)                                          # [H, B, 1]
        for it in range(iters + 1):
            d = xx - 2 * x @ c.transpose(1, 2) + (c * c).sum(-1)[:, None]            # [H, B, M] squared distances
            z = d.argmin(-1)                                                         # [H, B]
            if it == iters:
                break
            oh = F.one_hot(z, M).float()                                             # matmul sums: deterministic, so
            s, n = oh.transpose(1, 2) @ x, oh.sum(1)                                 # every rank picks the same set
            prev = c
            c = torch.where((n > 0)[..., None], s / n.clamp_min(1)[..., None], c)   # an empty cluster keeps its centroid
            if torch.equal(c, prev):
                break
        self.centroids[key] = c
        dist = d.gather(-1, z[..., None])[..., 0]                                    # [H, B] distance to own centroid
        # per (head, cluster): the member with the smallest distance, then the lowest index (stable sorts)
        o1 = dist.argsort(dim=-1, stable=True)
        o2 = z.gather(-1, o1).argsort(dim=-1, stable=True)
        order = o1.gather(-1, o2)                                                    # sorted by (cluster, dist, index)
        zs = z.gather(-1, order)
        first = torch.ones_like(zs, dtype=torch.bool)
        first[:, 1:] = zs[:, 1:] != zs[:, :-1]
        rank = first.cumsum(-1) - 1
        rep = torch.zeros(H, M, dtype=torch.long, device=x.device)
        valid = torch.zeros(H, M, dtype=torch.bool, device=x.device)
        hh = torch.arange(H, device=x.device)[:, None].expand_as(order)
        rep[hh[first], rank[first]] = order[first]
        valid[hh[first], rank[first]] = True
        return rep, valid


@dataclass
class Plan:
    qrows: torch.Tensor                  # [sum Q] query rows, piece after piece
    qlen: torch.Tensor                   # [n] queries per piece
    pad: torch.Tensor                    # [n, Lmax] local key rows per piece (padded)
    pmask: torch.Tensor                  # [n, Lmax]
    lens: torch.Tensor                   # [n] local keys per piece
    scene: torch.Tensor                  # [n] scene of each piece
    mem: torch.Tensor                    # [n] the piece reads the memory
    cand: torch.Tensor                   # [B] key-buffer rows of the boundary candidates
    owner: torch.Tensor                  # [B] their scene


def make_plan(pieces, cand, owner, device) -> Plan:
    """pieces: [(scene, query rows, local key rows, reads memory)]; cand / owner: candidate key rows and their scene"""
    t = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
    pieces = [(j, t(q), t(k), m) for j, q, k, m in pieces if len(q)]
    lens = [len(k) for _, _, k, _ in pieces]
    pad = torch.zeros(len(pieces), max(lens), dtype=torch.long, device=device)
    pmask = torch.zeros_like(pad, dtype=torch.bool)
    for i, (_, _, k, _) in enumerate(pieces):
        pad[i, :len(k)] = k
        pmask[i, :len(k)] = True
    return Plan(qrows=torch.cat([q for _, q, _, _ in pieces]), qlen=t([len(q) for _, q, _, _ in pieces]), pad=pad,
                pmask=pmask, lens=t(lens), scene=t([j for j, _, _, _ in pieces]),
                mem=torch.tensor([m for *_, m in pieces], dtype=torch.bool, device=device), cand=t(cand), owner=t(owner))


def frame_plan(layout: SceneLayout, T: int, token_row, Lv: int, q_pieces, text_ranges=None, text_for=(), Sq: int = 0,
               device='cuda') -> Plan:
    """frame-major models. Key buffer: Lv video rows (video token t at token_row[t]; a rank holds any token range, so a
    frame may be split), then the text tokens; query buffer: Sq video rows, then the text tokens. q_pieces: [(scene,
    video query row lo, hi)]. Video queries read their scene, their sentence and the memory; sentence queries (scenes in
    text_for) read their scene and sentence."""
    token_row = torch.as_tensor(token_row)
    ar = lambda a, b: torch.arange(a, b)
    frows = lambda fs: token_row[torch.cat([ar(f * T, (f + 1) * T) for f in fs])]
    text = lambda j: Lv + ar(*text_ranges[j]) if text_ranges else ar(0, 0)
    pieces = [(j, ar(lo, hi), torch.cat([frows(layout.frames[j]), text(j)]), True) for j, lo, hi in q_pieces]
    pieces += [(j, Sq + ar(*text_ranges[j]), torch.cat([frows(layout.frames[j]), text(j)]), False) for j in sorted(text_for)]
    cf, owner = candidate_frames(layout)
    return make_plan(pieces, frows(cf), torch.tensor(owner).repeat_interleave(T), device)


class MemoryAttention:
    """packed head-major attention with the bounded memory; plans are cached by the caller's key"""

    def __init__(self, memory: BoundaryMemory):
        self.memory = memory
        self.plans = {}

    def __call__(self, key, q, k, v, p: Plan) -> torch.Tensor:
        """q [Sq, H, D], k / v [L, H, D] buffers -> [Sq, H, D] (rows outside the plan's queries are zero)"""
        Sq, H, D = q.shape
        L = k.shape[0]
        dev = q.device
        rep, valid = self.memory.select(key, k[p.cand])                              # [H, M]
        rep_row, rep_scene = p.cand[rep], p.owner[rep]
        n = len(p.qlen)
        hh = torch.arange(H, device=dev)[:, None, None]
        foreign = valid[:, None, :] & (rep_scene[:, None, :] != p.scene[None, :, None]) & p.mem[None, :, None]   # [H, n, M]
        idx = torch.cat([p.pad[None].expand(H, -1, -1), rep_row[:, None, :].expand(-1, n, -1)], -1) * H + hh
        kidx = idx[torch.cat([p.pmask[None].expand(H, -1, -1), foreign], -1)]
        qidx = (p.qrows[None] * H + hh[:, 0]).flatten()                             # head-major, piece after piece
        lq = p.qlen.repeat(H)
        lk = (p.lens[None] + foreign.sum(-1)).flatten()
        cu_q = F.pad(lq.cumsum(0), (1, 0)).int()
        cu_k = F.pad(lk.cumsum(0), (1, 0)).int()
        qf = q.reshape(Sq * H, 1, D)
        o = A.varlen(qf[qidx], k.reshape(L * H, 1, D)[kidx], v.reshape(L * H, 1, D)[kidx], cu_q, cu_k,
                     int(lq.max()), int(lk.max()))
        out = q.new_zeros(Sq * H, 1, D)
        out[qidx] = o
        return out.view(Sq, H, D)
