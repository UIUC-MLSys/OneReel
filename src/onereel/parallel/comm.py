"""Partitioning, path choice (Eq. 2) and the collectives of the two parallel paths (K/V gathering, Ulysses)."""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from onereel.layout import SceneLayout
from onereel.parallel import dist as D

# per-token cost u = C_LIN + C_ATT * EFF_SCENE * L' (ms per token and forward, L' = keys per query), fitted on
# HunyuanVideo-1.5 720p, FlashAttention-3, frame memory M = 4, one H200: profiled attention vs query-key pairs and
# token-wise kernels vs tokens over 1-12 scenes of 84 latent frames. The
# earlier 480p / FA2 fit (C_LIN 0.0179, C_ATT 1.30e-6, EFF_SCENE 1.063) weighted attention about 2.9x too high.
C_LIN, C_ATT, EFF_SCENE = 0.02581842, 6.753627e-07, 1.0


def query_cost(layout: SceneLayout, T: int, text: list[int] | None = None, extra_keys: int = 0) -> list[float]:
    """per scene, the cost of one video query under the fitted model: C_LIN + C_ATT * EFF_SCENE * |S_j|, where |S_j| =
    scene + halo video keys, the sentence's text tokens and `extra_keys` (the memory budget M)"""
    return [C_LIN + C_ATT * EFF_SCENE * (len(layout.key_frames(j)) * T + (text[j] if text else 0) + extra_keys)
            for j in range(layout.n_scenes)]


def balanced(sizes: list[int], cost: list[float], n: int) -> list[tuple[int, int]]:
    """paper Algorithm 1: scenes in order with sizes[j] queries of per-query cost cost[j]; walk them accumulating work
    and place boundary p at the query where the work reaches p * W / n (any query, so a scene is cut only where a
    boundary falls inside it: at most n - 1 cuts). Returns per rank (query lo, query hi); every rank holds a query."""
    L = sum(sizes)
    assert L >= n, f'{L} queries for {n} ranks'
    a = np.cumsum([0] + list(sizes))
    W = np.cumsum([0.0] + [m * c for m, c in zip(sizes, cost)])
    target = W[-1] / n
    b = [0]
    for p in range(1, n):
        i = int(np.searchsorted(W, p * target - 1e-12 * W[-1], side='left'))     # min{j : W_j >= p W / n}
        r = int(round((p * target - W[i - 1]) / cost[i - 1]))
        b.append(min(max(int(a[i - 1]) + r, b[-1] + 1), L - (n - p)))
    b.append(L)
    return [(b[p], b[p + 1]) for p in range(n)]


def partition(layout: SceneLayout, n: int, T: int, mode: str, text: list[int] | None = None,
              extra_keys: int = 0, granularity: str = 'balanced') -> list[tuple[int, int]]:
    """per rank (video token lo, hi), tokens frame-major with T per frame. ulysses: equal split. scene (K/V
    gathering): work-balanced (Algorithm 1) at token granularity; ablations of the K/V gathering partition:
    granularity 'tokens' splits the tokens equally (ignoring the per-query cost), 'scenes' places the balanced
    boundaries at the nearest scene boundary (no scene is cut)"""
    L = layout.n_frames * T
    if mode == 'ulysses' or n == 1:
        assert L % n == 0, f'{L} video tokens do not split evenly over {n} ranks'
        return [(r * L // n, (r + 1) * L // n) for r in range(n)]
    if granularity == 'tokens':
        return [(r * L // n, (r + 1) * L // n) for r in range(n)]
    own = balanced([m * T for m in layout.lengths], query_cost(layout, T, text, extra_keys), n)
    if granularity == 'scenes':
        starts = [int(x) * T for x in np.cumsum([0] + layout.lengths)]
        assert layout.n_scenes >= n, f'{layout.n_scenes} scenes for {n} ranks'
        b = [0]
        for p in range(1, n):
            ok = [s for s in starts[1:-1] if s > b[-1] and starts.index(s) <= layout.n_scenes - (n - p)]
            b.append(min(ok, key=lambda s: abs(s - own[p][0])))
        b.append(L)
        return [(b[p], b[p + 1]) for p in range(n)]
    assert granularity == 'balanced', granularity
    return own


def merge(iv) -> list[tuple[int, int]]:
    """union of half-open intervals, sorted and merged"""
    out = []
    for lo, hi in sorted(x for x in iv if x[0] < x[1]):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(x) for x in out]


def clip(iv, a: int, b: int) -> list[tuple[int, int]]:
    return [(max(lo, a), min(hi, b)) for lo, hi in iv if lo < b and hi > a]


def frame_tokens(frames, T: int) -> list[tuple[int, int]]:
    return [(lo * T, hi * T) for lo, hi in runs(frames)]


def needed(layout: SceneLayout, T: int, own: tuple[int, int], extra=()) -> list[tuple[int, int]]:
    """video token intervals outside `own` that the scenes intersecting `own` read (their tokens and shared frames),
    plus the `extra` frames every rank reads (the memory's boundary candidates)"""
    a, b = own
    iv = frame_tokens(sorted(extra), T)
    for j, (lo, hi) in enumerate(layout.ranges):
        if lo * T < b and hi * T > a:
            iv += [(lo * T, hi * T)] + frame_tokens(layout.shared_frames(j), T)
    return [y for lo, hi in merge(iv) for y in ((lo, min(hi, a)), (max(lo, b), hi)) if y[0] < y[1]]


def gather_rows(layout: SceneLayout, own: list[tuple[int, int]], T: int, extra=()) -> int:
    """V_G / (2D) of Eq. 2 for the complete scopes: the most K/V rows (per tensor) any rank gathers"""
    return max(sum(hi - lo for lo, hi in needed(layout, T, o, extra)) for o in own)


def eq2(remote_rows: int, L: int, n: int, theta: float) -> dict:
    """Eq. 2: gather (scene) when V_G < theta * V_U, V_G = 2D max_p (remote rows of p), V_U = 4LD(P-1)/P^2 (Ulysses
    all-to-all of Q, K, V and the output); D cancels"""
    vg, vu = 2 * remote_rows, 4 * L * (n - 1) / n ** 2
    return dict(path='scene' if vg < theta * vu else 'ulysses', V_G=vg, V_U=vu, theta=theta)


def choose_path(layout: SceneLayout, own: list[tuple[int, int]], T: int, theta: float, extra=()) -> dict:
    return eq2(gather_rows(layout, own, T, extra), layout.n_frames * T, len(own), theta)


def runs(frames) -> list[tuple[int, int]]:
    """frame list (in order) -> contiguous runs"""
    out = []
    for f in frames:
        if out and out[-1][1] == f:
            out[-1][1] = f + 1
        else:
            out.append([f, f + 1])
    return [tuple(r) for r in out]


class HaloExchange:
    """posts the K/V exchange of one layer over video token ranges: rank r sends every peer the tokens of its range
    that the peer reads and receives the tokens it reads from their owners; rows() returns the rows of a frame list
    from local or received buffers, buffer() one [rows, H, D] buffer of everything held. heads: exchange only these
    heads (a head chunk of a pipelined exchange; the sent rows are copied to contiguous buffers)"""

    def __init__(self, layout: SceneLayout, own: list[tuple[int, int]], T: int, k: torch.Tensor, v: torch.Tensor,
                 extra=(), heads: slice | None = None):
        if heads is not None:
            k, v = k[:, heads], v[:, heads]
        self.T, self.k, self.v, self.L = T, k, v, layout.n_frames * T
        r, n = D.rank(), D.world()
        self.a, self.b = own[r]
        self.remote, self.works, self.keep = {}, [], []
        if n == 1:
            return
        need = [needed(layout, T, o, extra) for o in own]
        ops = []
        for p in range(n):                      # per peer: K then V of each interval, in order on both sides
            if p == r:
                continue
            peer = D.peer_rank(p)
            for lo, hi in clip(need[p], self.a, self.b):
                for x in (k, v):
                    buf = x[lo - self.a:hi - self.a]
                    if not buf.is_contiguous():
                        buf = buf.contiguous()
                        self.keep.append(buf)
                    ops.append(dist.P2POp(dist.isend, buf, peer, group=D.group()))
            for lo, hi in clip(need[r], *own[p]):
                bk, bv = (torch.empty(hi - lo, *k.shape[1:], dtype=k.dtype, device=k.device) for _ in range(2))
                self.remote[(lo, hi)] = (bk, bv)
                ops += [dist.P2POp(dist.irecv, bk, peer, group=D.group()), dist.P2POp(dist.irecv, bv, peer, group=D.group())]
        if ops:
            self.works = dist.batch_isend_irecv(ops)

    def buffer(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(k, v, token_row): local rows, then the received intervals; token_row [L] (CPU) buffer row of each held video
        token, -1 for tokens not held"""
        token_row = torch.full((self.L,), -1, dtype=torch.long)
        token_row[self.a:self.b] = torch.arange(self.b - self.a)
        ks, vs, off = [self.k], [self.v], self.b - self.a
        for (lo, hi), (bk, bv) in self.remote.items():
            token_row[lo:hi] = off + torch.arange(hi - lo)
            ks.append(bk)
            vs.append(bv)
            off += hi - lo
        return torch.cat(ks), torch.cat(vs), token_row

    def wait(self):
        for w in self.works:
            w.wait()
        self.works, self.keep = [], []

    def rows(self, i: int, side: str | None = None):
        """i = 0 for K, 1 for V: callable frame list -> rows (concatenated in the list's order); side 'local' / 'remote':
        only the rows held locally / received (possibly none)"""
        a, b = self.a, self.b
        local = (self.k, self.v)[i]

        def get(frames):
            parts = []
            for lo, hi in frame_tokens(frames, self.T):
                while lo < hi:
                    if a <= lo < b:
                        e = min(hi, b)
                        if side != 'remote':
                            parts.append(local[lo - a:e - a])
                    else:
                        (rlo, rhi), buf = next(((x, y), bb) for (x, y), bb in self.remote.items() if x <= lo < y)
                        e = min(hi, rhi)
                        if side != 'local':
                            parts.append(buf[i][lo - rlo:e - rlo])
                    lo = e
            return torch.cat(parts) if parts else local[:0]
        return get


def send_text_rows(x: torch.Tensor, layout: SceneLayout, own: list[tuple[int, int]], T: int, text_ranges) -> None:
    """K/V gathering: the text queries of scene j run on the rank holding its first frame; x [text rows, H, D] holds
    their outputs there. Every other rank whose video range intersects scene j reads scene j's text in the next layer,
    so it receives those rows in place. Text rows of scenes a rank does not touch are never read there and keep whatever
    they hold (the all-reduce this replaces moved every text row to every rank)"""
    r, n = D.rank(), D.world()
    touches = lambda p, lo, hi: lo * T < own[p][1] and hi * T > own[p][0]
    ops = []
    for (lo, hi), (t0, t1) in zip(layout.ranges, text_ranges):
        if t0 == t1:
            continue
        src = next(p for p in range(n) if own[p][0] <= lo * T < own[p][1])
        if src == r:
            ops += [dist.P2POp(dist.isend, x[t0:t1], D.peer_rank(p), group=D.group())
                    for p in range(n) if p != r and touches(p, lo, hi)]
        elif touches(r, lo, hi):
            ops.append(dist.P2POp(dist.irecv, x[t0:t1], D.peer_rank(src), group=D.group()))
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()


def row_plan(owner: torch.Tensor, need: list[torch.Tensor], r: int, device) -> dict:
    """row-granular exchange (MiniMax-H3's packed rows). owner [N] (CPU) rank of each row; need[p] sorted rows rank p
    reads but does not own. Returns local (rows held by r, ascending), bufpos [N] (row -> position in r's K/V buffer:
    local rows, then the rows received from each peer in ascending order; -1 if absent) and per peer (send positions,
    rows received)."""
    N, n = owner.numel(), len(need)
    local = (owner == r).nonzero()[:, 0]
    pos = torch.full((N,), -1, dtype=torch.long)
    pos[local] = torch.arange(len(local))
    bufpos, off, peers = pos.clone(), len(local), {}
    for p in range(n):
        if p == r:
            continue
        send = need[p][owner[need[p]] == r]
        recv = need[r][owner[need[r]] == p]
        bufpos[recv] = off + torch.arange(len(recv))
        off += len(recv)
        if len(send) or len(recv):
            peers[p] = (pos[send].to(device), len(recv))
    return dict(local=local, bufpos=bufpos, peers=peers)


class RowExchange:
    """posts the row exchange of one layer (plan from row_plan); buffer() = local rows, then each peer's rows"""

    def __init__(self, plan: dict, k: torch.Tensor, v: torch.Tensor):
        self.k, self.v, self.recv, self.works, ops, self.keep = k, v, [], [], [], []
        for p, (send, n_recv) in plan['peers'].items():
            peer = D.peer_rank(p)
            if len(send):
                for x in (k, v):
                    buf = x.index_select(0, send)
                    self.keep.append(buf)
                    ops.append(dist.P2POp(dist.isend, buf, peer, group=D.group()))
            if n_recv:
                bk, bv = (torch.empty(n_recv, *k.shape[1:], dtype=k.dtype, device=k.device) for _ in range(2))
                self.recv.append((bk, bv))
                ops += [dist.P2POp(dist.irecv, bk, peer, group=D.group()), dist.P2POp(dist.irecv, bv, peer, group=D.group())]
        if ops:
            self.works = dist.batch_isend_irecv(ops)

    def wait(self):
        for w in self.works:
            w.wait()
        self.works, self.keep = [], []

    def buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat([self.k] + [b[0] for b in self.recv]), torch.cat([self.v] + [b[1] for b in self.recv])


def a2a_heads(x: torch.Tensor) -> torch.Tensor:
    """[S/N, H, D] (local tokens) -> [S, H/N, D] (all tokens, local heads)"""
    n = D.world()
    if n == 1:
        return x
    x = x.unflatten(1, (n, x.shape[1] // n)).transpose(0, 1).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=D.group())
    return y.flatten(0, 1)


def a2a_tokens(x: torch.Tensor) -> torch.Tensor:
    """[S, H/N, D] -> [S/N, H, D]"""
    n = D.world()
    if n == 1:
        return x
    x = x.unflatten(0, (n, -1)).contiguous()
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=D.group())
    return y.transpose(0, 1).flatten(1, 2)


def gather_tokens(x: torch.Tensor, own: list[tuple[int, int]]) -> torch.Tensor:
    """[B, local tokens, C] -> [B, all tokens, C] for per-rank token ranges of unequal length"""
    n = D.world()
    if n == 1:
        return x
    m = max(b - a for a, b in own)
    buf = x.new_zeros(x.shape[0], m, x.shape[2])
    buf[:, :x.shape[1]] = x
    allb = [torch.empty_like(buf) for _ in range(n)]
    dist.all_gather(allb, buf, group=D.group())
    return torch.cat([y[:, :b - a] for y, (a, b) in zip(allb, own)], 1)
