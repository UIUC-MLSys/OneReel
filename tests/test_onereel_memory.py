"""Bounded cross-scene memory: packed head-major attention against a per-head masked reference (GPU)."""
import pytest
import torch

from onereel.layout import SceneLayout
from onereel.memory import BoundaryMemory, MemoryAttention, candidates, frame_plan

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def run_full(mem, key, lay, T, q, k, v, eq, ek, ev, tr):
    """every token on one device: video pieces per scene, every sentence queried"""
    S = q.shape[0]
    plan = frame_plan(lay, T, torch.arange(S), S, [(j, lo * T, hi * T) for j, (lo, hi) in enumerate(lay.ranges)],
                      tr, set(range(lay.n_scenes)), S, q.device)
    out = mem(key, torch.cat([q, eq]), torch.cat([k, ek]), torch.cat([v, ev]), plan)
    return out[:S], out[S:]


def reference(q, k, v, eq, ek, ev, lay, T, rep_row, rep_scene, valid, text_ranges):
    """per head and scene, softmax over (local video + sentence + foreign representatives) in float32"""
    S, H, D = q.shape
    ov, ot = torch.zeros(S, H, D, device=q.device), torch.zeros(ek.shape[0], H, D, device=q.device)
    kk, vv = torch.cat([k, ek]).float(), torch.cat([v, ev]).float()
    for h in range(H):
        for j, (lo, hi) in enumerate(lay.ranges):
            t0, t1 = text_ranges[j]
            fr = [r for r, s, ok in zip(rep_row[h].tolist(), rep_scene[h].tolist(), valid[h].tolist()) if ok and s != j]
            keys = list(range(lo * T, hi * T)) + [S + t for t in range(t0, t1)] + fr
            a = torch.softmax(q[lo * T:hi * T, h].float() @ kk[keys, h].T / D ** 0.5, -1)
            ov[lo * T:hi * T, h] = a @ vv[keys, h]
            tkeys = list(range(lo * T, hi * T)) + [S + t for t in range(t0, t1)]
            a = torch.softmax(eq[t0:t1, h].float() @ kk[tkeys, h].T / D ** 0.5, -1)
            ot[t0:t1, h] = a @ vv[tkeys, h]
    return ov, ot


@cuda
@pytest.mark.parametrize('lengths,M', [([3, 5, 2, 4], 8), ([2, 2, 2], 1000), ([6, 1, 3], 5)])
def test_memory_attention_matches_reference(lengths, M):
    torch.manual_seed(0)
    T, H, D = 16, 4, 64
    lay = SceneLayout(lengths, halo=0)
    S = lay.n_frames * T
    counts = [5, 7, 3, 6][:lay.n_scenes]
    tr, c = [], 0
    for n in counts:
        tr.append((c, c + n))
        c += n
    q, k, v = (torch.randn(S, H, D, device='cuda', dtype=torch.bfloat16) for _ in range(3))
    eq, ek, ev = (torch.randn(c, H, D, device='cuda', dtype=torch.bfloat16) for _ in range(3))
    ov, ot = run_full(MemoryAttention(BoundaryMemory(M)), ('l0', True), lay, T, q, k, v, eq, ek, ev, tr)
    # a fresh memory makes the same (deterministic) first selection as the call above
    rows, owner = candidates(lay, T, 'cuda')
    mem2 = BoundaryMemory(M)
    rep, valid = mem2.select(('l0', True), k[rows])
    ref_v, ref_t = reference(q, k, v, eq, ek, ev, lay, T, rows[rep], owner[rep], valid, tr)
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
    assert rel(ov, ref_v) < 2e-2
    assert rel(ot, ref_t) < 2e-2


@cuda
def test_all_candidates_kept_when_budget_covers_them():
    torch.manual_seed(1)
    lay = SceneLayout([3, 4, 2], halo=0)
    T, H, D = 8, 2, 32
    rows, owner = candidates(lay, T, 'cuda')
    k = torch.randn(lay.n_frames * T, H, D, device='cuda')
    rep, valid = BoundaryMemory(len(rows)).select('x', k[rows])
    for h in range(H):
        assert sorted(rep[h][valid[h]].tolist()) == list(range(len(rows)))


@cuda
def test_representative_is_nearest_member():
    torch.manual_seed(2)
    x = torch.randn(300, 3, 16, device='cuda')
    mem = BoundaryMemory(12)
    rep, valid = mem.select('y', x)
    c = mem.centroids['y']
    xn = torch.nn.functional.normalize(x.float().transpose(0, 1), dim=-1)
    for h in range(3):
        z = torch.cdist(xn[h], c[h]).argmin(-1)
        for r in rep[h][valid[h]].tolist():
            members = (z == z[r]).nonzero().flatten()
            d = (xn[h, members] - c[h, z[r]]).pow(2).sum(-1)
            assert d.min() >= (xn[h, r] - c[h, z[r]]).pow(2).sum() - 1e-6
