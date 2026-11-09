"""K/V gathering over the work-balanced partition against one device (run on P GPUs):
  torchrun --nproc_per_node 2 tests/dist_onereel_gather.py
Every rank builds the same full tensors, keeps its partition's token range (boundaries may fall inside a frame),
gathers what its scenes read and checks the result against the single-device computation."""
import torch

from onereel.layout import SceneLayout
from onereel.memory import BoundaryMemory, MemoryAttention, candidate_frames, frame_plan
from onereel.parallel import comm, dist as D

D.init()
r, n = D.rank(), D.world()
T, H, Dh = 16, 4, 64
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
mid = 0
for lengths, M in (([3, 5, 2, 4, 6], 8), ([10, 1, 1, 2], 5), ([2, 2, 2, 2, 2, 2], 1000), ([5, 2, 7], 8), ([1, 6, 2], 3),
                   ([3, 4, 4, 1], 6)):
    g = torch.Generator().manual_seed(len(lengths))
    rnd = lambda *s: torch.randn(*s, generator=g).to('cuda', torch.bfloat16)
    lay = SceneLayout(lengths, halo=0)
    S, everyone = lay.n_frames * T, set(range(lay.n_scenes))
    counts = [3 + j for j in range(lay.n_scenes)]
    tr = [(sum(counts[:j]), sum(counts[:j + 1])) for j in range(lay.n_scenes)]
    q, k, v = rnd(S, H, Dh), rnd(S, H, Dh), rnd(S, H, Dh)
    et = tuple(rnd(sum(counts), H, Dh) for _ in range(3))
    plan = frame_plan(lay, T, torch.arange(S), S, [(j, lo * T, hi * T) for j, (lo, hi) in enumerate(lay.ranges)],
                      tr, everyone, S)
    ref = MemoryAttention(BoundaryMemory(M))('l', torch.cat([q, et[0]]), torch.cat([k, et[1]]), torch.cat([v, et[2]]), plan)
    ref_v, ref_t = ref[:S], ref[S:]

    own = comm.partition(lay, n, T, 'scene', counts, M)
    a, b = own[r]
    ex = comm.HaloExchange(lay, own, T, k[a:b].contiguous(), v[a:b].contiguous(), extra=candidate_frames(lay)[0])
    ex.wait()
    kb, vb, token_row = ex.buffer()
    pieces = [(j, max(lo * T, a) - a, min(hi * T, b) - a) for j, (lo, hi) in enumerate(lay.ranges) if lo * T < b and hi * T > a]
    text_for = {j for j, (lo, _) in enumerate(lay.ranges) if a <= lo * T < b}
    Sq = b - a
    plan = frame_plan(lay, T, token_row, kb.shape[0], pieces, tr, text_for, Sq)
    out = MemoryAttention(BoundaryMemory(M))('l', torch.cat([q[a:b], et[0]]), torch.cat([kb, et[1]]),
                                             torch.cat([vb, et[2]]), plan)
    ov, ot = out[:Sq], out[Sq:].contiguous()
    D.all_reduce(ot)
    ev_, et_ = rel(ov, ref_v[a:b]), rel(ot, ref_t)
    assert ev_ < 1e-2 and et_ < 1e-2, (lengths, r, ev_, et_)

    lay2 = SceneLayout(lengths, halo=2)
    lay2.extra = {j: [(7 * j + 3) % lay2.n_frames, lay2.n_frames - 1] for j in range(lay2.n_scenes)}   # frame memory
    own2 = comm.partition(lay2, n, T, 'scene', counts)
    a, b = own2[r]
    ex = comm.HaloExchange(lay2, own2, T, k[a:b].contiguous(), v[a:b].contiguous())
    ex.wait()
    for j, (lo, hi) in enumerate(lay2.ranges):
        if lo * T < b and hi * T > a:
            want = torch.cat([k[f * T:(f + 1) * T] for f in lay2.key_frames(j)])
            assert torch.equal(ex.rows(0)(lay2.key_frames(j)), want), (lengths, r, j)
    mid += any(x % T for x, _ in own[1:])
    if r == 0:
        print(f'PASS {lengths} M={M} own={own} (mid-frame: {[x % T != 0 for x, _ in own[1:]]}) memory rel {ev_:.1e}/{et_:.1e}',
              flush=True)
assert mid, 'no case put a boundary inside a frame'
D.barrier()
