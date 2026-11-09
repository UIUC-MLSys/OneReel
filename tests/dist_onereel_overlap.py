"""K/V gathering overlap (ctx.overlap = lse: local keys while the exchange is in flight, then the received keys, merged by
log-sum-exp) against waiting for the exchange; Hunyuan scene attention on synthetic tensors (run on P GPUs):
  torchrun --nproc_per_node 2 tests/dist_onereel_overlap.py"""
import torch

from onereel.layout import SceneLayout
from onereel.models import hunyuan15 as HY
from onereel.parallel import comm, dist as D

D.init()
r, n = D.rank(), D.world()
T, H, Dh = 64, 16, 128
rel = lambda x, y: ((x.float() - y.float()).norm() / y.float().norm()).item()
for lengths, halo, extra in (([3, 5, 2, 4, 6], 0, True), ([10, 1, 1, 2], 2, False), ([5, 2, 7], 0, True),
                             ([4, 4, 4, 4, 4, 4, 4, 4], 2, True)):
    g = torch.Generator().manual_seed(len(lengths))
    rnd = lambda *s: torch.randn(*s, generator=g).to('cuda', torch.bfloat16)
    lay = SceneLayout(lengths, halo=halo)
    if extra:
        lay.extra = {j: [(7 * j + 3) % lay.n_frames, lay.n_frames - 1] for j in range(lay.n_scenes)}
    S = lay.n_frames * T
    counts = [3 + j for j in range(lay.n_scenes)]
    q, k, v = rnd(S, H, Dh), rnd(S, H, Dh), rnd(S, H, Dh)
    eq, ek, ev = (rnd(sum(counts), H, Dh) for _ in range(3))
    own = comm.partition(lay, n, T, 'scene', counts)
    a, b = own[r]
    out = {}
    for ov in ('none', 'lse'):
        c = HY.Context(lay, T=T, own=own, counts={True: counts}, fwd=1, overlap=ov)
        out[ov] = HY.Processor(c, 0).scene(*(x[a:b].contiguous() for x in (q, k, v)), eq, ek, ev)
    ev_, et_ = rel(out['lse'][0], out['none'][0]), rel(out['lse'][1], out['none'][1])
    if r == 0:
        print(f'{lengths} halo={halo} extra={extra}: video rel {ev_:.1e}, text rel {et_:.1e}', flush=True)
    assert ev_ < 1e-2 and et_ < 1e-2
D.barrier()
