"""Pipelined K/V gathering (exchange per head chunk) against the unchunked exchange (run on P GPUs):
  torchrun --nproc_per_node 2 tests/dist_onereel_kv_chunks.py
Hunyuan's and Wan's scene attention on the work-balanced partition, with halo frames and frame-memory extras (so some
scenes read only local keys and others wait for the exchange); every chunking must give the same output, bit for bit."""
import torch

from onereel.layout import SceneLayout
from onereel.models import hunyuan15 as HY, wan as WN
from onereel.parallel import comm, dist as D

D.init()
r, n = D.rank(), D.world()
T, H, Dh = 64, 16, 128
for lengths, halo, extra in (([3, 5, 2, 4, 6], 0, True), ([10, 1, 1, 2], 2, False), ([2, 2, 2, 2, 2, 2], 1, False),
                             ([5, 2, 7], 0, True), ([4, 4, 4, 4, 4, 4, 4, 4], 2, True)):
    g = torch.Generator().manual_seed(len(lengths))
    rnd = lambda *s: torch.randn(*s, generator=g).to('cuda', torch.bfloat16)
    lay = SceneLayout(lengths, halo=halo)
    if extra:                                                   # frame memory: a few frames of other scenes
        lay.extra = {j: [(7 * j + 3) % lay.n_frames, lay.n_frames - 1] for j in range(lay.n_scenes)}
    S = lay.n_frames * T
    counts = [3 + j for j in range(lay.n_scenes)]
    q, k, v = rnd(S, H, Dh), rnd(S, H, Dh), rnd(S, H, Dh)
    eq, ek, ev = (rnd(sum(counts), H, Dh) for _ in range(3))
    own = comm.partition(lay, n, T, 'scene', counts)
    a, b = own[r]
    ql, kl, vl = (x[a:b].contiguous() for x in (q, k, v))
    outs = {}
    for C in (1, 2, 3, 4, 16):
        hc = HY.Context(lay, T=T, own=own, counts={True: counts}, fwd=1, kv_chunks=C)
        wc = WN.Context(lay, T=T, own=own, kv_chunks=C)
        hv, ht = HY.Processor(hc, 0).scene(ql, kl, vl, eq, ek, ev)
        wv = WN.Processor.scene(type('P', (), {'ctx': wc})(), ql, kl, vl)
        outs[C] = (hv, ht, wv)
    for C, o in outs.items():
        for name, x, y in zip(('hunyuan video', 'hunyuan text', 'wan video'), o, outs[1]):
            assert torch.equal(x, y), (lengths, C, name, (x.float() - y.float()).abs().max().item())
    near = sum(all(a <= lo and hi <= b for lo, hi in comm.frame_tokens(lay.key_frames(j), T))
               for j, (lo, hi) in enumerate(lay.ranges) if lo * T < b and hi * T > a)
    if r == 0:
        print(f'PASS {lengths} halo={halo} extra={extra} own={own} local scenes on rank 0: {near}', flush=True)
D.barrier()
