"""OneReel + intra-scene Sol-Attn on one GPU, synthetic tensors (onereel env, SM90):
  python tests/test_intra_scene_sparse.py
The override with dense attention must equal OneReel's scene attention (same scopes, same layout); with the Sol-Attn
kernel it must stay close (sparse attention) and keep the shared frames and text exact."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from onereel.layout import SceneLayout  # noqa: E402
from onereel.models import hunyuan15 as HY  # noqa: E402
from onereel.parallel import dist as D  # noqa: E402
from intra_scene_sparse import IntraSceneSol  # noqa: E402

D.init()
T, H, Dh = 1590, 8, 128                                         # 480p HunyuanVideo frame, the local heads of 2 GPUs
rel = lambda x, y: ((x.float() - y.float()).norm() / y.float().norm()).item()
for lengths, extra in (([84], False), ([21, 21, 21, 21], True), ([14] * 6, True), ([7] * 12, True)):
    g = torch.Generator().manual_seed(len(lengths))
    rnd = lambda *s: torch.randn(*s, generator=g).to('cuda', torch.bfloat16)
    lay = SceneLayout(lengths, halo=0)
    if extra:                                                   # frame memory: 4 representatives of the request
        reps = [3, 30, 50, 80]
        lay.extra = {j: [f for f in reps if not lo <= f < hi] for j, (lo, hi) in enumerate(lay.ranges)}
    S = lay.n_frames * T
    counts = [60 + 7 * j for j in range(lay.n_scenes)]
    q, k, v = rnd(S, H, Dh), rnd(S, H, Dh), rnd(S, H, Dh)
    eq, ek, ev = (rnd(sum(counts), H, Dh) for _ in range(3))
    out = {}
    for name, ov in (('onereel', None), ('dense', True), ('sol', False)):
        c = HY.Context(lay, T=T, own=[(0, S)], counts={True: counts}, fwd=1, run_mode='ulysses')
        c.attn_override = None if ov is None else IntraSceneSol(c, dense=ov)
        out[name] = HY.Processor(c, 0).ulysses(q, k, v, eq, ek, ev)
    d = [rel(out['dense'][i], out['onereel'][i]) for i in (0, 1)]
    s = [rel(out['sol'][i], out['onereel'][i]) for i in (0, 1)]
    print(f'{lengths[:3]}x{len(lengths)} extra={extra}: dense-layout vs OneReel video {d[0]:.1e} text {d[1]:.1e} | '
          f'Sol vs OneReel video {s[0]:.1e} text {s[1]:.1e}', flush=True)
    assert max(d) < 1e-2 and s[1] < 1e-2, (lengths, d, s)
