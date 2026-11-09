"""OneReel + intra-scene Sol-Attn on real activations (HunyuanVideo-1.5 480p, 2 GPUs, CFG, frame memory):
  torchrun --nproc_per_node 2 tests/dist_intra_scene_sol.py
Latents after a few steps of a 4-scene request: OneReel+Sol against OneReel (both on the Ulysses path), next to the
Sol-Attn baseline against Dense on the same request, so the intra-scene sparsification can be judged against the error
Sol-Attn already makes on the full video."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from fig1_latency import BENCH, D, Runner  # noqa: E402
from intra_scene_sparse import IntraSceneSol  # noqa: E402

D.init()
bench = {r['id']: r for r in json.loads(BENCH.read_text())['prompts']}
R = Runner('hunyuan', 480)
row = dict(bench['T_12'], sentences=bench['T_12']['sentences'][:4], lengths=[21, 21, 21, 21])
row['prompt'] = ' '.join(row['sentences'])
rel = lambda x, y: ((x.float() - y.float()).norm() / y.float().norm()).item()
for steps in (2, 6):
    out = {}
    for name, method, sol in (('onereel', 'onereel_ulysses', False), ('onereel+sol', 'onereel_ulysses', True),
                              ('dense', 'dense', False), ('solk', 'solk', False)):
        R.configure(method)
        if sol:
            R.ctx.attn_override = IntraSceneSol(R.ctx)
        o = R.run(method, row, steps=steps, latent=True)
        out[name] = o[0] if isinstance(o, (list, tuple)) else o
    if D.rank() == 0:
        print(f'steps {steps}: OneReel+Sol vs OneReel {rel(out["onereel+sol"], out["onereel"]):.3f} | '
              f'Sol-Attn vs Dense {rel(out["solk"], out["dense"]):.3f} | OneReel vs Dense '
              f'{rel(out["onereel"], out["dense"]):.3f}', flush=True)
D.barrier()
