"""Latency against the number of scenes at a fixed video length (one multi-GPU instance, USP degree = world size).

The same 84 latent frames (333 frames, about 14 s at 24 fps) are split into K equal scenes, K in --counts, the first K
sentences of the timing-bench prompt T_12 describing them; with --uneven also one split of 6 scenes of unequal length
(4, 8, 12, 16, 20, 24 latent frames). Dense (the joint prompt, full attention over the whole video) is one request
whatever K is; OneReel and Separate (every scene its own request) run per split. The engine, warm-up and timing are
those of experiments/fig1_latency.py (its Runner): one full warm-up generation per method, one untimed one-step pass
per split, then the timed runs (text encoding + denoising + VAE decode, barrier to barrier).

Usage: torchrun --nproc_per_node 2 experiments/scene_count_latency.py --model hunyuan --res 480 --out DIR
Writes DIR/<model>_<res>.jsonl, one line per run (rank 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from fig1_latency import BENCH, D, Runner, timed  # noqa: E402

UNEVEN = [4, 8, 12, 16, 20, 24]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['hunyuan', 'wan', 'cogvideox'], required=True)
    ap.add_argument('--res', type=int, choices=[480, 720], required=True)
    ap.add_argument('--total', type=int, default=84, help='latent frames of every video (layout frames on CogVideoX)')
    ap.add_argument('--counts', default='1,2,4,6,12')
    ap.add_argument('--uneven', action='store_true')
    ap.add_argument('--methods', default='dense,onereel,separate')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    D.init()
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = torch._dynamo.config.recompile_limit = 64   # one compiled shape per split
    bench = {r['id']: r for r in json.loads(BENCH.read_text())['prompts']}
    base = bench['T_12']['sentences']
    splits = []
    for k in map(int, a.counts.split(',')):
        assert a.total % k == 0 and k <= len(base), f'{a.total} frames do not split into {k} scenes'
        splits.append((f'K{k}', [a.total // k] * k))
    if a.uneven:
        assert sum(UNEVEN) == a.total
        splits.append(('K6_uneven', UNEVEN))
    rows = {name: dict(id=name, sentences=base[:len(L)], prompt=' '.join(base[:len(L)]), lengths=L) for name, L in splits}
    R = Runner(a.model, a.res)
    a.out.mkdir(parents=True, exist_ok=True)
    log = a.out / f'{a.model}_{a.res}.jsonl'
    rank0 = D.rank() == 0

    def write(rec):
        if rank0:
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f'[scenes] {rec}', flush=True)

    base_rec = dict(model=a.model, res=a.res, world=D.world(), total=a.total)
    for method in a.methods.split(','):
        R.configure(method)
        R.run(method, bench['WARM_3'], scene=0 if method == 'separate' else None)          # full warm-up generation
        todo = [rows['K12' if 'K12' in rows else splits[-1][0]]] if method == 'dense' else [rows[n] for n, _ in splits]
        for row in todo:
            if method == 'separate':
                for L in sorted(set(row['lengths'])):                     # every scene length compiles before timing
                    R.run(method, row, scene=row['lengths'].index(L), steps=1, latent=True)
                per = [timed(lambda: R.run(method, row, scene=j))[1] for j in range(len(row['lengths']))]
                write(dict(base_rec, method=method, split=row['id'], lengths=row['lengths'], seconds=sum(per),
                           scene_seconds=per))
                continue
            R.run(method, row, steps=1, latent=True)                                       # per-shape compilation
            _, s = timed(lambda: R.run(method, row))
            write(dict(base_rec, method=method, split=row['id'], lengths=row['lengths'], seconds=s,
                       path=getattr(R.ctx, 'path', None)))


if __name__ == '__main__':
    main()
