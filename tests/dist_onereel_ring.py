"""The scene Ring path (onereel.parallel.ring) against K/V gathering and Ulysses on the same scene attention.

For each model, 2-step generations (480p, latent output) with the OneReel layout of the latency study (Hunyuan: frame
memory; Wan: halo of the next scene): 3 equal scenes (T_3) and 3 unequal ones (4, 8, 9 latent frames), each run with the
Eq.-2 path ('onereel', K/V gathering here), Ulysses ('onereel_ulysses') and Ring ('onereel_ring'). Compared to the
gathering run on the first transformer forward: Ulysses runs the same kernels (0 difference); Ring splits every scene's
keys and merges the partial results in fp32, so it differs by bf16 rounding, gauged by the gathering path itself on
FA2 instead of FA3 (the floor; ~6e-3 on Wan). The 2-step latents amplify that floor about tenfold, so they are printed
for reference only.
Run: torchrun --nproc_per_node 2 tests/dist_onereel_ring.py --model hunyuan
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from fig1_latency import BENCH, D, Runner  # noqa: E402
from onereel import attention  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['hunyuan', 'wan', 'cogvideox'], required=True)
    a = ap.parse_args()
    D.init()
    bench = {r['id']: r for r in json.loads(BENCH.read_text())['prompts']}
    t3 = bench['T_3']
    uneven = dict(t3, lengths=[4, 8, 9] if a.model != 'cogvideox' else [2, 4, 6])
    R = Runner(a.model, 480)
    first = {}

    def keep(m, args, out):
        if 'o' not in first:
            first['o'] = (out[0] if isinstance(out, tuple) else getattr(out, 'sample', out)).float().cpu()
    R.pipe.transformer.register_forward_hook(keep)
    rel = lambda x, y: ((x - y).norm() / y.norm()).item()
    rank0 = D.rank() == 0
    for name, row in (('equal', t3), ('uneven', uneven)):
        out = {}
        for run, method, be in (('onereel', 'onereel', 'fa3'), ('floor', 'onereel', 'fa2'),
                                ('onereel_ulysses', 'onereel_ulysses', 'fa3'), ('onereel_ring', 'onereel_ring', 'fa3')):
            attention.BACKEND = be
            R.configure(method)
            first.clear()
            lat = R.run(method, row, steps=2, latent=True)
            out[run] = first['o'], torch.as_tensor(lat).float().cpu()
            if rank0:
                c = R.ctx
                print(f'[ring-test] {a.model} {name} {run}: path {c.run_mode} own {c.own} V {c.path}', flush=True)
        attention.BACKEND = 'fa3'
        if rank0:
            ref = out['onereel']
            floor = rel(out['floor'][0], ref[0])
            for m in ('onereel_ulysses', 'onereel_ring'):
                err = rel(out[m][0], ref[0])
                ok = err <= 1.5 * floor
                print(f'[ring-test] {a.model} {name}: {m} vs gathering, first forward {err:.2e} (bf16 floor {floor:.2e}) '
                      f'{"PASS" if ok else "FAIL"}; 2-step latent {rel(out[m][1], ref[1]):.2e} '
                      f'(floor {rel(out["floor"][1], ref[1]):.2e})', flush=True)


if __name__ == '__main__':
    main()
