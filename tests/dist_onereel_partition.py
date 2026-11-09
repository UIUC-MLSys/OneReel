"""K/V gathering partitions (Algorithm 1, equal tokens, whole scenes) and the forced paths against each other.

HunyuanVideo-1.5 at 480p, 2 denoising steps, frame memory as in the latency study, on two layouts of the system
ablations (12 equal scenes; one 48-frame scene and nine 4-frame scenes). Every partition computes the same attention
scopes, only the rank that owns each query changes, so the latents should match Algorithm 1's exactly (bf16 kernels
are deterministic per query); Ulysses runs the same per-scene kernels (0 difference so far, tests/dist_onereel_ring.py).
Run: torchrun --nproc_per_node 2 tests/dist_onereel_partition.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import fig1_latency  # noqa: E402
from fig1_latency import BENCH, D, Runner  # noqa: E402
from e2e_breakdown import layout_row  # noqa: E402


def main():
    D.init()
    base = {r['id']: r for r in json.loads(BENCH.read_text())['prompts']}['T_12']
    R = Runner('hunyuan', 480)
    for lay in ('equal', 'one_long'):
        row = layout_row(base, lay)
        out = {}
        for name, method, part in (('balanced', 'onereel_gather', 'balanced'), ('tokens', 'onereel_gather', 'tokens'),
                                   ('scenes', 'onereel_gather', 'scenes'), ('ulysses', 'onereel_ulysses', 'balanced')):
            fig1_latency.PARTITION = part
            R.configure(method)
            out[name] = torch.as_tensor(R.run(method, row, steps=2, latent=True)).float().cpu()
            if D.rank() == 0:
                print(f'[partition-test] {lay} {name}: run {R.ctx.run_mode} own {R.ctx.own}', flush=True)
        if D.rank() == 0:
            ref = out['balanced']
            for k in ('tokens', 'scenes', 'ulysses'):
                print(f'[partition-test] {lay}: {k} vs balanced rel err {((out[k] - ref).norm() / ref.norm()).item():.2e}',
                      flush=True)


if __name__ == '__main__':
    main()
