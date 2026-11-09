"""K/V gathering's text exchange: rows sent to the ranks that read them (text_sync = 'rows') against the all-reduce,
on the full HunyuanVideo-1.5 model with frame memory (run on P GPUs):
  torchrun --nproc_per_node 2 tests/dist_hunyuan_text_rows.py
Two-step requests (step 1 reads frame memory) whose scenes split over the ranks; the latents must match bit for bit."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from fig1_latency import BENCH, D, Runner  # noqa: E402

D.init()
bench = {r['id']: r for f in (BENCH, ROOT / 'dataset/multishot_bench.json') for r in json.loads(f.read_text())['prompts']}
R = Runner('hunyuan', 480)
for pid, lengths in (('holo_1', None), ('s2s_HcT3uyNtPUE.41', None), ('T_12', [3, 9, 2, 5, 7, 4, 6, 8, 2, 11, 5, 6])):
    row = dict(bench[pid], lengths=lengths) if lengths else bench[pid]
    out = {}
    for sync in ('all_reduce', 'rows'):
        R.configure('onereel_gather')
        R.ctx.text_sync = sync
        out[sync] = R.run('onereel_gather', row, steps=2, latent=True)
        out[sync] = out[sync][0] if isinstance(out[sync], (list, tuple)) else out[sync]
    same = torch.equal(out['all_reduce'], out['rows'])
    diff = (out['all_reduce'].float() - out['rows'].float()).abs().max().item()
    if D.rank() == 0:
        print(f"{'PASS' if same else 'FAIL'} {pid} lengths {row.get('lengths')} own {R.ctx.own} max diff {diff:.3g}", flush=True)
    assert same, (pid, diff)
D.barrier()
