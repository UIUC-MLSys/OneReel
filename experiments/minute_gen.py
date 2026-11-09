"""One-minute multi-scene videos on N GPUs (torchrun) through fig1_latency's Runner, HunyuanVideo-1.5 480p, 20 steps,
CFG 6, seed 0:
  dense      every scene in one joint sequence with the joint prompt, full attention (Ulysses over the GPUs)
  onereel   scene attention with frame memory (M = 4), path by Eq. 2, scene-local temporal RoPE
--rope global|scene overrides the temporal RoPE (default: scene for onereel, global for dense; scene restarts the
index at every scene of the request, also under dense attention); a non-default RoPE is the setting <method>_<rope>rope.
Rank 0 writes OUT/hunyuan/<method>/<id>/{video.mp4, meta.json} (the layout the eval scripts read).
Usage: torchrun --nproc_per_node 2 experiments/minute_gen.py --method dense --ids minute_lighthouse --out OUT
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
import fig1_latency  # noqa: E402
from onereel.parallel import dist as D  # noqa: E402
from onereel.utils import save_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=('dense', 'onereel'), required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--bench', type=Path, default=ROOT / 'dataset/minute_bench.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rope', choices=('global', 'scene'))
    a = ap.parse_args()
    D.init()
    bench = {r['id']: r for r in json.loads(a.bench.read_text())['prompts']}
    R = fig1_latency.Runner('hunyuan', 480)
    R.configure(a.method)
    default = 'scene' if a.method == 'onereel' else 'global'
    rope = a.rope or default
    setting = a.method if rope == default else f'{a.method}_{rope}rope'
    R.ctx.scene_rope = rope == 'scene'
    for pid in a.ids.split(','):
        d = a.out / 'hunyuan' / setting / pid
        if (d / 'video.mp4').exists():
            continue
        R.ctx.rope_lengths = bench[pid]['lengths']
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video = R.run(a.method, bench[pid], steps=20)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        if D.rank() == 0:
            d.mkdir(parents=True, exist_ok=True)
            save_video(video, d / 'video.mp4', fps=R.M.FPS)
            meta = dict(model='hunyuan', setting=setting, id=pid, seed=0, frames=len(video), seconds=sec,
                        gpus=D.world(), lengths=bench[pid]['lengths'], rope=rope,
                        path=R.ctx.path.get('path') if R.ctx.path else None)
            (d / 'meta.json').write_text(json.dumps(meta, indent=1))
            print(f'[done] hunyuan {setting}/{pid} {sec:.0f}s on {D.world()} GPUs', flush=True)


if __name__ == '__main__':
    main()
