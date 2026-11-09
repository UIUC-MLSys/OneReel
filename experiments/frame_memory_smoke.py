"""Frame memory smoke: halo vs frame memory (all frames / boundary frames as candidates) on multi-shot prompts, one GPU.

Videos: OUT/<model>/<variant>/<id>/{video.mp4, meta.json} (the layout eval/event_order/judge_inputs.py reads), variants
  halo         the halo frames of the quality runs (Hunyuan / H3: 2 of both neighbours, Wan: 2 of the next)
  local        every scene reads only its own frames and sentence (no halo, no memory)
  full         every scene reads its own sentence and every frame of every other scene (full video attention)
  fm_m{M}      fm_all with a budget of M frames (memory-budget ablation; --budget is ignored)
  fm_all       frame memory, 4 representative frames per step, candidates = every latent frame (step 0: the halo)
  fm_boundary  the same, candidates = the first and last latent frame of every scene
  fm_all_layer   fm_all with the representatives chosen per layer (ablation of the per-step choice)
  fm_sink      fm_all with frame 0 as a sink every scene reads (1 sink + 3 representatives)
  fm_sink_layer  fm_sink with the representatives chosen per layer (ablation)
  fm_sink_s0next diagnostic: fm_sink, and scene 0 also reads the first 2 frames of scene 1
Hunyuan 480p CFG 6, Wan 704x1280 CFG 5, MiniMax-H3 544x960 (scenes split evenly over its latent frames), CogVideoX1.5
480x848 or --cog-res CFG 6 (layout in token frames of 2 latent frames: a shot of l latent frames takes ceil(l / 2); halo 1 token
frame; --budget counts token frames), seed 0.

Usage: CUDA_VISIBLE_DEVICES=0 python experiments/frame_memory_smoke.py --model hunyuan --ids holo_1,... --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from onereel.frame_memory import FrameMemory  # noqa: E402
from onereel.layout import SceneLayout  # noqa: E402
from onereel.parallel import dist as D  # noqa: E402
from onereel.utils import save_video  # noqa: E402

HALO = {'hunyuan': (2, 'adj'), 'wan': (2, 'next'), 'h3': (2, 'adj'), 'cogvideox': (1, 'adj')}
VARIANTS = ('halo', 'fm_all', 'fm_boundary')


class S0Next(FrameMemory):
    """diagnostic: scene 0 keeps the first 2 frames of scene 1 next to its representatives"""
    def _extra(self, reps, halo=None):
        out = super()._extra(reps, halo)
        lo, hi = self.layout.ranges[1]
        out[0] = sorted(set(out[0]) | set(range(lo, min(lo + 2, hi))))
        return out


def frame_memory(variant, budget, fallback):
    if variant in ('halo', 'local', 'full'):
        return None
    m = re.fullmatch(r'fm_m(\d+)', variant)
    if m:
        return FrameMemory(int(m.group(1)), 'all', fallback=fallback)
    if variant == 'fm_all_layer':
        return FrameMemory(budget, 'all', fallback=fallback, per_layer=True)
    if variant.startswith('fm_sink'):
        cls = S0Next if variant == 'fm_sink_s0next' else FrameMemory
        return cls(budget, 'all', fallback=fallback, sink=(0,), per_layer=variant == 'fm_sink_layer')
    return FrameMemory(budget, variant.split('_')[1], fallback=fallback)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['hunyuan', 'wan', 'h3', 'cogvideox'], required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--budget', type=int, default=4)
    ap.add_argument('--cog-res', default='480x848', help='CogVideoX height x width')
    ap.add_argument('--bench', type=Path, default=ROOT / 'dataset/multishot_bench.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scene-rope', action='store_true', help='Hunyuan: temporal RoPE restarts at every scene')
    ap.add_argument('--shared-pos', default='source', choices=['source', 'adjacent', 'switch'],
                    help='Hunyuan: temporal position of the shared frames (onereel.models.hunyuan15.Context.shared_pos)')
    ap.add_argument('--shared-switch', type=int, default=10, help='--shared-pos switch: first step with adjacent positions')
    a = ap.parse_args()
    D.init()
    bench = {r['id']: r for r in json.loads(a.bench.read_text())['prompts']}
    halo, sides = HALO[a.model]
    if a.model == 'hunyuan':
        from onereel.models import hunyuan15 as M
        ctx = M.Context(SceneLayout([7, 7, 7], halo=0))
        pipe = M.load()
        M.install(pipe, ctx, timing=True)
    elif a.model == 'wan':
        from onereel.models import wan as M
        ctx = M.Context(SceneLayout([7, 7, 7], halo=0))
        pipe = M.load()
        M.install(pipe, ctx)
    elif a.model == 'cogvideox':
        from onereel.models import cogvideox as M
        ch, cw = map(int, a.cog_res.split('x'))
        ctx = M.Context(SceneLayout([4, 4, 4], halo=0), height=ch, width=cw)
        pipe = M.load()
        M.install(pipe, ctx)
    else:
        from onereel.models import minimax_h3 as M
        M.allow_any_duration()
        ctx = M.Context(halo=0)
        dev = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
        conditioner, rest = M.load(dev, dev)
        M.install(rest.transformer, ctx)
    for variant in a.variants.split(','):
        for pid in a.ids.split(','):
            d = a.out / a.model / variant / pid
            if (d / 'video.mp4').exists():
                continue
            r = bench[pid]
            fm = frame_memory(variant, a.budget, (halo, sides))
            h = halo if variant == 'halo' else 0
            ctx.frame_memory = fm
            if a.shared_pos != 'source':
                assert a.model == 'hunyuan', '--shared-pos is implemented for HunyuanVideo-1.5'
                ctx.shared_pos, ctx.shared_switch = a.shared_pos, a.shared_switch
            if a.scene_rope:
                assert a.model == 'hunyuan', '--scene-rope is implemented for HunyuanVideo-1.5'
                ctx.scene_rope = True
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if a.model == 'h3':
                ctx.halo, ctx.lengths = h, None
                out = M.generate(conditioner, rest, ctx, r['sentences'], M.snap_frames(r['frames']), 544, 960, 0)
                v = out['videos'][0] if isinstance(out['videos'], (list, tuple)) else out['videos']
                import numpy as np
                video = v.float().cpu().numpy() if torch.is_tensor(v) else np.stack([np.asarray(f) for f in v]).astype(np.float32) / 255
            elif a.model == 'cogvideox':
                ctx.layout = SceneLayout([(l + 1) // 2 for l in r['lengths']], halo=h, sides=sides)
                video = M.generate(pipe, ctx, r['sentences'], steps=20, guidance=6.0, seed=0)
            else:
                ctx.layout = SceneLayout(r['lengths'], halo=h, sides=sides)
                if variant == 'full':
                    lay = ctx.layout
                    lay.extra = {j: [f for f in range(lay.n_frames) if f not in lay.frames[j]] for j in range(lay.n_scenes)}
                if a.model == 'hunyuan':
                    video = M.generate(pipe, ctx, r['sentences'], steps=20, guidance=6.0, seed=0)
                else:
                    video = M.generate(pipe, ctx, r['sentences'], steps=20, seed=0, guidance=M.GUIDANCE, negative=M.NEGATIVE)
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            d.mkdir(parents=True, exist_ok=True)
            save_video(video, d / 'video.mp4', fps=M.FPS)
            meta = dict(model=a.model, setting=variant, id=pid, seed=0, frames=len(video), seconds=sec, budget=fm.budget if fm else 0,
                        scene_rope=a.scene_rope, shared_pos=a.shared_pos,
                        shared_switch=a.shared_switch if a.shared_pos == 'switch' else None,
                        lengths=ctx.layout.lengths, reps=fm.history if fm else None)
            (d / 'meta.json').write_text(json.dumps(meta, indent=1))
            last = fm.history[-1] if fm and fm.history else None
            if isinstance(last, dict):
                last = {l: last[l] for l in list(last)[:3]}
            print(f'[done] {a.model} {variant}/{pid} {sec:.0f}s reps(last) {last}', flush=True)


if __name__ == '__main__':
    main()
