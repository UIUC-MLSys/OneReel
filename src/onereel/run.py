"""Generate one multi-shot video with OneReel.

Prompts come from a bench file ({"prompts": [{id, sentences, subject?, lengths?}]}) or --sentences (one per '|').
Scene lengths are in latent frames: --lengths 8,8,8, else the bench row's lengths (HunyuanVideo-1.5, Wan), else the
frames are split evenly.

Examples (onereel env; source scripts/env.sh first):
  python -m onereel.run --model hunyuan15 --bench B.json --id M6 --lengths 8,8,8,8,8,8 --out DIR
  python -m onereel.run --model hunyuan15 --bench dataset/multishot_bench.json --id holo_1 --frame-memory 4 \
    --scene-rope --shared-pos switch --shared-switch 5 --out DIR       (the HunyuanVideo-1.5 quality setting)
  python -m onereel.run --model wan --bench dataset/multishot_bench.json --id holo_1 --sides next --out DIR
                                                                       (Wan: 2 halo frames of the next scene, CFG 5)
  torchrun --nproc_per_node 4 -m onereel.run --model hunyuan15 --mode auto ... (Eq. 2 picks K/V gathering or Ulysses)
  python -m onereel.run --model wan --sentences "A ...|A ..." --frames 81 --out DIR
  python -m onereel.run --model h3 --bench B.json --id M6 --frames 124 --out DIR [--any-duration]
  torchrun --nproc_per_node 2 -m onereel.run --model wan|h3 --mode auto --memory 128 ... (every model runs on P GPUs)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from onereel import attention
from onereel.layout import SceneLayout
from onereel.parallel import dist as D


def _prompt(a):
    if a.sentences:
        return a.id or 'prompt', a.sentences.split('|'), a.subject, None
    rows = {r['id']: r for r in json.loads(Path(a.bench).read_text())['prompts']}
    r = rows[a.id]
    return a.id, r['sentences'], a.subject or r.get('subject', ''), r.get('lengths')


def _layout(a, n_scenes, n_frames, bench_lengths=None):
    lengths = [int(x) for x in a.lengths.split(',')] if a.lengths else None if a.frames else bench_lengths
    if lengths:
        lay = SceneLayout(lengths, halo=a.halo, sides=a.sides)
        assert lay.n_scenes == n_scenes, 'one length per sentence'
        return lay
    return SceneLayout.even(n_frames, n_scenes, halo=a.halo, sides=a.sides)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['hunyuan15', 'wan', 'h3'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bench', default='')
    ap.add_argument('--id', default='')
    ap.add_argument('--sentences', default='')
    ap.add_argument('--subject', default='')
    ap.add_argument('--lengths', default='', help='latent frames per scene')
    ap.add_argument('--frames', type=int, default=0, help='pixel frames when --lengths is not given')
    ap.add_argument('--halo', type=int, default=2)
    ap.add_argument('--sides', choices=['adj', 'prev', 'next'], default='adj')
    ap.add_argument('--subject-links', type=int, default=0, help='key frames per other scene (Hunyuan, H3)')
    ap.add_argument('--frame-memory', type=int, default=0,
                    help='representative frames chosen once per step (k-means over frames), replacing the halo frames')
    ap.add_argument('--fm-candidates', choices=['all', 'boundary'], default='all',
                    help='frame memory candidates: every latent frame, or the first and last frame of every scene')
    ap.add_argument('--fm-sink', default='', help='frames every scene reads, part of the budget (e.g. 0 on Wan)')
    ap.add_argument('--fm-per-layer', action='store_true', help='ablation: representatives chosen per layer')
    ap.add_argument('--memory', type=int, default=0,
                    help='bounded cross-scene memory: M representatives per head replace the halo frames')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg', type=float, default=None,
                    help='guidance scale (default: Hunyuan 6, Wan 5 with its negative prompt; 1 turns CFG off)')
    ap.add_argument('--scene-rope', action='store_true', help='Hunyuan: temporal RoPE restarts at every scene')
    ap.add_argument('--shared-pos', choices=['source', 'adjacent', 'switch'], default='source',
                    help='Hunyuan: temporal RoPE position of the shared frames (hunyuan15.Context.shared_pos)')
    ap.add_argument('--shared-switch', type=int, default=10, help='--shared-pos switch: first step with adjacent positions')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mode', choices=['auto', 'scene', 'ulysses'], default='auto',
                    help='multi-GPU path: K/V gathering over the work-balanced partition, Ulysses, or Eq. 2')
    ap.add_argument('--theta', type=float, default=1.0, help='Eq. 2: gather bandwidth relative to all-to-all')
    ap.add_argument('--backend', choices=['fa2', 'fa3', 'auto'], default='auto')
    ap.add_argument('--height', type=int, default=544, help='H3')
    ap.add_argument('--width', type=int, default=960, help='H3')
    ap.add_argument('--any-duration', action='store_true', help='H3: lift the 5-15 s check')
    a = ap.parse_args()
    assert a.model == 'hunyuan15' or not (a.scene_rope or a.shared_pos != 'source'), \
        '--scene-rope and --shared-pos are implemented for HunyuanVideo-1.5'
    fallback = (a.halo, a.sides)                       # frame memory reads these halo frames at step 0
    if a.memory or a.frame_memory:
        a.halo = 0

    def fmem():
        if not a.frame_memory:
            return None
        from onereel.frame_memory import FrameMemory
        sink = tuple(int(x) for x in a.fm_sink.split(',') if x)
        return FrameMemory(a.frame_memory, a.fm_candidates, fallback=fallback, sink=sink, per_layer=a.fm_per_layer)
    attention.BACKEND = a.backend
    pid, sentences, subject, bench_lengths = _prompt(a)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {'model': a.model, 'id': pid, 'sentences': sentences, 'seed': a.seed, 'backend': attention.backend()}
    t0 = time.perf_counter()
    if a.model == 'hunyuan15':
        from onereel.models import hunyuan15 as M
        D.init()
        lay = _layout(a, len(sentences), (a.frames - 1) // 4 + 1 if a.frames else 0, bench_lengths)
        ctx = M.Context(lay, mode=a.mode, theta=a.theta, subject_links=a.subject_links, memory=a.memory, frame_memory=fmem(),
                        scene_rope=a.scene_rope, shared_pos=a.shared_pos, shared_switch=a.shared_switch)
        cfg = 6.0 if a.cfg is None else a.cfg
        pipe = M.load()
        M.install(pipe, ctx, timing=True)
        D.barrier()
        t0 = time.perf_counter()
        video = M.generate(pipe, ctx, sentences, steps=a.steps, guidance=cfg, seed=a.seed, subject=subject)
        meta.update(lengths=lay.lengths, halo=lay.halo, sides=lay.sides, mode=ctx.run_mode, path=ctx.path, world=D.world(), own=ctx.own, memory=a.memory,
                    scene_rope=a.scene_rope, shared_pos=a.shared_pos, shared_switch=a.shared_switch if a.shared_pos == 'switch' else None,
                    steps=a.steps, cfg=cfg, denoise_s=sum(ctx.forward_ms) / 1e3, vae_s=sum(ctx.vae_ms) / 1e3)
        fps = M.FPS
    elif a.model == 'wan':
        from onereel.models import wan as M
        D.init()
        lay = _layout(a, len(sentences), (a.frames - 1) // 4 + 1 if a.frames else 0, bench_lengths)
        ctx = M.Context(lay, mode=a.mode, theta=a.theta, memory=a.memory, frame_memory=fmem())
        cfg = M.GUIDANCE if a.cfg is None else a.cfg
        pipe = M.load()
        M.install(pipe, ctx)
        D.barrier()
        t0 = time.perf_counter()
        video = M.generate(pipe, ctx, sentences, steps=a.steps, seed=a.seed, guidance=cfg, negative=M.NEGATIVE)
        meta.update(lengths=lay.lengths, halo=lay.halo, sides=lay.sides, steps=a.steps, cfg=cfg, mode=ctx.run_mode, path=ctx.path,
                    world=D.world(), own=ctx.own, memory=a.memory)
        fps = M.FPS
    else:
        from onereel.models import minimax_h3 as M
        if a.any_duration:
            M.allow_any_duration()
        lengths = [int(x) for x in a.lengths.split(',')] if a.lengths else None
        ctx = M.Context(halo=a.halo, lengths=lengths, subject_links=a.subject_links, mode=a.mode, theta=a.theta,
                        memory=a.memory, frame_memory=fmem())
        n = D.init()
        one = n > 1 or os.environ.get('H3_SINGLE_GPU') == '1'      # every rank (or the one GPU) holds the whole pipeline
        dev = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
        conditioner, rest = M.load(dev, dev) if one else M.load()
        ctx.offload_text_encoder = one
        M.install(rest.transformer, ctx)
        frames = M.snap_frames(a.frames or 124)
        D.barrier()
        t0 = time.perf_counter()
        out = M.generate(conditioner, rest, ctx, sentences, frames, a.height, a.width, a.seed, subject)
        v = out['videos'][0] if isinstance(out['videos'], (list, tuple)) else out['videos']
        if torch.is_tensor(v):
            video = v.float().cpu().numpy()
        else:  # default output_type pil: uint8 frames, save_video expects floats in [0, 1]
            import numpy as np
            video = np.stack([np.asarray(f) for f in v]).astype(np.float32) / 255
        meta.update(lengths=ctx.layout.lengths, halo=ctx.halo, frames=frames, forwards=ctx.fwd, mode=ctx.run_mode,
                    path=ctx.path, world=D.world(), memory=a.memory)
        fps = M.FPS
        try:
            import numpy as np
            import soundfile as sf
            x = out['audio'][0] if isinstance(out['audio'], (list, tuple)) else out['audio']
            x = (x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)).squeeze()
            if D.rank() == 0:
                sf.write(str(a.out / f'{pid}.wav'), x.T if x.ndim == 2 and x.shape[0] <= 2 else x, int(out['sampling_rate']))
        except ImportError:
            pass
    meta['seconds'] = time.perf_counter() - t0
    if a.frame_memory:
        meta.update(frame_memory=a.frame_memory, fm_candidates=a.fm_candidates, fm_reps=ctx.frame_memory.history)
    if D.rank() == 0:
        from onereel.utils import save_video
        save_video(video, a.out / f'{pid}.mp4', fps=fps)
        (a.out / f'{pid}.json').write_text(json.dumps(meta, indent=1))
        print(f"[onereel] {a.model} {pid}: {meta['seconds']:.1f} s -> {a.out / (pid + '.mp4')}", flush=True)


if __name__ == '__main__':
    main()
