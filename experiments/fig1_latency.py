"""Figure 1: end-to-end latency of every method on one multi-GPU instance (torchrun; USP degree = world size).

All methods run in the OneReel engine (src/onereel), one process group, one software environment:
  dense      the joint prompt as one scene: full attention, Ulysses over the world (pure-Ulysses USP)
  radial / svg / solk
             as dense, with the sparse method (experiments/sparse_baselines.py: Radial and SVG reimplementations,
             official Sol-Attn kernel) on the local heads after the Ulysses all-to-all, dense warm-up as configured there
  onereel_gather  OneReel forced onto K/V gathering (Eq. 2 not consulted), for the path ablation
  onereel   per-sentence scenes. Hunyuan / H3: frame memory (4 representative frames chosen once per denoising
             step from every latent frame; step 0 reads 2 halo frames of both neighbours). Wan: halo frames (2 of the
             next scene; Wan degrades with non-adjacent frames). Multi-GPU path chosen per request by Eq. 2 (--mode auto)
  separate   every scene is its own request (its sentence alone, dense attention, Ulysses over the world), served
             one after another; the latency is the sum over the scenes. H3: all scene sentences are encoded first
             with one text-encoder residency (the encoder then leaves the GPU once), counted once in the latency;
             H3 lengths must be 17n + 5, so the scenes get 22 or 39 frames, mixed so that the total is the
             smallest one not below the joint video's frames (separate_frames)
Timing: model loading excluded; per method one full warm-up generation (WARM_3), then every request once with one
denoising step and no VAE decode so per-length compilation (Wan q/k fusion, flex attention) is not timed; then the
timed runs (text encoding + denoising + VAE decode, barrier to barrier). CFG: Hunyuan 6, Wan 5, CogVideoX 6 (H3 is
distilled). CogVideoX1.5 (16 fps): scenes of 8 latent frames = 4 token frames; OneReel reads 1 halo token frame of
both neighbours (frame memory gives artifacts on it, as on Wan).
Results: OUT/<model>_<res>.jsonl, one line per run (rank 0); the first request's video per method is saved.

Usage:
  torchrun --nproc_per_node 2 experiments/fig1_latency.py --model hunyuan --res 480 \
      --methods dense,radial,svg,solk,separate,onereel --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from onereel.frame_memory import FrameMemory  # noqa: E402
from onereel.layout import SceneLayout  # noqa: E402
from onereel.parallel import dist as D  # noqa: E402

BENCH = ROOT / 'dataset/timing_bench.json'
IDS = 'T_3,T_4,T_5,T_6,T_8,T_10,T_12'
RES = {'hunyuan': {480: (480, 848), 720: (720, 1280)}, 'wan': {480: (480, 832), 720: (704, 1280)},
       'h3': {480: (480, 832), 720: (704, 1280)}, 'cogvideox': {480: (480, 848), 720: (768, 1360)}}
CFG = {'hunyuan': 6.0, 'wan': 5.0, 'cogvideox': 6.0}
HALO = {'hunyuan': (2, 'adj'), 'wan': (2, 'next'), 'h3': (2, 'adj'), 'cogvideox': (1, 'adj')}
SCENE_FRAMES = {'cogvideox': 4}             # layout frames per scene (CogVideoX: 4 token frames = 8 latent frames); else 7
FRAME_MEMORY = {'hunyuan': 4, 'h3': 4}          # OneReel budget (frames); models absent here use the halo frames
SPARSE = ('radial', 'svg', 'solk')


class SparseAttn:
    """experiments/sparse_baselines.Sparse on the local heads; `force` skips its dense steps and layers (warm-up of the
    per-length compilation)"""

    def __init__(self, model, method, n_layers, ctx):
        import sparse_baselines as S
        self.S, self.E, self.ctx, self.kind = S, S.E, ctx, model
        self.sp = S.Sparse(model, method, n_layers)
        self.E.St.n_layers = n_layers
        self.force = False
        if method == 'solk' and S.SOLK_RUN is None:
            os.environ['SOL_ATTN_STRICT'] = '1'
            sys.path.insert(0, str(ROOT / 'third_party/Sana'))
            from techniques.sparse_backends.sol_attn_backend import _run_sol_attn_bthd
            S.SOLK_RUN = _run_sol_attn_bthd

    def __call__(self, layer, q, k, v, eq=None, ek=None, ev=None):
        St, ctx = self.E.St, self.ctx
        St.fwd, St.cfg, St.T, St.F = ctx.fwd, ctx.cfg and self.kind != 'cogvideox', ctx.T, ctx.layout.n_frames  # Cog: CFG batched
        warm, dl = self.sp.warm, self.sp.dense_layers
        if self.force:
            self.sp.warm, self.sp.dense_layers = 0, 0
        try:
            if eq is None:
                r = self.sp('wan', layer, q, k, v)
                return None if r is None else r.view(q.shape)
            r = self.sp('hunyuan', layer, q, k, v, eq, ek, ev, torch.ones(eq.shape[0], device=q.device))
            return None if r is None else (r[0].view(q.shape), r[1].view(eq.shape))
        finally:
            self.sp.warm, self.sp.dense_layers = warm, dl


class H3SparseAttn(SparseAttn):
    """MiniMax-H3: the sparse method sees the video rows in frame order as its video and the text and audio rows as its
    always-attended prefix (Sol-Attn's exact prefix sink on H3); 50 denoising steps, no CFG"""

    def __init__(self, method, n_layers, ctx):
        super().__init__('h3', method, n_layers, ctx)
        self.sp = self.S.Sparse('h3', method, n_layers, steps=50)
        self.rows = {}

    def __call__(self, layer, q, k, v):
        c, St = self.ctx, self.E.St
        vid = c.rows['vid']
        N = c.par['N'] if c.par and 'N' in c.par else q.shape[0]
        key = (N, vid.data_ptr())
        if key not in self.rows:
            other = torch.ones(N, dtype=torch.bool, device=q.device)
            other[vid.flatten()] = False
            self.rows = {key: (vid.flatten(), other.nonzero().flatten())}
        vr, orow = self.rows[key]
        St.fwd, St.cfg, St.T, St.F = c.fwd, False, vid.shape[1], vid.shape[0]
        warm, dl = self.sp.warm, self.sp.dense_layers
        if self.force:
            self.sp.warm, self.sp.dense_layers = 0, 0
        try:
            r = self.sp('hunyuan', layer, q[vr], k[vr], v[vr], q[orow], k[orow], v[orow],
                        torch.ones(len(orow), device=q.device))
        finally:
            self.sp.warm, self.sp.dense_layers = warm, dl
        if r is None:
            return None
        o = torch.zeros_like(q)
        o[vr] = r[0].view(len(vr), *q.shape[1:]).to(q.dtype)
        o[orow] = r[1].view(len(orow), *q.shape[1:]).to(q.dtype)
        return o


ATTN_BACKEND = 'fa3'            # packed varlen attention on FlashAttention-3, as in the paper (onereel.attention)
PARTITION = 'balanced'          # K/V gathering partition (onereel.parallel.comm.partition); ablations change it


class Runner:
    def __init__(self, model, res):
        from onereel import attention
        attention.BACKEND = ATTN_BACKEND
        self.model, self.res = model, res
        self.h, self.w = RES[model][res]
        first = SceneLayout([7, 7, 7], halo=0)
        if model == 'hunyuan':
            from onereel.models import hunyuan15 as M
            self.ctx = M.Context(first, height=self.h, width=self.w)
            self.pipe = M.load()
            M.install(self.pipe, self.ctx, timing=True)
            self.n_layers = len(self.pipe.transformer.transformer_blocks)
        elif model == 'wan':
            from onereel.models import wan as M
            from prompt_time_attention import NEG
            self.neg = NEG
            self.ctx = M.Context(first, height=self.h, width=self.w)
            self.pipe = M.load()
            M.install(self.pipe, self.ctx)
            self.n_layers = len(self.pipe.transformer.blocks)
        elif model == 'cogvideox':
            from onereel.models import cogvideox as M
            self.ctx = M.Context(SceneLayout([4, 4, 4], halo=0), height=self.h, width=self.w)
            self.pipe = M.load()
            M.install(self.pipe, self.ctx)
            self.n_layers = len(self.pipe.transformer.transformer_blocks)
        else:
            from onereel.models import minimax_h3 as M
            M.allow_any_duration()
            self.ctx = M.Context(halo=0)
            dev = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
            self.conditioner, self.rest = M.load(dev, dev)
            self.ctx.offload_text_encoder = True             # both models share the GPU: text encoder to CPU after use
            M.install(self.rest.transformer, self.ctx)
            self.n_layers = len(self.rest.transformer.transformer_blocks)
        self.M = M

    def configure(self, method):
        c = self.ctx
        # separate: each scene a dense request; onereel_ulysses / onereel_ring: OneReel's scene attention on USP's
        # Ulysses or Ring (every scene split evenly over the ranks) instead of the K/V gathering chosen by Eq. 2
        c.mode = {'onereel': 'auto', 'onereel_ring': 'ring', 'onereel_gather': 'scene'}.get(method, 'ulysses')
        c.partition = PARTITION
        if method not in SPARSE:
            c.attn_override = None
        elif self.model == 'h3':
            c.attn_override = H3SparseAttn(method, self.n_layers, c)
        else:
            c.attn_override = SparseAttn(self.model, method, self.n_layers, c)
        return c.attn_override

    def encode(self, row):
        """H3 separate: every scene's sentence encoded up front, one text-encoder residency"""
        return self.M.encode(self.conditioner, self.ctx, [[s] for s in row['sentences']])

    def run(self, method, row, steps=20, latent=False, scene=None, encoded=None, frames=None):
        """one generation; scene = j: only sentence j (separate); encoded: H3 text state from encode(); frames: H3
        frame count (default: the layout's frames snapped to 17n + 5)"""
        M, c = self.M, self.ctx
        n, f = len(row['sentences']), SCENE_FRAMES.get(self.model, 7)
        per = row.get('lengths') or [f] * n                       # scene lengths (default: f layout frames each)
        if scene is not None:
            sentences, lengths = [row['sentences'][scene]], [per[scene]]
        elif method.startswith('onereel'):
            sentences, lengths = row['sentences'], per
        else:
            sentences, lengths = [row['prompt']], [sum(per)]
        halo, sides = HALO[self.model] if method.startswith('onereel') and scene is None else (0, 'adj')
        c.frame_memory = None
        if method.startswith('onereel') and scene is None and self.model in FRAME_MEMORY:
            c.frame_memory = FrameMemory(FRAME_MEMORY[self.model], 'all', fallback=(halo, sides))
            halo = 0
        if self.model == 'h3':
            c.halo, c.lengths = halo, None
            frames = frames or M.snap_frames(4 * (sum(lengths) - 1) + 1)
            out = M.generate(self.conditioner, self.rest, c, sentences, frames, self.h, self.w, 0,
                             steps=2 if latent else None, encoded=encoded)  # untimed compile pass (H3 needs >= 2)
            return out['videos'][0] if isinstance(out['videos'], (list, tuple)) else out['videos']
        c.layout = SceneLayout(lengths, halo=halo, sides=sides)
        ot = 'latent' if latent else 'np'
        if self.model == 'hunyuan':
            return M.generate(self.pipe, c, sentences, steps=steps, guidance=CFG['hunyuan'], seed=0, prompt=' '.join(sentences),
                              output_type=ot)
        if self.model == 'cogvideox':
            return M.generate(self.pipe, c, sentences, steps=steps, guidance=CFG['cogvideox'], seed=0, output_type=ot)
        return M.generate(self.pipe, c, sentences, steps=steps, seed=0, guidance=CFG['wan'], negative=self.neg, output_type=ot)


def separate_frames(n_scenes):
    """H3 separate: per-scene frame counts, each 22 or 39 (valid 17n + 5 lengths around one scene's 25 frames), with
    the 39s spread evenly and their number the smallest that keeps the total at or above the joint video's snapped
    frames (Separate never generates fewer frames than Dense / OneReel)"""
    from onereel.models.minimax_h3 import snap_frames
    target = snap_frames(4 * (7 * n_scenes - 1) + 1)
    k = next(k for k in range(n_scenes + 1) if 22 * n_scenes + 17 * k >= target)
    long = {int((i + 0.5) * n_scenes / k) for i in range(k)}
    return [39 if j in long else 22 for j in range(n_scenes)]


def timed(fn):
    D.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    D.barrier()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['hunyuan', 'wan', 'h3', 'cogvideox'], required=True)
    ap.add_argument('--res', type=int, choices=[480, 720], required=True)
    ap.add_argument('--methods', default='dense,radial,svg,solk,onereel')
    ap.add_argument('--ids', default=IDS)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    D.init()
    bench = {r['id']: r for r in json.loads(BENCH.read_text())['prompts']}
    ids = a.ids.split(',')
    R = Runner(a.model, a.res)
    a.out.mkdir(parents=True, exist_ok=True)
    log = a.out / f'{a.model}_{a.res}.jsonl'
    rank0 = D.rank() == 0
    from onereel.utils import save_video

    def write(rec):
        if rank0:
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"[fig1] {rec}", flush=True)

    base = dict(model=a.model, res=a.res, height=R.h, width=R.w, world=D.world(), cfg=CFG.get(a.model))
    for method in a.methods.split(','):
        sp = R.configure(method)
        if method == 'separate':                         # every scene has 7 latent frames: one shape to warm up
            R.run(method, bench['WARM_3'], scene=0)
            if a.model == 'h3':
                R.run(method, bench['WARM_3'], scene=0, frames=22)          # both H3 scene lengths
            for pid in ids:
                row = bench[pid]
                enc, encode_s = timed(lambda: R.encode(row)) if a.model == 'h3' else ([None] * len(row['sentences']), 0.0)
                fr = separate_frames(len(row['sentences'])) if a.model == 'h3' else [None] * len(row['sentences'])
                per = []
                for j in range(len(row['sentences'])):
                    _, s = timed(lambda: R.run(method, row, scene=j, encoded=enc[j], frames=fr[j]))
                    per.append(s)
                write(dict(base, method='separate', id=pid, n_scenes=len(row['sentences']), seconds=encode_s + sum(per),
                           encode_seconds=encode_s, scene_seconds=per, scene_frames=fr,
                           mode=getattr(R.ctx, 'run_mode', None)))
            continue
        R.run(method, bench['WARM_3'])                                               # full warm-up generation
        if a.model != 'h3' or sp is not None:                            # H3: only flex / kernel shapes to warm
            if sp is not None:
                sp.force = True
            for pid in ids:                                                          # per-length compilation, untimed
                R.run(method, bench[pid], steps=1, latent=True)
            if sp is not None:
                sp.force = False
        for i, pid in enumerate(ids):
            row = bench[pid]
            video, s = timed(lambda: R.run(method, row))
            rec = dict(base, method=method, id=pid, n_scenes=len(row['sentences']), seconds=s,
                       mode=getattr(R.ctx, 'run_mode', None), path=getattr(R.ctx, 'path', None),
                       variant=('frame_memory' if R.ctx.frame_memory is not None else 'halo') if method == 'onereel' else None)
            if a.model == 'hunyuan':
                rec.update(denoise_s=sum(R.ctx.forward_ms) / 1e3, vae_s=sum(R.ctx.vae_ms) / 1e3)
            write(rec)
            if i == 0 and rank0 and a.model != 'h3':
                save_video(video, a.out / f'{a.model}_{a.res}_{method}_{pid}.mp4')


if __name__ == '__main__':
    main()
