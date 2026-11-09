"""Time-window attention probes for multi-segment prompts in full-sequence Wan2.2-TI2V-5B.

Tokens are frame-major (index = f*T + h*W' + w), so attention over the video is an F x F grid of
frame blocks. Each prompt is a list of time-ordered segments joined by spaces.

  Q1 self-attention sparsity. At the profile steps (conditional branch) every layer's exact softmax
     is recomputed in query chunks and reduced to
       frame_mass [L, H, F, F]    mean mass a query of frame i puts on key frame j
       tile_mass  [L, H, nT, nT]  the same over 128-token tiles (block-sparse kernel granularity)
       stats      per tau: density of the frame / tile keep sets (fewest blocks covering tau of the
                  block-mean mass, diagonal always kept) and the per-query mass they retain;
                  token-level oracle (keys one query needs) on subsampled queries
  Q2 text-to-time assignment.
       cross_mass [L, H, F, n+1]  mean cross-attention mass per latent frame on each real prompt
                                  token (n, incl. EOS) and on all padding (last column)
       clip_sim   [frames, segments] CLIP similarity of every decoded frame to every segment
     --assign hard: latent frame f only sees the tokens of segment floor(f * nseg / F) (plus EOS and
     padding) in the conditional cross-attention, at every step.

Usage (onereel_metrics env; source scripts/env.sh first):
  python experiments/prompt_time_attention.py --out DIR [--prompts 0,1] [--seeds 0,1,2]
      [--frames 41,81,121] [--steps 50] [--assign none|hard] [--prompt-set short|long] [--shard 0/2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_
from diffusers import WanPipeline
from diffusers.models.transformers.transformer_wan import _get_qkv_projections
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.utils import export_to_video

MODEL = os.path.join(os.environ.get('ONEREEL_MODELS', 'data/models'), 'Wan2.2-TI2V-5B-Diffusers')
CLIP_MODEL = 'openai/clip-vit-base-patch32'
CLIP_REV = 'refs/pr/21'   # safetensors conversion; torch < 2.6 cannot load the .bin
PROMPTS = [
    ['A quiet city street at sunrise, pale pink sky, empty sidewalks and closed shops.',
     'The same street at noon under a bright blue sky, crowded with people and yellow taxis.',
     'The street at dusk, orange sunset light, shop signs switching on.',
     'The street at night, dark sky, glowing neon signs reflected on wet asphalt.'],
    ['A hiker walks along a sunny sandy beach next to turquoise waves.',
     'The hiker walks through a dense green rainforest with ferns and tall trees.',
     'The hiker climbs a snowy mountain ridge in a white blizzard.'],
    ['Close-up of hands chopping fresh carrots and green peppers on a wooden cutting board.',
     'The chopped vegetables sizzle in a black frying pan over a blue gas flame.',
     'A finished colorful stir-fry served on a white plate on a dining table.'],
    ['A green caterpillar crawls slowly along a leaf.',
     'A brown chrysalis hangs motionless from a twig.',
     'A bright orange monarch butterfly opens its wings and flies over purple flowers.'],
    ['A blank white canvas stands on an easel in a sunlit studio.',
     'A painter sketches pencil outlines of a landscape on the canvas.',
     'The canvas is now a finished, vivid oil painting of red poppies under a blue sky.'],
    ['A small white sailboat floats on a calm blue sea under a clear sky.',
     'A violent storm hits, dark clouds, lightning and huge crashing waves around the boat.',
     'The storm clears and a bright rainbow appears over the calm sea.'],
    ['A red sports car speeds along a desert highway between sand dunes.',
     'The car drives into a dark concrete tunnel lit by orange lamps.',
     'The car emerges onto a snowy mountain road surrounded by pine trees.'],
    ['A large clear ice cube sits on a blue ceramic plate.',
     'The ice cube is half melted, water pooling around it.',
     'Only a flat puddle of water remains on the plate.'],
    ['A child builds a tall sandcastle on the beach with a red bucket.',
     'A big wave rushes in and washes the sandcastle away.',
     'The child laughs and starts digging again in the wet sand.'],
    ['A white rocket stands on the launch pad at dawn.',
     'The rocket lifts off with bright flames and billowing smoke.',
     'The rocket flies through black space high above the blue Earth.',
     'The rocket deploys a satellite with golden solar panels.'],
]
PROMPTS_LONG = [   # same segments as PROMPTS, each expanded to ~35-40 words of visual detail
    ['A quiet city street at sunrise, seen from the sidewalk. The sky is pale pink and soft gold, the sidewalks are empty, the shop shutters are still closed, and thin morning mist hangs between the old brick buildings.',
     'The same street at noon under a bright blue cloudless sky. Harsh overhead sunlight casts short shadows, the sidewalks are crowded with people in summer clothes, and yellow taxis and delivery vans fill the road.',
     'The street at dusk. Warm orange sunset light glows low between the buildings, long shadows stretch across the road, the crowd thins out, and the shop signs and street lamps switch on one by one.',
     'The street at night under a dark sky. Glowing pink and blue neon signs reflect on the wet black asphalt after rain, a few people walk with umbrellas, and car headlights leave streaks of light.'],
    ['A hiker with a red backpack walks along a sunny sandy beach. Turquoise waves roll gently onto the shore, white foam washes over the sand, palm trees sway in the warm breeze, and the sky is clear blue.',
     'The hiker walks through a dense green rainforest. Giant ferns and tall moss-covered trees surround the narrow muddy trail, thick vines hang from the canopy, and beams of light pierce through the humid mist.',
     'The hiker climbs a steep snowy mountain ridge in a white blizzard. Strong wind blows snow sideways, the hiker leans forward with ice on the jacket, and jagged grey rocks poke out of the deep snow.'],
    ['Close-up of hands chopping fresh orange carrots and green peppers on a wooden cutting board. A sharp kitchen knife moves quickly, slices pile up neatly, and bright daylight falls across the countertop.',
     'The chopped vegetables sizzle in a black frying pan over a blue gas flame. Oil splatters, steam rises, a wooden spatula tosses the vegetables, and they turn glossy and slightly browned at the edges.',
     'A finished colorful stir-fry is served on a round white plate on a dining table. The vegetables glisten with sauce, sesame seeds are sprinkled on top, and a pair of chopsticks rests beside the plate.'],
    ['A green caterpillar crawls slowly along a broad leaf in a sunny garden. Its soft striped body ripples with each step, tiny legs grip the leaf edge, and it stops to chew a small hole in the leaf.',
     'A brown chrysalis hangs motionless from a thin twig. Its smooth shell is wrapped tightly and shines slightly in the light, the background is blurred green, and nothing moves except a gentle sway in the breeze.',
     'A bright orange monarch butterfly opens its wings and flies over a field of purple flowers. Black veins and white spots cover its wings, it flutters from flower to flower, and the sky behind is clear blue.'],
    ['A blank white canvas stands on a wooden easel in a sunlit studio. Brushes in jars and tubes of paint sit on a table nearby, sunlight streams through a tall window, and the canvas is completely empty.',
     'A painter in an apron sketches pencil outlines of a landscape on the canvas. Thin grey lines mark hills, a horizon and a few flowers, and the hand moves carefully across the white surface.',
     'The canvas is now a finished, vivid oil painting of red poppies under a blue sky. Thick brushstrokes of red, green and blue cover the whole canvas, and the wet paint shines in the sunlight.'],
    ['A small white sailboat floats on a calm blue sea under a clear sky. The water is smooth as glass, the sail hangs loosely, gentle sunlight sparkles on the surface, and a few seagulls glide overhead.',
     'A violent storm hits the sailboat. Dark grey clouds cover the sky, lightning flashes, heavy rain pours down, and huge crashing waves toss the boat up and down as the sail whips in the wind.',
     'The storm clears and a bright rainbow appears over the calm sea. The clouds break apart, golden sunlight returns, the water settles into gentle ripples, and the small boat drifts peacefully.'],
    ['A red sports car speeds along a straight desert highway between tall orange sand dunes. Heat haze shimmers above the asphalt, dust kicks up behind the wheels, and the sky is a hard bright blue.',
     'The car drives into a long dark concrete tunnel lit by rows of orange lamps. The lamps flash past overhead, the walls are grey and stained, and the headlights cut through the dim interior.',
     'The car emerges onto a winding snowy mountain road surrounded by tall pine trees. Fresh white snow covers the ground and the branches, cold mist hangs in the valley, and the car leaves tracks in the snow.'],
    ['A large clear ice cube sits on a blue ceramic plate on a kitchen table. Its sharp edges and smooth faces catch the light, tiny air bubbles are frozen inside, and the plate around it is completely dry.',
     'The ice cube is half melted. Its edges are rounded and glossy, it has shrunk to half its size, and a growing pool of clear water spreads across the blue plate around it.',
     'Only a flat puddle of water remains on the plate. The ice is completely gone, the water surface is still and reflects the ceiling light, and a few droplets cling to the rim of the plate.'],
    ['A child in a yellow shirt builds a tall sandcastle on the beach with a red bucket. The castle has towers and a moat, the child pats the wet sand carefully, and the sea is calm in the background.',
     'A big wave rushes up the beach and washes the sandcastle away. White foam surges over the towers, the walls collapse into the water, and the child jumps back in surprise.',
     'The child laughs and starts digging again in the wet sand. Only a smooth flat patch remains where the castle stood, the red bucket lies on its side, and the child scoops new sand with both hands.'],
    ['A tall white rocket stands on the launch pad at dawn. The steel launch tower stands beside it, floodlights illuminate the rocket, and the sky is turning from dark blue to pale orange on the horizon.',
     'The rocket lifts off with bright flames and billowing white smoke. The engines roar, the tower disappears in the exhaust cloud, and the rocket climbs quickly into the sky on a pillar of fire.',
     'The rocket flies through black space high above the blue Earth. The curved planet glows below with white clouds and oceans, stars fill the background, and the engine flame is a small blue glow.',
     'The rocket deploys a satellite with golden solar panels. The satellite separates slowly, its panels unfold and catch the sunlight, and the rocket stage drifts away against the black sky.'],
]
NEG = ('Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, '
       'JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face')
H, W, CFG, FPS = 704, 1280, 5.0, 24
TAUS = (0.9, 0.95, 0.99)
TILE, CHUNK, QSTRIDE = 128, 1024, 16


class State:
    fwd = 0
    F = T = None
    profile_steps = ()
    topk_steps = ()
    rec = {}
    assign_bias = None

    @classmethod
    def step(cls):
        return (cls.fwd - 1) // 2

    @classmethod
    def cond(cls):
        return (cls.fwd - 1) % 2 == 0


def keep_set(mean, tau):
    order = mean.argsort(-1, descending=True)
    srt = mean.gather(-1, order)
    need = (srt.cumsum(-1) - srt) < tau * mean.sum(-1, keepdim=True)
    return torch.zeros_like(mean, dtype=torch.bool).scatter(-1, order, need)


def qstats(x, tau):
    x = x.flatten().float()
    return {'mean': float(x.mean()), 'p1': float(x.kthvalue(max(1, int(0.01 * x.numel()))).values),
            'frac_below': float((x < tau - 0.05).float().mean())}


def rope(x, cos, sin):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


def profile_self(q, k, topk):
    """q, k: [S, heads, D] after RoPE."""
    F, T = State.F, State.T
    S, Hh, D = q.shape
    nT = math.ceil(S / TILE)
    kt = k.float().permute(1, 2, 0)
    fm = torch.empty(Hh, S, F, device=q.device, dtype=torch.float16)
    tm = torch.empty(Hh, S, nT, device=q.device, dtype=torch.float16)
    tok = {tau: [] for tau in TAUS}
    for s0 in range(0, S, CHUNK):
        qc = q[s0:s0 + CHUNK].float().transpose(0, 1)
        p = (torch.bmm(qc, kt) / math.sqrt(D)).softmax(-1)
        c = p.shape[1]
        fm[:, s0:s0 + c] = p.view(Hh, c, F, T).sum(-1)
        tm[:, s0:s0 + c] = F_.pad(p, (0, nT * TILE - S)).view(Hh, c, nT, TILE).sum(-1)
        if topk:
            cum = p[:, ::QSTRIDE].sort(-1, descending=True).values.cumsum(-1)
            for tau in TAUS:
                tok[tau].append(((cum < tau).sum(-1) + 1).float() / S)
        del p
    out = {}
    fmean = fm.view(Hh, F, T, F).float().mean(2)
    qtile = torch.arange(S, device=q.device) // TILE
    cnt = torch.bincount(qtile, minlength=nT).float()
    tmean = torch.zeros(Hh, nT, nT, device=q.device).index_add_(1, qtile, tm.float()) / cnt[None, :, None]
    eyeF, eyeT = torch.eye(F, dtype=torch.bool, device=q.device), torch.eye(nT, dtype=torch.bool, device=q.device)
    for tau in TAUS:
        kf = keep_set(fmean, tau) | eyeF
        ret_f = (fm.view(Hh, F, T, F).float() * kf[:, :, None, :]).sum(-1)
        kt_ = keep_set(tmean, tau) | eyeT
        ret_t = (tm.float() * kt_[:, qtile, :]).sum(-1)
        out[tau] = {'frame_density': float(kf.float().mean()), 'frame_retained': qstats(ret_f, tau),
                    'tile_density': float(kt_.float().mean()), 'tile_retained': qstats(ret_t, tau)}
        if topk:
            out[tau]['token_density'] = float(torch.cat(tok[tau], 1).mean())
        del ret_f, ret_t
    return fmean.half().cpu(), tmean.half().cpu(), out


def profile_cross(q, k, n_real, bias=None):
    """q: [S, heads, D], k: [512, heads, D] -> [heads, F, n_real + 1] mean mass per latent frame."""
    F, T = State.F, State.T
    S, Hh, D = q.shape
    kt = k.float().permute(1, 2, 0)
    acc = torch.zeros(Hh, F, k.shape[0], device=q.device)
    for s0 in range(0, S, CHUNK):
        qc = q[s0:s0 + CHUNK].float().transpose(0, 1)
        logits = torch.bmm(qc, kt) / math.sqrt(D)
        if bias is not None:
            logits = logits + bias[0, 0, s0:s0 + CHUNK].float()
        p = logits.softmax(-1)
        fidx = torch.arange(s0, s0 + p.shape[1], device=q.device) // T
        acc.index_add_(1, fidx, p)
    acc /= T
    return torch.cat([acc[..., :n_real], acc[..., n_real:].sum(-1, keepdim=True)], -1).half().cpu()


class Probe:
    def __init__(self, layer, kind, orig):
        self.layer, self.kind, self.orig = layer, kind, orig

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        st, cond = State.step(), State.cond()
        if self.kind == 'cross' and cond and State.assign_bias is not None:
            attention_mask = State.assign_bias
        if cond and st in State.profile_steps:
            with torch.no_grad():
                q, k, _ = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
                q, k = attn.norm_q(q), attn.norm_k(k)
                q, k = q.unflatten(2, (attn.heads, -1))[0], k.unflatten(2, (attn.heads, -1))[0]
                r = State.rec.setdefault(st, {})
                if self.kind == 'self':
                    q, k = rope(q[None], *rotary_emb)[0], rope(k[None], *rotary_emb)[0]
                    fmean, tmean, stats = profile_self(q, k, st in State.topk_steps)
                    r.setdefault('frame_mass', []).append(fmean)
                    r.setdefault('tile_mass', []).append(tmean)
                    r.setdefault('stats', []).append(stats)
                else:
                    r.setdefault('cross_mass', []).append(profile_cross(q, k, State.n_real, attention_mask))
        return self.orig(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)


def token_segments(tokenizer, segments):
    """-> (n_real incl. EOS, segment id per real token; -1 for special tokens)."""
    prompt = ' '.join(segments)
    assert prompt_clean(prompt) == prompt, 'prompt changes under prompt_clean; offsets would not match'
    bounds, c = [], 0
    for s in segments:
        bounds.append((c, c + len(s)))
        c += len(s) + 1
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    seg = []
    for a, b in enc['offset_mapping']:
        if b == 0:
            seg.append(-1)
            continue
        seg.append(next(i for i, (lo, hi) in enumerate(bounds) if lo <= b - 1 < hi + 1))
    return len(seg), seg


def build_assign_bias(seg, nseg, device, dtype):
    F, T = State.F, State.T
    fseg = torch.arange(F, device=device) * nseg // F
    tseg = torch.full((512,), -1, device=device)
    tseg[:len(seg)] = torch.tensor(seg, device=device)
    allowed = (tseg[None, :] == fseg[:, None]) | (tseg[None, :] < 0)
    bias = torch.zeros(F, 512, device=device, dtype=dtype).masked_fill(~allowed, float('-inf'))
    return bias.repeat_interleave(T, 0)[None, None]


@torch.no_grad()
def clip_sim(clip, proc, frames, segments):
    tx = proc(text=segments, return_tensors='pt', padding=True, truncation=True).to('cuda')
    t = F_.normalize(clip.get_text_features(**tx).float(), dim=-1)
    sims = []
    for i in range(0, len(frames), 32):
        im = proc(images=[(f * 255).round().astype(np.uint8) for f in frames[i:i + 32]], return_tensors='pt').to('cuda')
        v = F_.normalize(clip.get_image_features(pixel_values=im['pixel_values'].half()).float(), dim=-1)
        sims.append(v @ t.T)
    return torch.cat(sims).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prompts', default=','.join(map(str, range(len(PROMPTS)))))
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--frames', default='81')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--prompt-set', choices=['short', 'long'], default='short')
    ap.add_argument('--assign', choices=['none', 'hard'], default='none')
    ap.add_argument('--no-profile', action='store_true')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--model', default=MODEL)
    a = ap.parse_args()

    jobs = [(p, s, nf) for nf in map(int, a.frames.split(',')) for s in map(int, a.seeds.split(','))
            for p in map(int, a.prompts.split(','))]
    si, sn = map(int, a.shard.split('/'))
    jobs = jobs[si::sn]
    st = a.steps
    State.profile_steps = () if a.no_profile else tuple(sorted({0, st // 10, st // 5, 2 * st // 5, 3 * st // 5, 4 * st // 5, st - 1}))
    State.topk_steps = () if a.no_profile else (st // 5, 3 * st // 5)

    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REV, use_safetensors=True, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    pipe = WanPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16).to('cuda')
    for i, blk in enumerate(pipe.transformer.blocks):
        blk.attn1.set_processor(Probe(i, 'self', blk.attn1.processor))
        blk.attn2.set_processor(Probe(i, 'cross', blk.attn2.processor))
    pipe.transformer.register_forward_pre_hook(lambda m, args: setattr(State, 'fwd', State.fwd + 1))
    cfg = pipe.transformer.config

    for pid, seed, nf in jobs:
        d = a.out / f'p{pid}_s{seed}_f{nf}'
        if (d / 'probe.pt').exists():
            print(f'[skip] {d.name}', flush=True)
            continue
        segs = (PROMPTS_LONG if a.prompt_set == 'long' else PROMPTS)[pid]
        lt = (nf - 1) // pipe.vae_scale_factor_temporal + 1
        State.F = lt // cfg.patch_size[0]
        State.T = (H // pipe.vae_scale_factor_spatial // cfg.patch_size[1]) * (W // pipe.vae_scale_factor_spatial // cfg.patch_size[2])
        State.n_real, seg = token_segments(pipe.tokenizer, segs)
        State.assign_bias = build_assign_bias(seg, len(segs), 'cuda', torch.bfloat16) if a.assign == 'hard' else None
        State.fwd, State.rec = 0, {}
        g = torch.Generator('cuda').manual_seed(seed)
        video = pipe(prompt=' '.join(segs), negative_prompt=NEG, height=H, width=W, num_frames=nf,
                     num_inference_steps=st, guidance_scale=CFG, generator=g, output_type='np').frames[0]
        d.mkdir(parents=True, exist_ok=True)
        export_to_video(list(video), str(d / 'video.mp4'), fps=FPS)
        probe = {'meta': {'prompt_id': pid, 'seed': seed, 'num_frames': nf, 'steps': st, 'assign': a.assign, 'prompt_set': a.prompt_set,
                          'F': State.F, 'T': State.T, 'segments': segs, 'token_segment': seg},
                 'clip_sim': clip_sim(clip, proc, video, segs)}
        for s, r in State.rec.items():
            probe[s] = {key: (torch.stack(v) if key != 'stats' else v) for key, v in r.items()}
        torch.save(probe, d / 'probe.pt')
        s_mid = State.topk_steps[0] if State.topk_steps else None
        if s_mid is not None:
            r = State.rec[s_mid]['stats']
            fd = np.mean([x[0.95]['frame_density'] for x in r])
            td = np.mean([x[0.95]['tile_density'] for x in r])
            kd = np.mean([x[0.95]['token_density'] for x in r])
            print(f'[{d.name}] step {s_mid} tau0.95 density frame={fd:.3f} tile={td:.3f} token={kd:.4f}', flush=True)
        print(f'[done] {d.name}', flush=True)
    (a.out / f'done_shard{si}.json').write_text(json.dumps({'jobs': len(jobs)}))


if __name__ == '__main__':
    main()
