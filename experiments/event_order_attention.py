"""Scene-cut attention on the event-order benchmark (dataset/event_order_bench.json), Wan2.2-TI2V-5B and HunyuanVideo-1.5.

Latent frames are split evenly into the prompt's three sentences (scene j = latent frames [7j, 7j+7) at 81
frames). 'assign' restricts text: Wan's cross-attention lets scene j read only sentence j (plus EOS/padding);
Hunyuan's joint attention does the same for video->text and, symmetrically, lets sentence j's text tokens read
only scene j's video (template tokens stay global). Hunyuan's text tokens are updated by video at every layer,
so under iso they would relay information between scenes: there only sentence j's tokens read video (scene j),
template and negative-prompt tokens read none. The video-video mask is a frame table:
  iso   own scene + latent frame 0 (attention sink)
Settings (every one except baseline and separate also uses 'assign'):
  baseline, assign             dense video-video
  iso_all                      iso at every layer and step
  step_cut{K} / step_mix{K}    iso for steps < K then dense / dense for steps < K then iso
  layer_{shallow,mid,deep}     iso in that third of the layers, every step
  layer_{third}_cut{K}         iso in that third, steps < K only
  soft{K}_{table}              steps < K use a soft table, then dense; tables (all keep own scene + frame 0):
                               prev{m} last m frames of the previous scene, adj{m} the m frames nearest the
                               boundary in both adjacent scenes, key the middle frame of every other scene,
                               stride{k} every k-th latent frame of the clip, isons own scene only (no
                               frame-0 sink); soft20_iso / soft20_isons cut every step
  layerdeep_{table}            the deepest third of the layers use that table at every step
Dynamic (Hunyuan):
  dynK{t}      iso until the scenes' predicted clean latents settle: after step s (>= 1) the cut ends when
               ||x0_s - x0_{s-1}|| / ||x0_s|| < t/100 on 4x-pooled latents (x0 = x_t - sigma * v)
  dynX{p}      iso for 2 steps, then intra-scene and text attention stay dense while cross-scene video blocks
               are kept only where block-mean q/k estimate the top p/100 of each query block's cross-scene
               mass (per layer, head and CFG branch, refreshed every 5 steps)
  dynR         iso for 2 steps, then the window into adjacent scenes grows by one latent frame per step
Hunyuan dense attention runs through flex with the padding mask only (same result as diffusers' SDPA, which
gets a full S x S boolean mask and cannot use the flash kernel); --sdpa-dense restores the original path.
FLOPs: every attention call logs (joint length S, computed block fraction); meta.json stores attention and
transformer FLOPs (4 S^2 D per attention, 24 S D^2 for projections and MLP) and the dense equivalents.
  separate                     upper bound: every sentence generated alone (same seed), clips concatenated
  joint_sep, joint_sep_grope   (Hunyuan) separate inside one sequence: every sentence is encoded alone (MLLM and token
                               refiner), and scene j's video and sentence j's text form an independent block at every
                               layer and step (no sink, no shared template tokens). joint_sep restarts the temporal RoPE
                               index at 0 in every scene; joint_sep_grope keeps the global index
  joint_sep_grope_adj{m}       joint_sep_grope, plus every video frame also reads the m frames nearest the boundary in
                               each adjacent scene (a halo; text stays per sentence)
  joint_sep_grope_{prev,next}{m}  the halo from the previous / next scene only
  joint_sep_grope_all{m}       every scene reads the m frames nearest to it of every other scene
                               (all99: every frame of every other scene, i.e. full video attention with per-scene text)
  joint_sep_grope_adjkf{m}     adjacent scenes as adj{m}; every non-adjacent scene contributes its middle frame
  joint_sep_grope_graph{m}     adjacent scenes as adj{m}, plus the m nearest frames of non-adjacent scenes related to it
                               by an edge of the prompt's scene graph ('edges' in dataset/scene_graph_bench.json)
  joint_sep_grope_adj{m}_subj{A,B}{n}   (Hunyuan, --impl varlen) subject links: at step SUBJ_STEP the attention logits of
                               every scene's video queries to its sentence's subject tokens (the prompt's 'subject') are
                               averaged over the middle layers, smoothed 3x3; the top SUBJ_TOP of each latent frame is the
                               subject mask.
                               From the next step, scene j also reads the subject tokens of n key frames (evenly spaced)
                               of every other scene: A for all of scene j's video queries, B only for its subject queries
                               Wan: every sentence is encoded alone by umT5 and scene j's cross-attention reads only
                               sentence j; self-attention uses the same scene blocks and halo
Saved per run: video.mp4, meta.json.

Usage (onereel_metrics env; source scripts/env.sh first):
  python experiments/event_order_attention.py --model wan|hunyuan --out DIR --settings baseline,step_cut2
      [--ids W1_BCA,L1_CAB] [--seed 0] [--frames 81] [--steps 20] [--cfg 1.0] [--shard 0/2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_
from diffusers.utils import export_to_video
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_time_attention import rope as wan_rope  # noqa: E402

BENCH = Path(__file__).resolve().parents[1] / 'dataset' / 'event_order_bench.json'
GRAPH_BENCH = BENCH.with_name('scene_graph_bench.json')
WAN = dict(id='Wan-AI/Wan2.2-TI2V-5B-Diffusers', H=704, W=1280, fps=24)
HUNYUAN = dict(id='hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v', H=480, W=848, fps=24)
for _d, _env in ((WAN, 'WAN_HW'), (HUNYUAN, 'HUNYUAN_HW')):   # e.g. HUNYUAN_HW=720x1280 overrides the resolution
    if os.environ.get(_env):
        _d['H'], _d['W'] = map(int, os.environ[_env].lower().split('x'))


def local(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)
flex = torch.compile(flex_attention, dynamic=False)


class St:
    fwd = 0
    cfg = False
    setting = 'baseline'
    n_layers = 30
    F = T = Sv = None          # latent frames, tokens per frame, video tokens
    nseg = 3
    masks = {}                 # name -> BlockMask
    mask_cache = {}            # Hunyuan: (encoder length, valid tokens) -> masks
    mask_density = {}          # id(BlockMask) -> computed block fraction
    calls = []                 # (joint length, block fraction) per attention call
    cut_until = 20             # dynK: first step that is no longer cut
    sep_counts = {}            # joint_sep*: CFG branch (cond) -> valid MLLM tokens per sentence copy
    impl = 'flex'              # joint_sep*: 'varlen' packs the independent scenes for flash_attn_varlen (same result);
                               # Wan 'fast': fused q/k norm + RoPE, one flash_attn call per scene on contiguous slices
    rope_cache = {}
    varlen_cache = {}
    valid_text = {}
    subject = ''
    subj_mask = subj_score = None
    subj_tokens = []
    edges = set()              # scene-graph edges of the current prompt, as frozenset({i, j})
    scene_prev = None          # per-scene step-to-step change of v, x0 and x_t (cond branch), logged to meta.json
    probe = None               # callable(kind, layer, q, k, ...) for attention statistics
    override = None            # experiments/sparse_baselines.py: callable(kind, layer, q, k, v, ...) -> attention output or None
    latent_path = None         # Hunyuan: where to save the final latents before VAE decoding (long videos)
    scene_log = []
    compile = False
    x0_prev = None
    x0_log = []
    dyn_cache = {}             # dynX: (layer, cond) -> (step built, BlockMask, density)
    sdpa_dense = False         # dense Hunyuan calls: diffusers SDPA with a full S x S mask (slow) or flex
    assign = False
    wan_bias = None            # Wan cross-attention bias [1, 1, Sv, 512]
    uniform_timestep = True    # Wan: collapse identical per-token timesteps (t2v) before the transformer
    lrope_global_check = False # *_lrope: use global frame indices (must reproduce the global-RoPE path exactly)
    text_seg = None            # Hunyuan: [St] sentence id per encoder token, -1 shared, -2 invalid

    @classmethod
    def step(cls):
        return (cls.fwd - 1) // 2 if cls.cfg else cls.fwd - 1

    @classmethod
    def cond(cls):
        return (cls.fwd - 1) % 2 == 0 if cls.cfg else True


def plan(layer, step):
    """frame-table name for this (layer, step), or None for dense video-video"""
    s, L = St.setting, St.n_layers
    if s in ('baseline', 'assign', 'separate'):
        return None
    if s in SEP_SETTINGS:
        return 'SEP'
    if s == 'iso_all':
        return 'iso'
    m = re.fullmatch(r'step_(cut|mix)(\d+)', s)
    if m:
        early = step < int(m.group(2))
        return 'iso' if early == (m.group(1) == 'cut') else None
    m = re.fullmatch(r'soft(\d+)_(\w+)', s)
    if m:
        return m.group(2) if step < int(m.group(1)) else None
    m = re.fullmatch(r'layerdeep_(\w+)', s)
    if m:
        return m.group(1) if layer >= 2 * L // 3 else None
    if re.fullmatch(r'dynK\d+', s):
        return 'iso' if step < St.cut_until else None
    if re.fullmatch(r'dynX\d+', s):
        return 'iso' if step < 2 else 'DYNX'
    if s == 'dynR':
        return 'iso' if step < 2 else f'ramp{step - 1}'
    m = re.fullmatch(r'layer_(shallow|mid|deep)(?:_cut(\d+))?', s)
    if m:
        third = {'shallow': 0, 'mid': 1, 'deep': 2}[m.group(1)]
        in_third = third * L // 3 <= layer < (third + 1) * L // 3
        in_steps = m.group(2) is None or step < int(m.group(2))
        return 'iso' if in_third and in_steps else None
    raise ValueError(s)


SEP_SETTINGS = ('joint_sep', 'joint_sep_grope') + \
    tuple(f'joint_sep_grope_{d}{m}' for d in ('adj', 'prev', 'next', 'all', 'adjkf', 'graph', 'rel', 'anchor') for m in (1, 2, 4)) + \
    ('joint_sep_grope_all99',) + \
    tuple(f'joint_sep_grope_adj2_subj{v}{n}' for v in 'AB' for n in (1, 2, 3)) + \
    tuple(f'joint_sep_grope_adj{m}_lrope' for m in (1, 2, 4)) + \
    tuple(f'joint_sep_grope_{d}{m}_txthalo' for d in ('adj', 'next', 'prev') for m in (1, 2, 4))
SUBJ_STEP, SUBJ_TOP, SUBJ_LAYERS = 3, 0.2, (18, 36)


def subj_cfg():
    m = re.fullmatch(r'joint_sep_grope_\w+?_subj([AB])(\d+)', St.setting)
    return (m.group(1), int(m.group(2))) if m else None


def halo_of(j, frames):
    """joint_sep_grope_{adj,prev,next}{m}: frames of the previous and/or next scene that scene j also reads.
    rel{m}: the k frames nearest to scene j of every scene related to j by a graph edge (adjacent or not, unrelated
    neighbours excluded). anchor{m}: of related adjacent scenes the k nearest frames, plus the k middle frames of the
    anchor of j's component (its earliest scene) when that anchor is not adjacent; at most three sources per scene"""
    m = re.match(r'joint_sep_grope_(adjkf|adj|prev|next|all|graph|rel|anchor)(\d+)', St.setting)
    if not m:
        return []
    d, k = m.group(1), int(m.group(2))
    near = lambda i: frames[i][-k:] if i < j else frames[i][:k]        # the k frames of scene i nearest to scene j
    related = lambda i: frozenset((i, j)) in St.edges
    if d == 'rel':
        return [f for i in range(len(frames)) if i != j and related(i) for f in near(i)]
    if d == 'anchor':
        out = [f for i in (j - 1, j + 1) if 0 <= i < len(frames) and related(i) for f in near(i)]
        comp, todo = {j}, [j]
        while todo:
            x = todo.pop()
            for y in range(len(frames)):
                if y not in comp and frozenset((x, y)) in St.edges:
                    comp.add(y)
                    todo.append(y)
        a = min(comp)
        if a != j and abs(a - j) > 1:
            mid = len(frames[a]) // 2
            out += frames[a][max(0, mid - k // 2):max(0, mid - k // 2) + k]
        return out
    out = []
    for i in range(len(frames)):
        if i == j:
            continue
        adjacent = abs(i - j) == 1
        if d == 'all' or (adjacent and d in ('adj', 'adjkf', 'graph')) or (d == 'prev' and i == j - 1) or (d == 'next' and i == j + 1):
            out += near(i)
        elif d == 'adjkf':
            out.append(frames[i][len(frames[i]) // 2])
        elif d == 'graph' and frozenset((i, j)) in St.edges:
            out += near(i)
    return out


def frame_table(F, nseg, name='iso'):
    seg = torch.arange(F) * nseg // F
    t = seg[:, None] == seg[None, :]
    t[:, 0] = True
    frames = [(seg == j).nonzero().flatten() for j in range(nseg)]
    if name == 'isons':                      # own scene only, no frame-0 sink
        return seg[:, None] == seg[None, :]
    if name.startswith('ramp'):
        idx = torch.arange(F)
        return t | ((idx[:, None] - idx[None, :]).abs() <= int(name[4:]))
    if name.startswith('prev'):
        m = int(name[4:])
        for i in range(F):
            if seg[i] > 0:
                t[i, frames[seg[i] - 1][-m:]] = True
    elif name.startswith('adj'):
        m = int(name[3:])
        for i in range(F):
            if seg[i] > 0:
                t[i, frames[seg[i] - 1][-m:]] = True
            if seg[i] + 1 < nseg:
                t[i, frames[seg[i] + 1][:m]] = True
    elif name == 'key':
        mids = torch.stack([f[len(f) // 2] for f in frames])
        t[:, mids] = True
    elif name.startswith('stride'):
        t[:, ::int(name[6:])] = True
    return t


def table_names():
    if St.setting in SEP_SETTINGS:
        return []
    m = re.fullmatch(r'(?:soft\d+|layerdeep)_(\w+)', St.setting)
    if St.setting == 'dynR':
        return ['iso'] + [f'ramp{w}' for w in range(1, 19)]
    return ['iso'] + ([m.group(1)] if m and m.group(1) != 'iso' else [])


def scene_of_frame():
    return torch.arange(St.F) * St.nseg // St.F


# ------------------------------------------------------------------ Wan
def wan_setup(pipe):
    from diffusers.models.transformers.transformer_wan import _get_qkv_projections

    class Proc:
        def __init__(self, layer, kind, orig):
            self.layer, self.kind, self.orig = layer, kind, orig

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
            if self.kind == 'cross':
                if St.probe:
                    qx = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
                    kx = attn.norm_k(attn.to_k(encoder_hidden_states)).unflatten(2, (attn.heads, -1))
                    St.probe('wan_cross', self.layer, qx[0], kx[0])
                if St.setting in SEP_SETTINGS:              # scene j reads only its own sentence encoding
                    L, T = St.wan_text_len, St.T
                    return torch.cat([self.orig(attn, hidden_states[:, lo * T:hi * T], encoder_hidden_states[:, j * L:(j + 1) * L],
                                                None, rotary_emb) for j, (lo, hi) in enumerate(scene_ranges())], 1)
                if St.assign and St.cond():
                    attention_mask = St.wan_bias
                return self.orig(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
            name = plan(self.layer, St.step())
            if St.impl == 'fast' and name in (None, 'SEP'):
                return wan_fast_self(attn, hidden_states, rotary_emb, name, self.layer)
            if name is None:
                return self.orig(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
            q, k, v = _get_qkv_projections(attn, hidden_states, None)
            q, k = attn.norm_q(q), attn.norm_k(k)
            q, k, v = (x.unflatten(2, (attn.heads, -1)) for x in (q, k, v))
            q, k = wan_rope(q, *rotary_emb), wan_rope(k, *rotary_emb)
            if name == 'SEP' and St.impl == 'varlen':
                from flash_attn import flash_attn_varlen_func
                T, pl = St.T, wan_varlen_plan()
                qp = torch.cat([q[0, lo * T:hi * T] for lo, hi in pl['video']]).to(torch.bfloat16)
                kv = [torch.cat([x[0, lo * T:hi * T]] + [x[0, f * T:(f + 1) * T] for f in h]) for (lo, hi), h in zip(pl['video'], pl['halo'])
                      for x in (k, v)]
                kp, vp = torch.cat(kv[0::2]).to(torch.bfloat16), torch.cat(kv[1::2]).to(torch.bfloat16)
                o = flash_attn_varlen_func(qp, kp, vp, pl['cu_q'], pl['cu_k'], pl['max_q'], pl['max_k'])
                return attn.to_out[1](attn.to_out[0](o.flatten(1, 2)[None].type_as(hidden_states)))
            Sp = St.masks[name].shape[-1]
            q, k, v = (F_.pad(x.transpose(1, 2), (0, 0, 0, Sp - St.Sv)) for x in (q, k, v))
            o = flex(q, k, v, block_mask=St.masks[name])[:, :, :St.Sv]
            return attn.to_out[1](attn.to_out[0](o.transpose(1, 2).flatten(2, 3).type_as(hidden_states)))
    Proc.__call__ = torch.compiler.disable(Proc.__call__)
    for i, blk in enumerate(pipe.transformer.blocks):
        blk.attn1.set_processor(Proc(i, 'self', blk.attn1.processor))
        blk.attn2.set_processor(Proc(i, 'cross', blk.attn2.processor))
    St.n_layers = len(pipe.transformer.blocks)

    def uniform_timestep(m, args, kwargs):
        t = kwargs.get('timestep')                # Wan2.2 TI2V passes one timestep per token; in t2v they are all equal
        if t is not None and t.ndim == 2 and bool((t == t[:, :1]).all()):
            kwargs['timestep'] = t[:, 0]          # the model's broadcast path: same result, no [S, 6, D] fp32 modulation
        return args, kwargs
    if St.uniform_timestep:
        pipe.transformer.register_forward_pre_hook(uniform_timestep, with_kwargs=True)
    if St.compile:
        torch._dynamo.config.cache_size_limit = 512
        torch.backends.cudnn.benchmark = True
        for blk in pipe.transformer.blocks:
            blk.compile(dynamic=False)
        pipe.vae.decoder.compile(dynamic=False)


torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)   # wan_qk: one graph per length


@torch.compile(fullgraph=True, dynamic=False)
def wan_qk(q, k, wq, wk, eps: float, cos, sin, heads: int):
    """RMSNorm (as torch.nn.RMSNorm) and the interleaved RoPE of diffusers' Wan attention, fused"""
    def norm(x, w):
        return F_.rms_norm(x, (x.shape[-1],), w, eps)

    def rope(x):
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        c, s_ = cos[..., 0::2], sin[..., 1::2]
        return torch.stack([x1 * c - x2 * s_, x1 * s_ + x2 * c], -1).flatten(-2).type_as(x)
    return rope(norm(q, wq).unflatten(2, (heads, -1))), rope(norm(k, wk).unflatten(2, (heads, -1)))


def wan_fast_self(attn, hidden_states, rotary_emb, name, layer=0):
    """Wan self-attention, dense or per scene; a scene's keys (scene plus halo) form a contiguous token range"""
    from flash_attn import flash_attn_func
    q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
    q, k = wan_qk(q, k, attn.norm_q.weight, attn.norm_k.weight, attn.norm_q.eps, *rotary_emb, attn.heads)
    v = v.unflatten(2, (attn.heads, -1))
    q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
    if St.probe:
        St.probe('wan_self', layer, q[0], k[0])
    r = St.override('wan', layer, q[0], k[0], v[0]) if St.override and name is None else None
    if r is not None:
        return attn.to_out[1](attn.to_out[0](r[None].type_as(hidden_states)))
    if name is None:
        o = flash_attn_func(q, k, v)
    else:
        T, pl = St.T, wan_varlen_plan()
        outs = []
        for (lo, hi), h in zip(pl['video'], pl['halo']):
            a, b = min([lo] + h), max([hi - 1] + h) + 1
            assert b - a == hi - lo + len(h), 'halo must be contiguous with its scene'
            outs.append(flash_attn_func(q[:, lo * T:hi * T], k[:, a * T:b * T], v[:, a * T:b * T]))
        o = torch.cat(outs, 1)
    return attn.to_out[1](attn.to_out[0](o.flatten(2, 3).type_as(hidden_states)))


def wan_varlen_plan():
    """Wan self-attention: one varlen sequence per scene, keys extended by its halo frames (no text tokens)"""
    key = ('wan', St.setting, St.F)
    if key not in St.varlen_cache:
        ranges = scene_ranges()
        frames = [list(range(lo, hi)) for lo, hi in ranges]
        halo = [halo_of(j, frames) for j in range(St.nseg)]
        lq = [(hi - lo) * St.T for lo, hi in ranges]
        lk = [q + len(h) * St.T for q, h in zip(lq, halo)]
        cu = lambda x: torch.tensor([0] + np.cumsum(x).tolist(), dtype=torch.int32, device='cuda')
        St.varlen_cache[key] = dict(video=ranges, halo=halo, cu_q=cu(lq), cu_k=cu(lk), max_q=max(lq), max_k=max(lk))
    return St.varlen_cache[key]


def scene_ranges():
    sc = scene_of_frame()
    return [((sc == j).nonzero()[0, 0].item(), (sc == j).nonzero()[-1, 0].item() + 1) for j in range(St.nseg)]


def sep_table(F, nseg):
    """joint_sep*: own scene only (no sink), plus the halo frames of joint_sep_grope_adj{m}"""
    t = frame_table(F, nseg, 'isons')
    sc = torch.arange(F) * nseg // F
    frames = [(sc == j).nonzero().flatten().tolist() for j in range(nseg)]
    for j, fr in enumerate(frames):
        for f in halo_of(j, frames):
            t[fr, f] = True
    return t


def wan_prepare(pipe, sentences, frames):
    from prompt_time_attention import token_segments
    cfg = pipe.transformer.config
    St.F = ((frames - 1) // pipe.vae_scale_factor_temporal + 1) // cfg.patch_size[0]
    St.T = (WAN['H'] // pipe.vae_scale_factor_spatial // cfg.patch_size[1]) * (WAN['W'] // pipe.vae_scale_factor_spatial // cfg.patch_size[2])
    St.Sv = St.F * St.T
    St.sentences = sentences
    _, seg = token_segments(pipe.tokenizer, sentences)
    fseg = scene_of_frame().cuda()
    tseg = torch.full((512,), -1, device='cuda')
    tseg[:min(len(seg), 512)] = torch.tensor(seg[:512], device='cuda')     # the joint prompt is only used by the old settings
    allowed = (tseg[None, :] == fseg[:, None]) | (tseg[None, :] < 0)
    bias = torch.zeros(St.F, 512, device='cuda', dtype=torch.bfloat16).masked_fill(~allowed, float('-inf'))
    St.wan_bias = bias.repeat_interleave(St.T, 0)[None, None]
    F, T, S = St.F, St.T, St.Sv
    Sp = math.ceil(S / 128) * 128
    St.masks = {}
    St.varlen_cache = {}
    St.sep_counts = {True: [0] * St.nseg, False: [0] * St.nseg}
    if St.impl in ('fast', 'varlen') and (St.setting in SEP_SETTINGS or St.setting in ('baseline', 'separate')):
        return                                   # these paths do not use the flex mask, whose dense build is O(S^2) memory
    for name in (['SEP'] if St.setting in SEP_SETTINGS else table_names()):
        tab = (sep_table(F, St.nseg) if name == 'SEP' else frame_table(F, St.nseg, name)).cuda()

        def mod(b, h, q, kv, tab=tab):
            return (kv < S) & (tab[torch.clamp(q // T, max=F - 1), torch.clamp(kv // T, max=F - 1)] | (q >= S))
        St.masks[name] = create_block_mask(mod, None, None, Sp, Sp, device='cuda')


def wan_generate(pipe, prompt, frames, steps, seed, cfg):
    g = torch.Generator('cuda').manual_seed(seed)
    neg = None
    if cfg > 1:
        from prompt_time_attention import NEG
        neg = NEG
    if St.setting in SEP_SETTINGS:
        St.wan_text_len = 512
        with torch.no_grad():
            e = torch.cat([pipe._get_t5_prompt_embeds(prompt=s, max_sequence_length=512, device='cuda') for s in St.sentences], 1)
            ne = None
            if cfg > 1:          # the negative prompt once per scene: scene j's cross-attention reads chunk j in both passes
                ne = pipe._get_t5_prompt_embeds(prompt=neg, max_sequence_length=512, device='cuda').repeat(1, len(St.sentences), 1)
        return pipe(prompt_embeds=e, negative_prompt_embeds=ne, height=WAN['H'], width=WAN['W'], num_frames=frames,
                    num_inference_steps=steps, guidance_scale=cfg, generator=g, output_type='np').frames[0]
    return pipe(prompt=prompt, negative_prompt=neg, height=WAN['H'], width=WAN['W'], num_frames=frames,
                num_inference_steps=steps, guidance_scale=cfg, generator=g, output_type='np').frames[0]


# ------------------------------------------------------------------ Hunyuan
def hunyuan_setup(pipe):
    class Proc:
        def __init__(self, layer, orig):
            self.layer, self.orig = layer, orig

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
            name = plan(self.layer, St.step())
            assign = St.assign and St.cond()          # the negative-prompt pass has no sentences to assign
            if name is None and not assign and St.sdpa_dense and St.impl != 'varlen':
                St.calls.append((hidden_states.shape[1] + encoder_hidden_states.shape[1], 1.0))
                return self.orig(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
            from diffusers.models.embeddings import apply_rotary_emb
            varlen = St.impl == 'varlen' and (St.setting in SEP_SETTINGS or St.setting == 'baseline')
            if St.setting == 'joint_sep':
                image_rotary_emb = local_rope(image_rotary_emb)
            q, k, v = (f(hidden_states).unflatten(2, (attn.heads, -1)) for f in (attn.to_q, attn.to_k, attn.to_v))
            q, k = attn.norm_q(q), attn.norm_k(k)
            lrope = varlen and St.setting.endswith('_lrope') and St.cond() is not None
            if not lrope:
                q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
                k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
            eq, ek, ev = (f(encoder_hidden_states).unflatten(2, (attn.heads, -1)) for f in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            if St.probe:
                St.probe('hunyuan', self.layer, q[0], k[0], eq[0], ek[0], attention_mask[0])
            if St.override:
                r = St.override('hunyuan', self.layer, q[0], k[0], v[0], eq[0], ek[0], ev[0], attention_mask[0])
                if r is not None:
                    return attn.to_out[1](attn.to_out[0](r[0][None].to(q.dtype))), attn.to_add_out(r[1][None].to(q.dtype))
            if varlen:
                if St.setting == 'baseline':
                    St.valid_text[St.cond()] = St.valid_text.get(St.cond()) or int(attention_mask[0].sum())
                if subj_cfg():
                    subj_detect(self.layer, q[0], ek[0])
                    if St.subj_mask is not None and St.step() > SUBJ_STEP:
                        return self.varlen_subj(attn, q[0], k[0], v[0], eq[0], ek[0], ev[0])
                if lrope:
                    return self.varlen_lrope(attn, q[0], k[0], v[0], eq[0], ek[0], ev[0], image_rotary_emb)
                return self.varlen(attn, q[0], k[0], v[0], eq[0], ek[0], ev[0])
            q, k, v = (torch.cat([a, b], 1) for a, b in ((q, eq), (k, ek), (v, ev)))
            valid = attention_mask[0].bool()
            mkey = (encoder_hidden_states.shape[1], int(valid.sum()), St.cond())
            if mkey not in St.mask_cache:
                hunyuan_build_masks(valid)
                St.mask_cache[mkey] = St.masks
            S = q.shape[1]
            if name == 'DYNX':
                base = St.mask_cache[mkey][('dense', assign)]
                Sp = base.shape[-1]
                q, k, v = (F_.pad(x.transpose(1, 2), (0, 0, 0, Sp - S)) for x in (q, k, v))
                key = (self.layer, St.cond())
                hit = St.dyn_cache.get(key)
                if hit is None or St.step() - hit[0] >= 5:
                    mask, dens = dynx_mask(q[0], k[0], base, S)
                    hit = (St.step(), mask, dens)
                    St.dyn_cache[key] = hit
                mask, dens = hit[1], hit[2]
            else:
                mask = St.mask_cache[mkey][(name or 'dense', assign)]
                dens = St.mask_density[id(mask)]
                Sp = mask.shape[-1]
                q, k, v = (F_.pad(x.transpose(1, 2), (0, 0, 0, Sp - S)) for x in (q, k, v))
            St.calls.append((S, dens))
            o = flex(q, k, v, block_mask=mask)[:, :, :S].transpose(1, 2).flatten(2, 3).to(q.dtype)
            hs, ehs = o[:, :St.Sv], o[:, St.Sv:]
            hs = attn.to_out[1](attn.to_out[0](hs))
            ehs = attn.to_add_out(ehs)
            return hs, ehs

        def varlen(self, attn, q, k, v, eq, ek, ev):
            """per scene, one flash_attn varlen sequence for the video queries (keys: scene video, halo frames, sentence)
            and one without halo for the sentence queries, matching the joint_sep* block mask"""
            from flash_attn import flash_attn_varlen_func
            T, Le = St.T, eq.shape[0]
            plan_ = varlen_plan()
            def pack(a, ea, kv):
                parts = []
                for (lo, hi), halo, (t0, t1) in zip(plan_['video'], plan_['halo'], plan_['text']):
                    vid, txt = a[lo * T:hi * T], ea[t0:t1]
                    if kv:
                        shared = St.setting.endswith('_txthalo')
                        parts += [vid] + [a[f * T:(f + 1) * T] for f in halo] + [txt] + ([vid, txt] if halo and not shared else [])
                    else:
                        parts += [vid, txt]
                return torch.cat(parts).to(torch.bfloat16)
            o = flash_attn_varlen_func(pack(q, eq, False), pack(k, ek, True), pack(v, ev, True), plan_['cu_q'], plan_['cu_k'],
                                       plan_['max_q'], plan_['max_k'])
            hs, ehs = [], []
            for (lo, hi), (t0, t1), piece in zip(plan_['video'], plan_['text'], o.split(plan_['len_scene'])):
                hs.append(piece[:(hi - lo) * T])
                ehs.append(piece[(hi - lo) * T:])
            ehs.append(o.new_zeros(Le - plan_['text'][-1][1], *o.shape[1:]))
            St.calls.append((St.Sv + Le, plan_['flops'] / (St.Sv + Le) ** 2))
            hs = attn.to_out[1](attn.to_out[0](torch.cat(hs).flatten(1, 2)[None].to(q.dtype)))
            ehs = attn.to_add_out(torch.cat(ehs).flatten(1, 2)[None].to(q.dtype))
            return hs, ehs
        def varlen_lrope(self, attn, q, k, v, eq, ek, ev, rot):
            """varlen with per-scene temporal RoPE: q, k arrive unrotated; every packed sequence is rotated in its scene's
            local frame index (1 + previous-halo frames before the scene, then the scene, then the next halo), so a halo
            frame read by two scenes gets a different index in each; sentence tokens stay unrotated"""
            from flash_attn import flash_attn_varlen_func
            from diffusers.models.embeddings import apply_rotary_emb
            T, Le, Sv = St.T, eq.shape[0], St.Sv
            plan_ = varlen_plan()
            key = ('lrope', St.setting, St.cond(), St.F, Le, St.lrope_global_check)
            if key not in St.varlen_cache:
                ar = lambda a, b: torch.arange(a, b, device=q.device)
                qi, qr, ki, kr = [], [], [], []
                for (lo, hi), halo, (t0, t1) in zip(plan_['video'], plan_['halo'], plan_['text']):
                    off = 0 if St.lrope_global_check else lo - sum(f < lo for f in halo) - 1
                    tok = lambda fs: torch.cat([ar(f * T, (f + 1) * T) for f in fs]) if fs else ar(0, 0)
                    loc = lambda fs: torch.cat([ar((f - off) * T, (f - off + 1) * T) for f in fs]) if fs else ar(0, 0)
                    fr, txt, none = list(range(lo, hi)), ar(Sv + t0, Sv + t1), torch.full((t1 - t0,), -1, device=q.device)
                    qi += [tok(fr), txt]
                    qr += [loc(fr), none]
                    if halo:
                        ki += [tok(fr), tok(halo), txt, tok(fr), txt]
                        kr += [loc(fr), loc(halo), none, loc(fr), none]
                    else:
                        ki += [tok(fr), txt]
                        kr += [loc(fr), none]
                St.varlen_cache[key] = tuple(torch.cat(x) for x in (qi, qr, ki, kr))
            qi, qr, ki, kr = St.varlen_cache[key]
            cos, sin = rot

            def rotate(x, ridx):
                vid = ridx >= 0
                out = x.clone()
                out[vid] = apply_rotary_emb(x[vid][None], (cos[ridx[vid]], sin[ridx[vid]]), sequence_dim=1)[0].to(x.dtype)
                return out
            fq, fk, fv = (torch.cat([a, b]) for a, b in ((q, eq), (k, ek), (v, ev)))
            o = flash_attn_varlen_func(rotate(fq[qi], qr).to(torch.bfloat16), rotate(fk[ki], kr).to(torch.bfloat16), fv[ki].to(torch.bfloat16),
                                       plan_['cu_q'], plan_['cu_k'], plan_['max_q'], plan_['max_k'])
            out = o.new_zeros(Sv + Le, *o.shape[1:])
            out[qi] = o
            St.calls.append((Sv + Le, plan_['flops'] / (Sv + Le) ** 2))
            hs = attn.to_out[1](attn.to_out[0](out[:Sv].flatten(1, 2)[None].to(q.dtype)))
            ehs = attn.to_add_out(out[Sv:].flatten(1, 2)[None].to(q.dtype))
            return hs, ehs

        def varlen_subj(self, attn, q, k, v, eq, ek, ev):
            """subject links: gathered query / key index sequences (see subj_plan)"""
            from flash_attn import flash_attn_varlen_func
            pl = subj_plan(eq.shape[0])
            fq, fk, fv = (torch.cat([a, b]).to(torch.bfloat16) for a, b in ((q, eq), (k, ek), (v, ev)))
            o = flash_attn_varlen_func(fq[pl['q']], fk[pl['k']], fv[pl['k']], pl['cu_q'], pl['cu_k'], pl['max_q'], pl['max_k'])
            out = fq.new_zeros(fq.shape)
            out[pl['q']] = o
            St.calls.append((fq.shape[0], pl['flops'] / fq.shape[0] ** 2))
            hs = attn.to_out[1](attn.to_out[0](out[:St.Sv].flatten(1, 2)[None].to(q.dtype)))
            ehs = attn.to_add_out(out[St.Sv:].flatten(1, 2)[None].to(q.dtype))
            return hs, ehs
    Proc.__call__ = torch.compiler.disable(Proc.__call__)
    for i, blk in enumerate(pipe.transformer.transformer_blocks):
        blk.attn.set_processor(Proc(i, blk.attn.processor))
    St.n_layers = len(pipe.transformer.transformer_blocks)
    pipe.transformer.context_embedder = SplitRefiner(pipe.transformer.context_embedder)
    if St.compile:
        torch._dynamo.config.cache_size_limit = 512
        for blk in pipe.transformer.transformer_blocks:
            blk.compile(dynamic=False)


def subj_detect(layer, q, ek):
    """accumulate video-to-subject-token attention logits at SUBJ_STEP (cond branch, middle layers); then build the mask"""
    if St.subj_mask is not None or not St.cond() or St.step() != SUBJ_STEP:
        return
    if SUBJ_LAYERS[0] <= layer < SUBJ_LAYERS[1]:
        counts = St.sep_counts[True]
        off = np.cumsum([0] + counts[:-1])
        T = St.T
        score = torch.zeros(St.Sv, device=q.device)
        for j, (lo, hi) in enumerate(scene_ranges()):
            idx = torch.tensor([off[j] + t for t in St.subj_tokens[j]], device=q.device)
            ks = ek[idx].float()                                         # [n, H, D]
            score[lo * T:hi * T] = torch.einsum('shd,nhd->s', q[lo * T:hi * T].float(), ks) / (ks.shape[0] * ks.shape[1])
        St.subj_score = score if St.subj_score is None else St.subj_score + score
    if layer == SUBJ_LAYERS[1] - 1:
        h, w = HUNYUAN['H'] // 16, HUNYUAN['W'] // 16
        sc = F_.avg_pool2d(St.subj_score.view(St.F, 1, h, w), 3, 1, 1, count_include_pad=False).view(St.F, St.T)
        thr = sc.quantile(1 - SUBJ_TOP, dim=1, keepdim=True)
        St.subj_mask = sc >= thr                                         # [F, T]


def subj_plan(Le):
    """index sequences: per scene, video queries (B: split into subject / other) with keys = scene video + halo +
    sentence (+ other scenes' key-frame subject tokens), and sentence queries with keys = scene video + sentence"""
    key = ('subj', St.setting, St.cond(), Le)
    if key in St.varlen_cache:
        return St.varlen_cache[key]
    var, n_kf = subj_cfg()
    T, Sv = St.T, St.Sv
    counts = St.sep_counts[St.cond()]
    off = np.cumsum([0] + counts[:-1])
    ranges = scene_ranges()
    frames = [list(range(lo, hi)) for lo, hi in ranges]
    mask = St.subj_mask.view(-1)
    ar = lambda a, b: torch.arange(a, b, device='cuda')
    kf = [[fr[int(round(x))] for x in np.linspace(0, len(fr) - 1, n_kf + 2)[1:-1]] for fr in frames]
    subj_tok = [torch.cat([ar(f * T, (f + 1) * T)[mask[f * T:(f + 1) * T]] for f in kfs]) for kfs in kf]
    qs, ks = [], []
    for j, (lo, hi) in enumerate(ranges):
        vid = ar(lo * T, hi * T)
        halo = torch.cat([ar(f * T, (f + 1) * T) for f in halo_of(j, frames)] or [vid[:0]])
        txt = Sv + ar(off[j], off[j] + counts[j])
        extra = torch.cat([subj_tok[i] for i in range(len(ranges)) if i != j])
        base_k = torch.cat([vid, halo, txt])
        if var == 'A':
            qs.append(vid); ks.append(torch.cat([base_k, extra]))
        else:
            sel = mask[lo * T:hi * T]
            qs += [vid[sel], vid[~sel]]; ks += [torch.cat([base_k, extra]), base_k]
        qs.append(txt); ks.append(torch.cat([vid, txt]))
    lq, lk = [len(x) for x in qs], [len(x) for x in ks]
    cu = lambda x: torch.tensor([0] + np.cumsum(x).tolist(), dtype=torch.int32, device='cuda')
    St.varlen_cache[key] = dict(q=torch.cat(qs), k=torch.cat(ks), cu_q=cu(lq), cu_k=cu(lk), max_q=max(lq), max_k=max(lk),
                                flops=float(sum(a * b for a, b in zip(lq, lk))), extra=[len(x) for x in subj_tok])
    return St.varlen_cache[key]


def save_subject_overlay(video, path, per_scene=3):
    """frames of every scene with the detected subject mask in red"""
    import cv2
    h, w = HUNYUAN['H'] // 16, HUNYUAN['W'] // 16
    m = St.subj_mask.view(St.F, h, w).float().cpu().numpy()
    tiles = []
    for lo, hi in scene_ranges():
        for f in np.linspace(lo, hi - 1, per_scene).round().astype(int):
            img = (video[min(4 * f, len(video) - 1)] * 255).astype(np.uint8).copy()
            mk = cv2.resize(m[f], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            img[mk] = (0.5 * img[mk] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            tiles.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    rows = [np.concatenate(tiles[i:i + per_scene], 1) for i in range(0, len(tiles), per_scene)]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def subject_tokens(pipe, sentences, subject):
    """per sentence, the MLLM token positions (after crop_start) that overlap the subject phrase"""
    from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import format_text_input
    tok, out = pipe.tokenizer, []
    for s in sentences:
        text = tok.apply_chat_template(format_text_input([s], pipe.system_message), add_generation_prompt=True, tokenize=False)
        text = text[0] if isinstance(text, list) else text
        a = text.index(s) + s.lower().index(subject.lower())
        b = a + len(subject)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        out.append([i - pipe.prompt_template_encode_start_idx for i, (x, y) in enumerate(enc['offset_mapping']) if x < b and y > a])
    return out


def varlen_plan():
    """scene frame ranges, halo frames, sentence token ranges and cu_seqlens for the current CFG branch (cached)"""
    key = (St.setting, St.cond(), St.F)
    if key in St.varlen_cache:
        return St.varlen_cache[key]
    if St.setting == 'baseline':                # one dense sequence: all video plus the valid text tokens
        n = St.valid_text[St.cond()]
        L = St.Sv + n
        cu = torch.tensor([0, L], dtype=torch.int32, device='cuda')
        St.varlen_cache[key] = dict(video=[(0, St.F)], halo=[[]], text=[(0, n)], len_q=[L], len_scene=[L], cu_q=cu, cu_k=cu,
                                    max_q=L, max_k=L, flops=float(L * L))
        return St.varlen_cache[key]
    counts = St.sep_counts[St.cond()]
    sc = scene_of_frame()
    frames = [(sc == j).nonzero().flatten().tolist() for j in range(St.nseg)]
    video, halo, text, lq, lk, ls = [], [], [], [], [], []
    t = 0
    for j, fr in enumerate(frames):
        video.append((fr[0], fr[-1] + 1))
        h = halo_of(j, frames)
        halo.append(h)
        text.append((t, t + counts[j]))
        t += counts[j]
        nv = len(fr) * St.T
        ls.append(nv + counts[j])
        if h and St.setting.endswith('_txthalo'):   # sentence queries share the video queries' keys, halo included
            lq.append(nv + counts[j])
            lk.append(nv + len(h) * St.T + counts[j])
        elif h:                                 # video queries see the halo, sentence queries do not
            lq += [nv, counts[j]]
            lk += [nv + len(h) * St.T + counts[j], nv + counts[j]]
        else:
            lq.append(nv + counts[j])
            lk.append(nv + counts[j])
    cu = lambda x: torch.tensor([0] + np.cumsum(x).tolist(), dtype=torch.int32, device='cuda')
    St.varlen_cache[key] = dict(video=video, halo=halo, text=text, len_q=lq, len_scene=ls, cu_q=cu(lq), cu_k=cu(lk), max_q=max(lq),
                                max_k=max(lk), flops=float(sum(a * b for a, b in zip(lq, lk))))
    return St.varlen_cache[key]


class SplitRefiner(torch.nn.Module):
    """joint_sep*: the token refiner pools and attends over all valid text tokens, so run it on each sentence copy alone"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, timestep, mask):
        if St.setting not in SEP_SETTINGS:
            return self.inner(x, timestep, mask)
        B, L3, C = x.shape
        n = St.nseg
        out = self.inner(x.reshape(B * n, L3 // n, C), timestep.repeat_interleave(n), mask.reshape(B * n, L3 // n))
        return out.reshape(B, L3, -1)


def local_rope(rot):
    """restart the temporal RoPE index at 0 in every scene (video tokens are frame-major)"""
    key = id(rot[0])
    if key not in St.rope_cache:
        sc = scene_of_frame().to(rot[0].device)
        start = torch.stack([(sc == j).nonzero()[0, 0] for j in range(St.nseg)])
        idx = torch.arange(St.F, device=sc.device) - start[sc]
        St.rope_cache = {key: tuple(r.view(St.F, St.T, -1)[idx].reshape(St.Sv, -1) for r in rot)}
    return St.rope_cache[key]


def dynx_mask(q, k, base, S):
    """q, k [H, Sp, D]; keep every block pair except cross-scene video pairs outside the top-p estimated mass"""
    from torch.nn.attention.flex_attention import BlockMask
    H, Sp, D = q.shape
    nB = Sp // 128
    cnt = torch.full((nB,), 128.0, device=q.device)
    cnt[-1] = S - (nB - 1) * 128
    qb = q.float().view(H, nB, 128, D).sum(2) / cnt[None, :, None]
    kb = k.float().view(H, nB, 128, D).sum(2) / cnt[None, :, None]
    P = (torch.einsum('hqd,hkd->hqk', qb, kb) / math.sqrt(D) + cnt.log()[None, None]).softmax(-1)
    lo = torch.arange(nB, device=q.device) * 128
    hi = torch.clamp(lo + 127, max=S - 1)
    vid = lo < St.Sv
    has_text = hi >= St.Sv
    sc = scene_of_frame().to(q.device)
    s_lo = sc[torch.clamp(lo // St.T, max=St.F - 1)]
    s_hi = sc[torch.clamp(torch.minimum(hi, torch.tensor(St.Sv - 1, device=q.device)) // St.T, max=St.F - 1)]
    intra = (s_lo[:, None] <= s_hi[None, :]) & (s_lo[None, :] <= s_hi[:, None])
    sink = (lo[None, :] < St.T).expand(nB, nB)
    fixed = intra | sink | has_text[None, :] | has_text[:, None] | ~vid[:, None] | ~vid[None, :]
    cross = P.masked_fill(fixed[None], 0)
    srt, order = cross.sort(-1, descending=True)
    total = cross.sum(-1, keepdim=True).clamp_min(1e-12)
    need = ((srt.cumsum(-1) - srt) < St.dyn_p * total) & (srt > 0)
    keep = torch.zeros_like(cross, dtype=torch.bool).scatter(-1, order, need) | fixed[None]
    num = keep.sum(-1).int()
    idx = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True).int()
    return BlockMask.from_kv_blocks(num[None], idx[None], BLOCK_SIZE=128, mask_mod=base.mask_mod), keep.float().mean()


def x0_hook(module, args, kwargs, output):
    """dynK: after each conditional pass, measure how much the predicted clean latent still changes"""
    if not St.cond():
        return
    xt = kwargs.get('hidden_states', args[0] if args else None)
    t = kwargs.get('timestep')
    v = output[0] if isinstance(output, (tuple, list)) else output.sample
    sigma = (t.float() / 1000).view(-1, 1, 1, 1, 1)
    x0 = F_.avg_pool3d((xt[:, :v.shape[1]].float() - sigma * v.float()), (1, 4, 4))
    if St.x0_prev is not None:
        rel = float((x0 - St.x0_prev).norm() / x0.norm().clamp_min(1e-6))
        St.x0_log.append(rel)
        m = re.fullmatch(r'dynK(\d+)', St.setting)
        step = St.step()
        if m and St.cut_until == 20 and step >= 1 and rel < int(m.group(1)) / 100:
            St.cut_until = max(2, step + 1)
    St.x0_prev = x0
    cur = {'v': v.float(), 'x0': xt[:, :v.shape[1]].float() - sigma * v.float(), 'xt': xt[:, :v.shape[1]].float()}
    if St.scene_prev is not None and cur['v'].shape == St.scene_prev['v'].shape:
        rec = {'step': St.step(), 'sigma': float(sigma.flatten()[0])}
        for k in ('v', 'x0', 'xt'):
            rec[k] = [float((cur[k][:, :, lo:hi] - St.scene_prev[k][:, :, lo:hi]).norm() / St.scene_prev[k][:, :, lo:hi].norm().clamp_min(1e-6))
                      for lo, hi in scene_ranges()]
        St.scene_log.append(rec)
    St.scene_prev = cur


def hunyuan_text_segments(pipe, sentences):
    """sentence id per MLLM token kept by the pipeline (after crop_start), -1 for template tokens"""
    from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import format_text_input
    prompt = ' '.join(sentences)
    tok = pipe.tokenizer
    conv = format_text_input([prompt], pipe.system_message)
    ids = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True)['input_ids']
    ids = ids[0] if isinstance(ids[0], list) else ids
    text = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        text = text[0]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    assert enc['input_ids'] == list(ids), 'chat template tokenization mismatch'
    start = text.index(prompt)
    bounds, c = [], start
    for s in sentences:
        bounds.append((c, c + len(s)))
        c += len(s) + 1
    seg = []
    for a, b in enc['offset_mapping'][pipe.prompt_template_encode_start_idx:]:
        last = b - 1
        seg.append(next((j for j, (lo, hi) in enumerate(bounds) if lo <= last < hi + 1), -1) if b > a else -1)
    return seg


VAE_CHUNK, VAE_CTX, VAE_FULL_MAX = 16, 4, 44


def chunked_vae_decode(pipe):
    """Hunyuan's VAE tiles only spatially and its decoder attends over all frames, so long latents run out of memory.
    Decode latents longer than VAE_FULL_MAX frames in temporal chunks of VAE_CHUNK, each with VAE_CTX preceding latents as
    causal context whose decoded frames are dropped. Also saves the latents first if St.latent_path is set."""
    orig = pipe.vae.decode

    def decode(z, return_dict=True, **kw):
        if St.latent_path is not None:
            torch.save(z.detach().cpu(), St.latent_path)
        F = z.shape[2]
        if F <= VAE_FULL_MAX:
            return orig(z, return_dict=return_dict, **kw)
        outs = []
        for s in range(0, F, VAE_CHUNK):
            e = min(F, s + VAE_CHUNK)
            c = max(0, s - VAE_CTX)
            x = orig(z[:, :, c:e], return_dict=False)[0]
            outs.append(x if s == 0 else x[:, :, -4 * (e - s):])
            torch.cuda.empty_cache()
        x = torch.cat(outs, 2)
        return (x,) if not return_dict else type('DecoderOutput', (), {'sample': x})()
    pipe.vae.decode = decode


def hunyuan_prepare(pipe, sentences, frames):
    vt, vs = pipe.vae_scale_factor_temporal, pipe.vae_scale_factor_spatial
    St.F = (frames - 1) // vt + 1
    St.T = (HUNYUAN['H'] // vs) * (HUNYUAN['W'] // vs)
    St.Sv = St.F * St.T
    seg = hunyuan_text_segments(pipe, sentences)
    St.text_seg_mllm = seg
    St.sentences = sentences
    St.mask_cache, St.rope_cache, St.varlen_cache, St.valid_text = {}, {}, {}, {}


def hunyuan_build_masks(valid):
    """joint sequence = [video Sv | encoder tokens]; valid [n_text] marks real encoder tokens, which for t2v
    without quoted text are the MLLM prompt tokens in order (image and byT5 slots are all invalid)"""
    F, T, Sv = St.F, St.T, St.Sv
    S = Sv + valid.numel()
    Sp = math.ceil(S / 128) * 128
    tseg = torch.full((Sp,), -2, dtype=torch.long)
    pos = valid.nonzero().flatten().cpu() + Sv
    seg = (St.text_seg_mllm + [-1] * len(pos))[:len(pos)]
    tseg[pos] = torch.tensor(seg)
    tseg = tseg.cuda()
    fscene = scene_of_frame().cuda()
    masks = {}
    if St.setting in SEP_SETTINGS:
        counts = St.sep_counts[St.cond()]
        assert len(pos) == sum(counts), (len(pos), counts)
        tseg[pos.cuda()] = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).cuda()
        seg = torch.cat([fscene.repeat_interleave(T), tseg[Sv:]])
        halo = (sep_table(F, St.nseg) & ~frame_table(F, St.nseg, 'isons')).cuda()

        def mod(b, h, q, kv):
            k_valid = (kv < Sv) | (seg[kv] > -2)
            fq, fk = torch.clamp(q // T, max=F - 1), torch.clamp(kv // T, max=F - 1)
            vv_halo = (q < Sv) & (kv < Sv) & halo[fq, fk]
            return k_valid & ((seg[q] == seg[kv]) | vv_halo | (seg[q] == -2) | (q >= S))
        bm = create_block_mask(mod, None, None, Sp, Sp, device='cuda')
        St.mask_density[id(bm)] = 1 - bm.sparsity() / 100
        St.masks = {('SEP', False): bm, ('SEP', True): bm}
        return
    for name in ['dense'] + table_names():
        tab = frame_table(F, St.nseg, name).cuda() if name != 'dense' else None
        for assign in (False, True):
            use_tab = name != 'dense'

            def mod(b, h, q, kv, use_tab=use_tab, assign=assign, tab=tab):
                qv, kvv = q < Sv, kv < Sv
                fq, fk = torch.clamp(q // T, max=F - 1), torch.clamp(kv // T, max=F - 1)
                ts_q, ts_k = tseg[q], tseg[kv]
                k_valid = kvv | (ts_k > -2)
                vv = tab[fq, fk] if use_tab else torch.ones_like(qv)
                vt = (ts_k == -1) | (ts_k == fscene[fq]) if assign else torch.ones_like(qv)
                # text tokens are updated by video at every layer, so a text token that reads several scenes
                # relays information between them; under iso only sentence j's tokens read video, and only scene j
                if use_tab:
                    tv = (ts_q >= 0) & (ts_q == fscene[fk])
                else:
                    tv = (ts_q < 0) | (ts_q == fscene[fk]) if assign else torch.ones_like(qv)
                allowed = torch.where(qv, torch.where(kvv, vv, vt), torch.where(kvv, tv, torch.ones_like(qv)))
                return k_valid & (allowed | (~qv & (ts_q == -2)) | (q >= S))   # padded text queries: any valid key
            masks[(name, assign)] = create_block_mask(mod, None, None, Sp, Sp, device='cuda')
            St.mask_density[id(masks[(name, assign)])] = 1 - masks[(name, assign)].sparsity() / 100
    St.masks = masks


@torch.no_grad()
def sentence_embeds(pipe, texts):
    """each text encoded alone by the MLLM, copies concatenated along the sequence"""
    es, ms = [], []
    for t in texts:
        e, m = pipe._get_mllm_prompt_embeds(text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, prompt=t, device='cuda',
                                            tokenizer_max_length=pipe.tokenizer_max_length, system_message=pipe.system_message,
                                            crop_start=pipe.prompt_template_encode_start_idx)
        es.append(e)
        ms.append(m)
    return torch.cat(es, 1), torch.cat(ms, 1), [int(m.sum()) for m in ms]


def hunyuan_generate(pipe, prompt, frames, steps, seed, cfg):
    g = torch.Generator('cuda').manual_seed(seed)
    kw = dict(height=HUNYUAN['H'], width=HUNYUAN['W'], num_frames=frames, num_inference_steps=steps, generator=g, output_type='np')
    if St.setting not in SEP_SETTINGS:
        return pipe(prompt=prompt, **kw).frames[0]
    e, m, St.sep_counts[True] = sentence_embeds(pipe, St.sentences)
    e2, m2 = pipe._get_byt5_prompt_embeds(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2, prompt=prompt,
                                          device='cuda', tokenizer_max_length=pipe.tokenizer_2_max_length)
    assert int(m2.sum()) == 0, 'glyph text present'
    ne, nm, St.sep_counts[False] = sentence_embeds(pipe, [''] * len(St.sentences))
    St.subj_mask, St.subj_score = None, None
    if subj_cfg():
        assert St.impl == 'varlen', 'subject links need --impl varlen'
        St.subj_tokens = subject_tokens(pipe, St.sentences, St.subject)
    return pipe(prompt_embeds=e, prompt_embeds_mask=m, prompt_embeds_2=e2, prompt_embeds_mask_2=m2,
                negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm, negative_prompt_embeds_2=e2,
                negative_prompt_embeds_mask_2=m2, **kw).frames[0]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['wan', 'hunyuan'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--ids', default='')
    ap.add_argument('--bench', type=Path, default=BENCH, help='prompt set (event_order_bench.json or scene_graph_bench.json)')
    ap.add_argument('--variant', choices=['prompt', 'prompt_explicit'], default='prompt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frames', type=int, default=81)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--dyn-p', type=float, default=0.9)
    ap.add_argument('--sdpa-dense', action='store_true', help='Hunyuan dense calls via the original SDPA processor')
    ap.add_argument('--impl', choices=['flex', 'varlen', 'fast'], default='flex', help='attention implementation (fast: Wan only)')
    ap.add_argument('--compile', action='store_true', help='torch.compile every transformer block (attention stays eager)')
    a = ap.parse_args()
    bench = json.loads(a.bench.read_text())['prompts']
    rows = [r for r in bench if not a.ids or r['id'] in a.ids.split(',')]
    si, sn = map(int, a.shard.split('/'))
    rows = rows[si::sn]
    St.cfg = a.cfg > 1
    St.dyn_p = a.dyn_p
    St.sdpa_dense = a.sdpa_dense
    St.impl, St.compile = a.impl, a.compile
    if a.model == 'wan':
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(local(WAN['id']), torch_dtype=torch.bfloat16).to('cuda')
        wan_setup(pipe)
        prepare, generate, fps = wan_prepare, wan_generate, WAN['fps']
    else:
        from diffusers import HunyuanVideo15Pipeline
        pipe = HunyuanVideo15Pipeline.from_pretrained(local(HUNYUAN['id']), torch_dtype=torch.bfloat16).to('cuda')
        pipe.vae.enable_tiling()
        chunked_vae_decode(pipe)
        if a.cfg <= 1:
            pipe.guider.disable()
        else:
            pipe.guider.guidance_scale = a.cfg
        hunyuan_setup(pipe)
        pipe.transformer.register_forward_hook(x0_hook, with_kwargs=True)
        prepare, generate, fps = hunyuan_prepare, hunyuan_generate, HUNYUAN['fps']
    pipe.transformer.register_forward_pre_hook(lambda m, args: setattr(St, 'fwd', St.fwd + 1))

    import time
    D = getattr(pipe.transformer.config, 'num_attention_heads', 16) * getattr(pipe.transformer.config, 'attention_head_dim', 128)
    for setting in a.settings.split(','):
        St.setting = setting
        St.assign = setting not in ('baseline', 'separate')
        m = re.fullmatch(r'dynX(\d+)', setting)
        if m:
            St.dyn_p = int(m.group(1)) / 100
        for r in rows:
            d = a.out / setting / r['id']
            if (d / 'video.mp4').exists():
                print(f'[skip] {setting}/{r["id"]}', flush=True)
                continue
            sentences = r['sentences']
            prompt = r[a.variant]
            St.calls, St.cut_until, St.x0_prev, St.x0_log, St.dyn_cache = [], 20, None, [], {}
            St.subj_mask, St.subj_score = None, None
            St.subject = r.get('subject', '')
            St.scene_prev, St.scene_log = None, []
            St.latent_path = None
            if a.model == 'hunyuan' and a.frames > 4 * (VAE_FULL_MAX - 1) + 1:
                (a.out / setting / r['id']).mkdir(parents=True, exist_ok=True)
                St.latent_path = a.out / setting / r['id'] / 'latents.pt'
            St.nseg = len(r['sentences'])
            St.edges = {frozenset(e[:2]) for e in r.get('edges', [])}
            seg_px = [hi - lo for lo, hi in (x['pixel'] for x in r['segments'][str(a.frames)]['per_sentence'])]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if setting == 'separate':
                clips = []
                for s, n in zip(sentences, seg_px):
                    St.fwd, St.assign, St.x0_prev, St.nseg = 0, False, None, 1
                    frames = n if (n - 1) % 4 == 0 else n + (4 - (n - 1) % 4)
                    prepare(pipe, [s], frames)
                    St.setting = 'baseline'
                    clips.append(generate(pipe, s, frames, a.steps, a.seed, a.cfg)[:n])
                St.setting, St.nseg = 'separate', len(sentences)
                video = np.concatenate(clips)
            else:
                prepare(pipe, sentences, a.frames)
                St.fwd = 0
                video = generate(pipe, prompt, a.frames, a.steps, a.seed, a.cfg)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            dens = [float(x) for _, x in St.calls]
            attn = sum(4 * S * S * D * x for (S, _), x in zip(St.calls, dens))
            attn_dense = sum(4 * S * S * D for S, _ in St.calls)
            lin = sum(24 * S * D * D for S, _ in St.calls)
            d.mkdir(parents=True, exist_ok=True)
            export_to_video(list(video), str(d / 'video.mp4'), fps=fps)
            if St.subj_mask is not None:
                save_subject_overlay(video, d / 'subject_mask.png')
            (d / 'meta.json').write_text(json.dumps({'model': a.model, 'setting': setting, 'id': r['id'], 'seed': a.seed,
                                                     'frames': len(video), 'steps': a.steps, 'cfg': a.cfg, 'variant': a.variant,
                                                     'n_layers': St.n_layers, 'latent_frames': St.F, 'seconds': elapsed,
                                                     'attn_calls': len(St.calls), 'attn_block_fraction': float(np.mean(dens)) if dens else None,
                                                     'attn_flops': attn, 'attn_flops_dense': attn_dense, 'linear_flops': lin,
                                                     'flops_ratio': (attn + lin) / (attn_dense + lin) if St.calls else None,
                                                     'cut_until': St.cut_until, 'x0_change': St.x0_log,
                                                     'scene_stats': St.scene_log}, indent=1))
            print(f'[done] {a.model} {setting}/{r["id"]} frames {len(video)} {elapsed:.0f}s'
                  + (f' flops {(attn + lin) / (attn_dense + lin):.3f}' if St.calls else '') + (f' cut_until {St.cut_until}' if setting.startswith('dynK') else ''), flush=True)
    (a.out / f'done_shard{si}.json').write_text('{}')


if __name__ == '__main__':
    main()
