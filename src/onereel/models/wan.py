"""Wan2.2-TI2V-5B (t2v, 704 x 1280 or 480 x 832): per-sentence text and scene attention, optional CFG.

Self-attention: scene j's queries read scene j's frames plus its halo frames, which are adjacent to it, so a scene's
keys are one contiguous token range; or, with the bounded memory, scene j plus M representatives per head of the other
scenes' boundary frames (onereel.memory). Cross-attention: every sentence is encoded alone by umT5 (512 tokens) and
scene j reads only sentence j. With CFG the negative prompt is encoded once and repeated per scene, so scene j reads
chunk j in both passes.

Parallel (torchrun): video tokens are split per rank by a pre-hook on the first block and gathered by a hook on the
last block; the sentences stay replicated. 'scene' gathers K/V over the work-balanced partition (paper Algorithm 1),
'ulysses' redistributes heads (all-to-all), 'auto' picks per request by Eq. 2 (onereel.parallel.comm).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from onereel import attention as A
from onereel.frame_memory import FrameMemory
from onereel.layout import SceneLayout
from onereel.memory import BoundaryMemory, MemoryAttention, candidate_frames, frame_plan
from onereel.parallel import comm, dist as D, ring as R

REPO = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
HEIGHT, WIDTH, FPS, TEXT_LEN = 704, 1280, 24, 512
GUIDANCE = 5.0                           # the quality runs: CFG 5 with NEGATIVE
NEGATIVE = ('Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, '
            'JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face')


@dataclass
class Context:
    layout: SceneLayout
    mode: str = 'scene'                  # 'scene' (single GPU or K/V gathering) | 'ulysses' | 'auto' (Eq. 2 per request)
    theta: float = 1.0                   # Eq. 2: effective bandwidth of K/V gathering relative to all-to-all
    kv_chunks: int = 1                   # K/V gathering: head chunks of a pipelined exchange (1: one exchange; 4 helps
                                         # only without P2P, 3% on short requests, and costs 5-6% on NVLink)
    memory: int = 0                      # M > 0: bounded cross-scene memory of M representatives per head (halo = 0)
    frame_memory: FrameMemory = None     # representative frames chosen once per step (halo = 0)
    T: int = 0
    fwd: int = 0
    own: list = field(default_factory=list)          # per rank (video token lo, hi)
    path: dict = field(default_factory=dict)
    run_mode: str = 'scene'
    ring: R.RingPlan = None              # mode 'ring': per-scene split, rows in the ring's permuted order
    height: int = HEIGHT
    width: int = WIDTH
    cfg: bool = False                    # two transformer passes per step (conditional first)
    attn_override: object = None         # baselines: (layer, q, k, v) on the local heads after the Ulysses all-to-all
    _mem: MemoryAttention = None         # -> output, or None = dense

    def cond(self) -> bool:
        return (self.fwd - 1) % 2 == 0 if self.cfg else True

    def step(self) -> int:
        return (self.fwd - 1) // 2 if self.cfg else self.fwd - 1


torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)  # one graph per sequence length


@torch.compile(fullgraph=True, dynamic=False)
def _qk(q, k, wq, wk, eps: float, cos, sin, heads: int):
    """RMSNorm and the interleaved RoPE of diffusers' Wan attention, fused"""
    def norm(x, w):
        return F.rms_norm(x, (x.shape[-1],), w, eps)

    def rope(x):
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        c, s_ = cos[..., 0::2], sin[..., 1::2]
        return torch.stack([x1 * c - x2 * s_, x1 * s_ + x2 * c], -1).flatten(-2).type_as(x)
    return rope(norm(q, wq).unflatten(2, (heads, -1))), rope(norm(k, wk).unflatten(2, (heads, -1)))


class Processor:
    def __init__(self, ctx: Context, kind: str, orig, layer: int):
        self.ctx, self.kind, self.orig, self.layer = ctx, kind, orig, layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        ctx, T, lay = self.ctx, self.ctx.T, self.ctx.layout
        a, b = ctx.own[D.rank()]
        if self.kind == 'cross':                  # this rank's piece of every scene it holds reads that scene's sentence
            pieces = ([(j, lo, hi) for j, lo, hi in ctx.ring.local[D.rank()] if hi > lo] if ctx.run_mode == 'ring' else
                      [(j, lo - a, hi - a) for j, (lo, hi) in [(j, (max(lo * T, a), min(hi * T, b))) for j, (lo, hi) in enumerate(lay.ranges) if lo * T < b and hi * T > a]])
            return torch.cat([self.orig(attn, hidden_states[:, lo:hi],
                                        encoder_hidden_states[:, j * TEXT_LEN:(j + 1) * TEXT_LEN], None, rotary_emb)
                              for j, lo, hi in pieces], 1)
        rows = ctx.ring.perm[a:b] if ctx.run_mode == 'ring' else slice(a, b)
        cos, sin = (x[:, rows] for x in rotary_emb)
        q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        q, k = _qk(q, k, attn.norm_q.weight, attn.norm_k.weight, attn.norm_q.eps, cos, sin, attn.heads)
        v = v.unflatten(2, (attn.heads, -1))
        q, k, v = (x[0].to(torch.bfloat16) for x in (q, k, v))
        if ctx.attn_override is not None:
            qa, ka, va = comm.a2a_heads(q), comm.a2a_heads(k), comm.a2a_heads(v)
            r = ctx.attn_override(self.layer, qa, ka, va)
            o = comm.a2a_tokens(self.ranges(qa, ka, va) if r is None else r)
        elif ctx.memory:
            o = self.memory(q, k, v)
        elif ctx.run_mode == 'ulysses' or D.world() == 1:
            o = comm.a2a_tokens(self.ranges(comm.a2a_heads(q), comm.a2a_heads(k), comm.a2a_heads(v)))
        elif ctx.run_mode == 'ring':
            o = R.attention(ctx.ring, q, k, v, tag=id(lay.extra))[0]
        else:
            o = self.scene(q, k, v)
        return attn.to_out[1](attn.to_out[0](o.flatten(1, 2)[None].type_as(hidden_states)))

    def ranges(self, q, k, v):
        """all tokens: every scene's keys are one contiguous range (packed per scene when it reads extra frames)"""
        T, lay = self.ctx.T, self.ctx.layout
        if any(lay.extra.get(j) for j in range(lay.n_scenes)):
            rows = lambda x: (lambda fl: torch.cat([x[lo:hi] for lo, hi in comm.frame_tokens(fl, T)]))
            segs = [(q[lo * T:hi * T], rows(k)(lay.key_frames(j)), rows(v)(lay.key_frames(j))) for j, (lo, hi) in enumerate(lay.ranges)]
            o = q.new_empty(q.shape)
            for (lo, hi), oj in zip(lay.ranges, A.packed(segs)):
                o[lo * T:hi * T] = oj
            return o
        q_ranges = [(lo * T, hi * T) for lo, hi in lay.ranges]
        k_ranges = [(a * T, b * T) for a, b in (lay.key_range(j) for j in range(lay.n_scenes))]
        for j, (a, b) in enumerate(lay.key_range(j) for j in range(lay.n_scenes)):
            assert b - a == len(lay.key_frames(j)), 'halo must be contiguous with its scene'
        return A.overlapping_ranges(q, k, v, q_ranges, k_ranges)

    def scene(self, q, k, v):
        """K/V gathering: this rank's query pieces. The exchange is posted per head chunk (ctx.kv_chunks); scenes whose
        keys are all local run first, then the other scenes chunk by chunk as each chunk arrives"""
        ctx, T, lay = self.ctx, self.ctx.T, self.ctx.layout
        a, b = ctx.own[D.rank()]
        H = q.shape[1]
        C = max(1, min(ctx.kv_chunks, H)) if D.world() > 1 else 1
        heads = [slice(c * H // C, (c + 1) * H // C) for c in range(C)]
        exs = [comm.HaloExchange(lay, ctx.own, T, k, v, heads=h if C > 1 else None) for h in heads]
        mine = [(j, lo, hi) for j, (lo, hi) in [(j, (max(lo * T, a), min(hi * T, b))) for j, (lo, hi) in enumerate(lay.ranges) if lo * T < b and hi * T > a]]
        local = lambda j: all(a <= lo and hi <= b for lo, hi in comm.frame_tokens(lay.key_frames(j), T))
        own_rows = lambda x: (lambda fl: torch.cat([x[lo - a:hi - a] for lo, hi in comm.frame_tokens(fl, T)]))
        ov = q.new_empty(q.shape)

        def run(g, kf, vf, h):
            segs = [(q[lo - a:hi - a, h], kf(lay.key_frames(j)), vf(lay.key_frames(j))) for j, lo, hi in g]
            for (j, lo, hi), o in zip(g, A.packed(segs)):
                ov[lo - a:hi - a, h] = o
        near = [s for s in mine if local(s[0])]
        far = [s for s in mine if not local(s[0])]
        if near:
            run(near, own_rows(k), own_rows(v), slice(None))
        for ex, h in zip(exs, heads):
            ex.wait()
            if far:
                run(far, ex.rows(0), ex.rows(1), h)
        return ov

    def memory(self, q, k, v):
        """bounded cross-scene memory: all tokens of the local heads (one GPU, Ulysses), or this rank's pieces with the
        missing scene frames and every boundary candidate gathered"""
        ctx, T, lay = self.ctx, self.ctx.T, self.ctx.layout
        ulysses = ctx.run_mode == 'ulysses' or D.world() == 1
        if ulysses:
            q, k, v = comm.a2a_heads(q), comm.a2a_heads(k), comm.a2a_heads(v)
            token_row = torch.arange(lay.n_frames * T)
            pieces = [(j, lo * T, hi * T) for j, (lo, hi) in enumerate(lay.ranges)]
        else:
            a, b = ctx.own[D.rank()]
            ex = comm.HaloExchange(lay, ctx.own, T, k, v, extra=candidate_frames(lay)[0])
            ex.wait()
            k, v, token_row = ex.buffer()
            pieces = [(j, lo - a, hi - a) for j, (lo, hi) in [(j, (max(lo * T, a), min(hi * T, b))) for j, (lo, hi) in enumerate(lay.ranges) if lo * T < b and hi * T > a]]
        pk = (q.shape[0], k.shape[0])
        if pk not in ctx._mem.plans:
            ctx._mem.plans[pk] = frame_plan(lay, T, token_row, k.shape[0], pieces, device=q.device)
        o = ctx._mem((self.layer, ctx.cond()), q, k, v, ctx._mem.plans[pk])
        return comm.a2a_tokens(o) if ulysses else o


def load(device='cuda'):
    from diffusers import WanPipeline
    from onereel.utils import local_snapshot
    pipe = WanPipeline.from_pretrained(local_snapshot(REPO), torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def install(pipe, ctx: Context):
    Processor.__call__ = torch.compiler.disable(Processor.__call__)
    tr = pipe.transformer
    for i, blk in enumerate(tr.blocks):
        blk.attn1.set_processor(Processor(ctx, 'self', blk.attn1.processor, i))
        blk.attn2.set_processor(Processor(ctx, 'cross', blk.attn2.processor, i))

    def pre(m, args, kwargs):
        ctx.fwd += 1
        if ctx.frame_memory is not None:
            ctx.frame_memory.begin_step(ctx.step())
        t = kwargs.get('timestep')                # TI2V passes one timestep per token; in t2v they are all equal
        if t is not None and t.ndim == 2 and bool((t == t[:, :1]).all()):
            kwargs['timestep'] = t[:, 0]          # the broadcast path: same result, no [S, 6, D] fp32 modulation
        return args, kwargs
    tr.register_forward_pre_hook(pre, with_kwargs=True)

    def post(m, args, kwargs, out):
        if ctx.frame_memory is not None and ctx.cond():
            ctx.frame_memory.finish()
    tr.register_forward_hook(post, with_kwargs=True)
    _observe(ctx, tr.blocks, lambda out: out)
    if D.world() > 1:
        def shard(m, args):
            assert args[2].ndim == 3, 'per-token modulation (TI2V image conditioning) is not split'
            a, b = ctx.own[D.rank()]
            return (args[0][:, ctx.ring.perm[a:b] if ctx.run_mode == 'ring' else slice(a, b)],) + tuple(args[1:])

        def gather(m, args, out):
            x = comm.gather_tokens(out, ctx.own)
            return x[:, ctx.ring.inv] if ctx.run_mode == 'ring' else x
        tr.blocks[0].register_forward_pre_hook(shard)
        tr.blocks[-1].register_forward_hook(gather)


def _observe(ctx, blocks, hs_of):
    """frame memory: before each block its extra frames (per-layer ablation); after an observed block, its local video
    rows pooled per frame in the conditional pass"""
    cache = {}

    def hooks(i):
        def pre(m, args):
            if ctx.frame_memory is not None:
                ctx.frame_memory.use_layer(i)

        def post(m, args, out):
            fm = ctx.frame_memory
            if fm is None or not fm.observes(i) or not ctx.cond():
                return
            a, b = ctx.own[D.rank()]
            hs = hs_of(out)[0]
            key = (a, b, ctx.run_mode, id(ctx.ring))
            if key not in cache:
                cache.clear()
                rows = ctx.ring.perm[a:b] if ctx.run_mode == 'ring' else torch.arange(a, b, device=hs.device)
                cache[key] = rows.to(hs.device) // ctx.T
            fm.collect(cache[key], hs, i)
        return pre, post
    for i, blk in enumerate(blocks):
        pre, post = hooks(i)
        blk.register_forward_pre_hook(pre)
        blk.register_forward_hook(post)


def _plan_parallel(ctx: Context, heads: int):
    """rank token ranges: the work-balanced partition (Algorithm 1) for K/V gathering, equal ranges for Ulysses; 'auto'
    takes the gathering path when Eq. 2 says it moves less (and when Ulysses cannot split the frames or heads)"""
    n, lay = D.world(), ctx.layout
    if ctx.memory:
        assert lay.halo == 0, 'the bounded memory replaces the halo frames: use halo = 0'
        ctx._mem = MemoryAttention(BoundaryMemory(ctx.memory))
    ctx.ring = None
    if n == 1:
        ctx.run_mode, ctx.own, ctx.path = 'scene', [(0, lay.n_frames * ctx.T)], {}
        return
    if ctx.mode == 'ring':                       # baseline: USP's Ring with every scene split evenly over the ranks
        assert not ctx.memory, 'the ring path runs the halo / frame-memory scene attention'
        ctx.ring = R.RingPlan(lay, n, ctx.T)
        ctx.run_mode, ctx.own = 'ring', ctx.ring.own
        ctx.path = dict(path='ring', V_R=ctx.ring.volume(), V_U=4 * lay.n_frames * ctx.T * (n - 1) / n ** 2)
        return
    extra = candidate_frames(lay)[0] if ctx.memory else ()
    gather = comm.partition(lay, n, ctx.T, 'scene', None, ctx.memory)
    ctx.path = comm.choose_path(lay, gather, ctx.T, ctx.theta, extra)
    ctx.path['ulysses_ok'] = lay.n_frames * ctx.T % n == 0 and heads % n == 0
    ctx.run_mode = ctx.mode if ctx.mode != 'auto' else ctx.path['path'] if ctx.path['ulysses_ok'] else 'scene'
    assert ctx.run_mode != 'ulysses' or ctx.path['ulysses_ok'], f'Ulysses needs video tokens and heads divisible by {n}'
    ctx.own = gather if ctx.run_mode == 'scene' else comm.partition(lay, n, ctx.T, 'ulysses')


def generate(pipe, ctx: Context, sentences: list[str], steps: int = 20, seed: int = 0, guidance: float = 1.0,
             negative: str = '', output_type: str = 'np'):
    """guidance > 1: CFG with `negative` repeated per scene; returns frames [N, H, W, 3] in [0, 1]"""
    assert len(sentences) == ctx.layout.n_scenes
    cfg = pipe.transformer.config
    ctx.T = (ctx.height // pipe.vae_scale_factor_spatial // cfg.patch_size[1]) * (ctx.width // pipe.vae_scale_factor_spatial // cfg.patch_size[2])
    ctx.fwd, ctx.cfg = 0, guidance > 1
    if ctx.frame_memory is not None:
        assert ctx.layout.halo == 0 and not ctx.memory, 'frame memory replaces the halo frames: use halo = 0'
        ctx.frame_memory.reset(ctx.layout, len(pipe.transformer.blocks))
    _plan_parallel(ctx, cfg.num_attention_heads)
    frames = (ctx.layout.n_frames * cfg.patch_size[0] - 1) * pipe.vae_scale_factor_temporal + 1
    with torch.no_grad():
        e = torch.cat([pipe._get_t5_prompt_embeds(prompt=s, max_sequence_length=TEXT_LEN, device='cuda') for s in sentences], 1)
        ne = None
        if ctx.cfg:
            ne = pipe._get_t5_prompt_embeds(prompt=negative, max_sequence_length=TEXT_LEN, device='cuda').repeat(1, len(sentences), 1)
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt_embeds=e, negative_prompt_embeds=ne, height=ctx.height, width=ctx.width, num_frames=frames,
                num_inference_steps=steps, guidance_scale=guidance, generator=g, output_type=output_type).frames[0]
