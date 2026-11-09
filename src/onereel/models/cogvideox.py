"""CogVideoX1.5-5B (t2v, 768 x 1360 or 480 x 848): per-sentence text, scene attention, scene / Ulysses parallelism.

The transformer attends over one joint sequence [text | video] with shared projections. The patch embedding groups 2
latent frames per token frame (patch_size_t = 2), so the scene layout counts token frames: a scene of 8 latent frames
is 4 layout frames of T = (H / 16) * (W / 16) tokens. Every sentence is encoded alone by T5 (226 tokens, padding
included, as the pipeline does) and the copies are concatenated; the negative prompt is encoded per scene the same way.
Per layer and step, scene j's video queries read scene j's video, its halo or memory frames and sentence j; sentence j's
text queries read scene j's video and sentence j (the attention paths of models.hunyuan15, reused). Temporal RoPE keeps
the global token-frame index.

CFG runs both branches in one forward (batch [negative, positive]); the branches share the scene structure, so the
batch is folded into the head axis for the packed attention. Parallel (torchrun): video tokens are split per rank by a
pre-hook on the first block and gathered by a hook on the last block; text tokens stay replicated. 'scene' gathers K/V
over the work-balanced partition (Algorithm 1), 'ulysses' redistributes heads, 'auto' picks per request by Eq. 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from onereel.frame_memory import FrameMemory
from onereel.layout import SceneLayout
from onereel.models import hunyuan15 as H
from onereel.parallel import comm, dist as D

REPO = 'zai-org/CogVideoX1.5-5B'
HEIGHT, WIDTH, FPS, TEXT_LEN, PATCH_T = 768, 1360, 16, 226, 2


@dataclass
class Context:
    layout: SceneLayout                  # token frames (PATCH_T latent frames each)
    mode: str = 'scene'                  # 'scene' (single GPU or K/V gathering) | 'ulysses' | 'auto' (Eq. 2 per request)
    theta: float = 1.0                   # Eq. 2: effective bandwidth of K/V gathering relative to all-to-all
    cfg: bool = True
    frame_memory: FrameMemory = None     # representative token frames chosen once per step (layout.halo = 0)
    memory: int = 0                      # hunyuan15's per-head boundary memory is not ported (stays 0)
    subject_links: int = 0               # not ported (stays 0)
    T: int = 0
    fwd: int = 0
    counts: dict = field(default_factory=dict)       # True / False -> text tokens per sentence (TEXT_LEN each)
    own: list = field(default_factory=list)          # per rank (video token lo, hi)
    path: dict = field(default_factory=dict)         # Eq. 2 inputs and the chosen path
    run_mode: str = 'scene'
    ring: object = None                              # mode 'ring' (hunyuan15.R.RingPlan)
    height: int = HEIGHT
    width: int = WIDTH
    attn_override: object = None                     # baselines, as in hunyuan15.Context (heads = batch x heads)
    _cache: dict = field(default_factory=dict)

    def step(self) -> int:
        return self.fwd - 1                          # one forward per step (CFG batched)

    def cond(self) -> bool:
        return True                                  # every forward carries the conditional branch

    def text_ranges(self) -> list[tuple[int, int]]:
        c = np.cumsum([0] + self.counts[True])
        return [(int(c[j]), int(c[j + 1])) for j in range(len(c) - 1)]


def fold(x):
    """[B, S, H, D] -> [S, B * H, D]"""
    return x.transpose(0, 1).flatten(1, 2)


def unfold(x, B):
    """[S, B * H, D] -> [B, S, H * D]"""
    return x.unflatten(1, (B, -1)).transpose(0, 1).flatten(2, 3)


class Processor(H.Processor):
    """CogVideoX joint attention on hunyuan15's scene / Ulysses paths (video and text share to_q / to_k / to_v, the
    q / k norms and to_out)"""

    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb
        ctx = self.ctx
        a, b = ctx.own[D.rank()]
        B, heads = hidden_states.shape[0], attn.heads
        proj = lambda x: [f(x).unflatten(2, (heads, -1)) for f in (attn.to_q, attn.to_k, attn.to_v)]
        q, k, v = proj(hidden_states)
        eq, ek, ev = proj(encoder_hidden_states)
        q, k, eq, ek = attn.norm_q(q), attn.norm_k(k), attn.norm_q(eq), attn.norm_k(ek)
        cos, sin = image_rotary_emb
        rows = H.local_rows(ctx)
        rot = (cos[rows], sin[rows])
        q = apply_rotary_emb(q, rot, sequence_dim=1)
        k = apply_rotary_emb(k, rot, sequence_dim=1)
        q, k, v, eq, ek, ev = (fold(x).to(torch.bfloat16) for x in (q, k, v, eq, ek, ev))
        if ctx.run_mode == 'ulysses' or ctx.attn_override is not None:
            hs, ehs = self.ulysses(q, k, v, eq, ek, ev)
        elif ctx.run_mode == 'ring':
            hs, ehs = H.R.attention(ctx.ring, q, k, v, eq, ek, ev, ctx.text_ranges(), id(ctx.layout.extra))
        else:
            hs, ehs = self.scene(q, k, v, eq, ek, ev)
        out = lambda x: attn.to_out[1](attn.to_out[0](unfold(x, B).to(hidden_states.dtype)))
        return out(hs), out(ehs)


def load(device='cuda'):
    from diffusers import CogVideoXPipeline
    from onereel.utils import local_snapshot
    pipe = CogVideoXPipeline.from_pretrained(local_snapshot(REPO), torch_dtype=torch.bfloat16).to(device)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def install(pipe, ctx: Context):
    """OneReel attention on every block, frame-memory hooks, token split / gather when parallel"""
    tr = pipe.transformer
    for i, blk in enumerate(tr.transformer_blocks):
        blk.attn1.set_processor(Processor(ctx, i))

    def pre(m, args, kw):
        ctx.fwd += 1
        if ctx.frame_memory is not None:
            ctx.frame_memory.begin_step(ctx.step())

    def post(m, args, kw, out):
        if ctx.frame_memory is not None:
            ctx.frame_memory.finish()
    tr.register_forward_pre_hook(pre, with_kwargs=True)
    tr.register_forward_hook(post, with_kwargs=True)
    H._observe(ctx, tr.transformer_blocks, lambda out: out[0][-1:])     # the conditional branch is batch row -1
    if D.world() > 1:
        def shard(m, args, kw):
            kw['hidden_states'] = kw['hidden_states'][:, H.local_rows(ctx)]
            return args, kw

        def gather(m, args, kw, out):
            return H.gather_rows(ctx, out[0]), out[1]
        tr.transformer_blocks[0].register_forward_pre_hook(shard, with_kwargs=True)
        tr.transformer_blocks[-1].register_forward_hook(gather, with_kwargs=True)


@torch.no_grad()
def _sentence_embeds(pipe, texts):
    return torch.cat([pipe._get_t5_prompt_embeds(prompt=t, max_sequence_length=TEXT_LEN, device='cuda') for t in texts], 1)


def generate(pipe, ctx: Context, sentences: list[str], steps: int = 20, guidance: float = 6.0, seed: int = 0,
             negative: str = '', output_type: str = 'np'):
    """one video of ctx.layout.n_frames token frames (PATCH_T latent frames each); returns float frames [N, H, W, 3]"""
    lay = ctx.layout
    assert len(sentences) == lay.n_scenes
    ctx.cfg = guidance > 1
    ctx.T = (ctx.height // 16) * (ctx.width // 16)
    ctx.fwd, ctx._cache = 0, {}
    if ctx.frame_memory is not None:
        assert lay.halo == 0, 'frame memory replaces the halo frames: use halo = 0'
        ctx.frame_memory.reset(lay, len(pipe.transformer.transformer_blocks))
    e = _sentence_embeds(pipe, sentences)
    ne = _sentence_embeds(pipe, [negative] * len(sentences)) if ctx.cfg else None
    ctx.counts = {True: [TEXT_LEN] * lay.n_scenes, False: [TEXT_LEN] * lay.n_scenes}
    H._plan_parallel(ctx, pipe.transformer.config.num_attention_heads * (2 if ctx.cfg else 1))
    frames = 4 * (PATCH_T * lay.n_frames - 1) + 1                 # an even number of latent frames: no padding
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt_embeds=e, negative_prompt_embeds=ne, height=ctx.height, width=ctx.width, num_frames=frames,
                num_inference_steps=steps, guidance_scale=guidance, generator=g, output_type=output_type).frames[0]
