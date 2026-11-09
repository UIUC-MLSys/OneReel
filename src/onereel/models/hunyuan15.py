"""HunyuanVideo-1.5 (480p t2v): per-sentence text, scene attention, subject links, scene / Ulysses parallelism.

The transformer attends over one joint sequence [video | text]. Every sentence is encoded alone by the MLLM and refined
alone by the token refiner; the copies are concatenated. Per layer and step, scene j's video queries read scene j's
video, its halo frames and sentence j; sentence j's text queries read scene j's video and sentence j. Template-free:
no token is shared between scenes. Temporal RoPE keeps the global frame index, or with Context.scene_rope restarts at
every scene (a token is rotated by its frame index within its own scene; shared frames keep their scene's rotation).

Parallel (torchrun, xfuser sequence-parallel group): video tokens are split per rank by a pre-hook on the first block
and gathered by a hook on the last block; text tokens stay replicated. 'scene' gathers K/V over a work-balanced
partition (paper Algorithm 1), 'ulysses' redistributes heads (all-to-all), 'auto' picks per request by Eq. 2. Both
paths reproduce the single-GPU result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from onereel import attention as A
from onereel.frame_memory import FrameMemory
from onereel.layout import SceneLayout
from onereel.memory import BoundaryMemory, MemoryAttention, candidate_frames, frame_plan
from onereel.parallel import comm, dist as D, ring as R
from onereel.subject import SubjectMask, key_frames, logit_score

REPO = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v'
HEIGHT, WIDTH, FPS = 480, 848, 24
VAE_CHUNK, VAE_CTX, VAE_FULL_MAX = 16, 4, 44


@dataclass
class Context:
    layout: SceneLayout
    mode: str = 'scene'                  # 'scene' (single GPU or K/V gathering) | 'ulysses' | 'auto' (Eq. 2 per request)
    partition: str = 'balanced'          # K/V gathering partition: 'balanced' (Algorithm 1) | 'tokens' | 'scenes' (ablations)
    theta: float = 1.0                   # Eq. 2: effective bandwidth of K/V gathering relative to all-to-all
    cfg: bool = True
    shared_pos: str = 'source'           # temporal RoPE position of a scene's shared frames: 'source' (their own frame
                                         # index) or 'adjacent' (global RoPE: earlier shared frames just before the
                                         # scene's first frame, later ones just after its last; scene-local RoPE: all
                                         # of them just after its last frame, in time order); 'switch': 'source'
                                         # before denoising step shared_switch, 'adjacent' from it on
    shared_switch: int = 10
    scene_rope: bool = False             # temporal RoPE index = frame index within the token's own scene (+1 after scene 0)
    rope_lengths: list = None            # scene lengths for scene_rope (default: the attention layout's; dense: one scene)
    kv_chunks: int = 1                   # K/V gathering: head chunks of a pipelined exchange (1: one exchange; 4 helps
                                         # only without P2P, 3% on short requests, and costs 5-6% on NVLink)
    overlap: str = 'none'                # K/V gathering: 'lse' = scenes reading remote keys first attend to their local
                                         # keys while the exchange is in flight, then to the received ones, the two
                                         # merged by log-sum-exp (as Ring does); 'none' = they wait for the exchange
    text_sync: str = 'rows'              # K/V gathering: text query outputs sent to the ranks that read them ('rows')
                                         # or all-reduced to every rank ('all_reduce'); the video output is the same
    memory: int = 0                      # M > 0: bounded cross-scene memory of M representatives per head (layout.halo = 0)
    frame_memory: FrameMemory = None     # representative frames chosen once per step (layout.halo = 0)
    subject_links: int = 0               # n key frames per other scene whose subject tokens every scene also reads
    subject_step: int = 3
    subject_layers: tuple = (18, 36)
    subject_top: float = 0.2
    T: int = 0
    fwd: int = 0
    counts: dict = field(default_factory=dict)       # CFG branch (True = cond) -> MLLM tokens per sentence
    subj_tokens: list = field(default_factory=list)  # per sentence, token positions of the subject phrase
    subject: SubjectMask = None
    own: list = field(default_factory=list)          # per rank (video token lo, hi)
    path: dict = field(default_factory=dict)         # Eq. 2 inputs and the chosen path
    run_mode: str = 'scene'                          # the path this request runs ('auto' resolved; 'ring' only on request)
    ring: R.RingPlan = None                          # mode 'ring': per-scene split, rows in the ring's permuted order
    height: int = HEIGHT                             # 480p checkpoint; 720 x 1280 also runs on it
    width: int = WIDTH
    attn_override: object = None                     # baselines: (layer, q, k, v, eq, ek, ev) on the local heads after
                                                     # the Ulysses all-to-all -> (video out, text out), or None = dense
    forward_ms: list = field(default_factory=list)
    vae_ms: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict)
    _rot_full: tuple = None                          # this forward's rotary (cos, sin) of every video token
    _mem: MemoryAttention = None

    def step(self) -> int:
        return (self.fwd - 1) // 2 if self.cfg else self.fwd - 1

    def cond(self) -> bool:
        return (self.fwd - 1) % 2 == 0 if self.cfg else True

    def text_ranges(self) -> list[tuple[int, int]]:
        c = np.cumsum([0] + self.counts[self.cond()])
        return [(int(c[j]), int(c[j + 1])) for j in range(len(c) - 1)]


def adjacent_positions(lay, j: int) -> list[int]:
    """shared_pos = 'adjacent': per shared frame of scene j (lay.shared_frames order), its temporal position: the shared
    frames before the scene in order just before its first frame, those after it just after its last frame"""
    lo, hi = lay.ranges[j]
    shared = lay.shared_frames(j)
    before, after = sorted(f for f in shared if f < lo), sorted(f for f in shared if f >= hi)
    pos = {f: lo - len(before) + i for i, f in enumerate(before)}
    pos.update({f: hi + i for i, f in enumerate(after)})
    return [pos[f] for f in shared]


def frame_position(ctx, f: int) -> int:
    """temporal RoPE index of latent frame f: f, or with scene_rope its index within its scene (+1 after scene 0)"""
    if not ctx.scene_rope:
        return f
    lengths = list(ctx.rope_lengths or ctx.layout.lengths)
    start = 0
    for n in lengths:
        if f < start + n:
            return f - start + (start > 0)
        start += n
    raise IndexError(f)


def shared_positions(ctx, j: int) -> list[int]:
    """per shared frame of scene j (lay.shared_frames order), its temporal RoPE index under shared_pos = 'adjacent'"""
    lay = ctx.layout
    if not ctx.scene_rope:
        return adjacent_positions(lay, j)
    shared = lay.shared_frames(j)
    last = max(frame_position(ctx, f) for f in lay.frames[j])
    pos = {f: last + 1 + i for i, f in enumerate(sorted(shared))}
    return [pos[f] for f in shared]


def _move_shared(ctx, j: int, k: torch.Tensor) -> torch.Tensor:
    """k = scene j's keys (own frames, then shared frames) rotated at their own frame index: the shared frames' keys
    re-rotated to their adjacent positions (temporal RoPE composes: R(dst) = R(dst - src) R(src); the spatial part
    cancels, so the shift is taken between two frames inside the table)"""
    from diffusers.models.embeddings import apply_rotary_emb
    lay, T = ctx.layout, ctx.T
    shared = lay.shared_frames(j)
    if not shared:
        return k
    n = len(lay.frames[j]) * T
    key = ('adj', j, tuple(shared))
    if key not in ctx._cache:
        a_rows, b_rows = [], []
        for f, dst in zip(shared, shared_positions(ctx, j)):
            d = dst - frame_position(ctx, f)
            a = 0 if d >= 0 else -d                                   # rows of frames a and a + d, same spatial order
            a_rows.append(torch.arange(a * T, (a + 1) * T))
            b_rows.append(torch.arange((a + d) * T, (a + d + 1) * T))
        ctx._cache[key] = (torch.cat(a_rows).to(k.device), torch.cat(b_rows).to(k.device))
    a_rows, b_rows = ctx._cache[key]
    cos, sin = ctx._rot_full
    x = apply_rotary_emb(k[None, n:], (cos[a_rows], -sin[a_rows]), sequence_dim=1)
    x = apply_rotary_emb(x, (cos[b_rows], sin[b_rows]), sequence_dim=1)[0]
    return torch.cat([k[:n], x.to(k.dtype)])


def _segments(ctx, scenes, qf, kf, vf, eq, ek, ev, text_for):
    """packed (q, k, v) per scene piece. scenes: [(j, (token lo, token hi))] video query ranges; qf (lo, hi) -> rows;
    kf / vf frame list -> rows; text queries of the scenes in text_for"""
    lay, tr = ctx.layout, ctx.text_ranges()
    segs, meta = [], []
    for j, (qlo, qhi) in scenes:
        t0, t1 = tr[j]
        keys = lay.key_frames(j)
        kk = kf(keys)
        if ctx.shared_pos == 'adjacent' or (ctx.shared_pos == 'switch' and ctx.step() >= ctx.shared_switch):
            kk = _move_shared(ctx, j, kk)
        segs.append((qf(qlo, qhi), torch.cat([kk, ek[t0:t1]]), torch.cat([vf(keys), ev[t0:t1]])))
        meta.append(('v', j, qlo, qhi))
        if j in text_for:
            fr = lay.frames[j]
            segs.append((eq[t0:t1], torch.cat([kf(fr), ek[t0:t1]]), torch.cat([vf(fr), ev[t0:t1]])))
            meta.append(('t', j, t0, t1))
    return segs, meta


class Processor:
    def __init__(self, ctx: Context, layer: int):
        self.ctx, self.layer = ctx, layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb
        ctx, T = self.ctx, self.ctx.T
        a, b = ctx.own[D.rank()]
        cos, sin = image_rotary_emb
        ctx._rot_full = image_rotary_emb
        rows = local_rows(ctx)
        if ctx.scene_rope:
            rows = scene_rope_rows(ctx, cos.device)[rows]
        rot = (cos[rows], sin[rows])
        q, k, v = (f(hidden_states).unflatten(2, (attn.heads, -1)) for f in (attn.to_q, attn.to_k, attn.to_v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        q = apply_rotary_emb(q, rot, sequence_dim=1)[0].to(torch.bfloat16)
        k = apply_rotary_emb(k, rot, sequence_dim=1)[0].to(torch.bfloat16)
        v = v[0].to(torch.bfloat16)
        eq, ek, ev = (f(encoder_hidden_states).unflatten(2, (attn.heads, -1)) for f in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
        if attn.norm_added_q is not None:
            eq = attn.norm_added_q(eq)
        if attn.norm_added_k is not None:
            ek = attn.norm_added_k(ek)
        eq, ek, ev = (x[0].to(torch.bfloat16) for x in (eq, ek, ev))
        if ctx.subject_links:
            hs, ehs = self.subject(q, k, v, eq, ek, ev)
        elif ctx.memory:
            hs, ehs = self.memory(q, k, v, eq, ek, ev)
        elif ctx.run_mode == 'ulysses' or ctx.attn_override is not None:
            hs, ehs = self.ulysses(q, k, v, eq, ek, ev)
        elif ctx.run_mode == 'ring':
            hs, ehs = R.attention(ctx.ring, q, k, v, eq, ek, ev, ctx.text_ranges(), id(ctx.layout.extra))
        else:
            hs, ehs = self.scene(q, k, v, eq, ek, ev)
        hs = attn.to_out[1](attn.to_out[0](hs.flatten(1, 2)[None].to(hidden_states.dtype)))
        ehs = attn.to_add_out(ehs.flatten(1, 2)[None].to(hidden_states.dtype))
        return hs, ehs

    def scene(self, q, k, v, eq, ek, ev):
        """scenes of this rank's token range. The K/V exchange is posted per head chunk (ctx.kv_chunks); scenes whose
        keys are all local run first, then the other scenes chunk by chunk as each chunk arrives, so the later chunks
        travel while the earlier ones compute (attention is per head: the result does not depend on the chunking)"""
        ctx, T, lay = self.ctx, self.ctx.T, self.ctx.layout
        a, b = ctx.own[D.rank()]
        H = q.shape[1]
        C = max(1, min(ctx.kv_chunks, H)) if D.world() > 1 else 1
        heads = [slice(c * H // C, (c + 1) * H // C) for c in range(C)]
        exs = [comm.HaloExchange(lay, ctx.own, T, k, v, heads=h if C > 1 else None) for h in heads]
        mine = [(j, (max(lo * T, a), min(hi * T, b))) for j, (lo, hi) in enumerate(lay.ranges) if lo * T < b and hi * T > a]
        text_for = {j for j, (lo, _) in enumerate(lay.ranges) if a <= lo * T < b}
        local = lambda j: all(a <= lo and hi <= b for lo, hi in comm.frame_tokens(lay.key_frames(j), T))
        own_rows = lambda x: (lambda fl: torch.cat([x[lo - a:hi - a] for lo, hi in comm.frame_tokens(fl, T)]))
        ov, ot = q.new_empty(q.shape), eq.new_zeros(eq.shape)

        def run(g, kf, vf, h):
            segs, meta = _segments(ctx, g, lambda lo, hi: q[lo - a:hi - a, h], kf, vf, eq[:, h], ek[:, h], ev[:, h],
                                   text_for & {j for j, _ in g})
            for (kind, j, lo, hi), o in zip(meta, A.packed(segs)):
                if kind == 'v':
                    ov[lo - a:hi - a, h] = o
                else:
                    ot[lo:hi, h] = o
        near = [s for s in mine if local(s[0])]
        far = [s for s in mine if not local(s[0])]
        if near:
            run(near, own_rows(k), own_rows(v), slice(None))
        if ctx.overlap == 'lse' and C == 1 and far:
            self._far_lse(far, exs[0], q, eq, ek, ev, text_for, ov, ot)
            far = []
        for ex, h in zip(exs, heads):
            ex.wait()
            if far:
                run(far, ex.rows(0), ex.rows(1), h)
        if ctx.text_sync == 'all_reduce' or D.world() == 1:
            D.all_reduce(ot)
        else:
            comm.send_text_rows(ot, lay, ctx.own, T, ctx.text_ranges())
        return ov, ot

    def _far_lse(self, far, ex, q, eq, ek, ev, text_for, ov, ot):
        """the scene pieces that read remote keys: attention to their local keys (and text) while the exchange is in
        flight, then to the received keys, merged by log-sum-exp"""
        ctx, lay = self.ctx, self.ctx.layout
        a = ctx.own[D.rank()][0]
        segs, meta = _segments(ctx, far, lambda lo, hi: q[lo - a:hi - a], ex.rows(0, 'local'), ex.rows(1, 'local'),
                               eq, ek, ev, text_for & {j for j, _ in far})
        first = A.packed_lse(segs)
        ex.wait()
        kr, vr = ex.rows(0, 'remote'), ex.rows(1, 'remote')
        segs2, which = [], []
        for n, ((kind, j, _, _), s) in enumerate(zip(meta, segs)):
            frames = lay.key_frames(j) if kind == 'v' else lay.frames[j]
            kk = kr(frames)
            if len(kk):
                segs2.append((s[0], kk, vr(frames)))
                which.append(n)
        second = dict(zip(which, A.packed_lse(segs2))) if segs2 else {}
        for n, ((kind, j, lo, hi), (o, l1)) in enumerate(zip(meta, first)):
            if n in second:
                o2, l2 = second[n]
                m = torch.maximum(l1, l2)
                w1, w2 = (l1 - m).exp().T[..., None], (l2 - m).exp().T[..., None]
                o = (o.float() * w1 + o2.float() * w2) / (w1 + w2)
            (ov[lo - a:hi - a] if kind == 'v' else ot[lo:hi]).copy_(o)

    def ulysses(self, q, k, v, eq, ek, ev):
        ctx, T, lay = self.ctx, self.ctx.T, self.ctx.layout
        n, r, H = D.world(), D.rank(), q.shape[1]
        sl = slice(r * H // n, (r + 1) * H // n)
        qa, ka, va = comm.a2a_heads(q), comm.a2a_heads(k), comm.a2a_heads(v)
        if ctx.attn_override is not None:
            r = ctx.attn_override(self.layer, qa, ka, va, eq[:, sl], ek[:, sl], ev[:, sl])
            if r is not None:
                ot = eq.new_zeros(eq.shape)
                ot[:, sl] = r[1]
                D.all_reduce(ot)
                return comm.a2a_tokens(r[0]), ot
        rows = lambda x: (lambda fl: torch.cat([x[lo:hi] for lo, hi in comm.frame_tokens(fl, T)]))
        scenes = [(j, (lo * T, hi * T)) for j, (lo, hi) in enumerate(lay.ranges)]
        segs, meta = _segments(ctx, scenes, lambda lo, hi: qa[lo:hi], rows(ka), rows(va), eq[:, sl], ek[:, sl], ev[:, sl],
                               set(range(lay.n_scenes)))
        ov = qa.new_empty(qa.shape)
        ot = eq.new_zeros(eq.shape)
        for (kind, j, lo, hi), o in zip(meta, A.packed(segs)):
            if kind == 'v':
                ov[lo:hi] = o
            else:
                ot[lo:hi, sl] = o
        D.all_reduce(ot)
        return comm.a2a_tokens(ov), ot

    def memory(self, q, k, v, eq, ek, ev):
        """bounded cross-scene memory (onereel.memory). One GPU or Ulysses: all tokens of the local heads, each rank
        clusters its own heads. K/V gathering: this rank's query pieces read their scenes (split scenes completed from
        the peers) and every boundary candidate (gathered from all ranks)."""
        ctx, T, lay = self.ctx, self.ctx.T, self.ctx.layout
        key, tr = (self.layer, ctx.cond()), ctx.text_ranges()
        if ctx.run_mode == 'ulysses' or D.world() == 1:
            n, r, H = D.world(), D.rank(), q.shape[1]
            sl = slice(r * H // n, (r + 1) * H // n)
            q, k, v = comm.a2a_heads(q), comm.a2a_heads(k), comm.a2a_heads(v)
            token_row = torch.arange(lay.n_frames * T)
            pieces = [(j, lo * T, hi * T) for j, (lo, hi) in enumerate(lay.ranges)]
            text_for = set(range(lay.n_scenes))
        else:
            a, b = ctx.own[D.rank()]
            sl = slice(None)
            ex = comm.HaloExchange(lay, ctx.own, T, k, v, extra=candidate_frames(lay)[0])
            ex.wait()
            k, v, token_row = ex.buffer()
            pieces = [(j, lo - a, hi - a) for j, (lo, hi) in [(j, (max(lo * T, a), min(hi * T, b))) for j, (lo, hi) in enumerate(lay.ranges) if lo * T < b and hi * T > a]]
            text_for = {j for j, (lo, _) in enumerate(lay.ranges) if a <= lo * T < b}
        Sq = q.shape[0]
        pk = (ctx.cond(), Sq, k.shape[0])
        if pk not in ctx._mem.plans:
            ctx._mem.plans[pk] = frame_plan(lay, T, token_row, k.shape[0], pieces, tr, text_for, Sq, q.device)
        out = ctx._mem(key, torch.cat([q, eq[:, sl]]), torch.cat([k, ek[:, sl]]), torch.cat([v, ev[:, sl]]), ctx._mem.plans[pk])
        ot = eq.new_zeros(eq.shape)
        ot[:, sl] = out[Sq:]
        D.all_reduce(ot)
        return comm.a2a_tokens(out[:Sq]) if sl != slice(None) else out[:Sq], ot

    def subject(self, q, k, v, eq, ek, ev):
        """subject links (single GPU): detection at subject_step, then index-gathered sequences"""
        ctx = self.ctx
        assert D.world() == 1, 'subject links run on one GPU'
        sm = ctx.subject
        if ctx.cond() and sm.active(ctx.step()):
            T, off = ctx.T, np.cumsum([0] + ctx.counts[True][:-1])
            score = torch.zeros(ctx.layout.n_frames * T, device=q.device)
            for j, (lo, hi) in enumerate(ctx.layout.ranges):
                idx = torch.tensor([off[j] + t for t in ctx.subj_tokens[j]], device=q.device)
                score[lo * T:hi * T] = logit_score(q[lo * T:hi * T], ek[idx])
            sm.add(self.layer, score.view(ctx.layout.n_frames, T))
        if sm.mask is None or ctx.step() <= ctx.subject_step:
            return self.scene(q, k, v, eq, ek, ev)
        pl = _subject_plan(ctx, eq.shape[0])
        fq, fk, fv = (torch.cat([x, y]) for x, y in ((q, eq), (k, ek), (v, ev)))
        o = A.varlen(fq[pl['q']], fk[pl['k']], fv[pl['k']], pl['cu_q'], pl['cu_k'], pl['max_q'], pl['max_k'])
        out = fq.new_zeros(fq.shape)
        out[pl['q']] = o
        Sv = q.shape[0]
        return out[:Sv], out[Sv:]


def _subject_plan(ctx: Context, Le: int):
    """per scene: video queries with keys = scene + halo + sentence + the other scenes' key-frame subject tokens, and
    sentence queries with keys = scene + sentence"""
    key = ('subj', ctx.cond(), Le)
    if key in ctx._cache:
        return ctx._cache[key]
    T, lay = ctx.T, ctx.layout
    Sv = lay.n_frames * T
    mask = ctx.subject.mask.view(-1)
    ar = lambda a, b: torch.arange(a, b, device='cuda')
    tok = lambda fs: torch.cat([ar(f * T, (f + 1) * T) for f in fs]) if fs else ar(0, 0)
    subj = [torch.cat([ar(f * T, (f + 1) * T)[mask[f * T:(f + 1) * T]] for f in key_frames(fr, ctx.subject_links)]) for fr in lay.frames]
    qs, ks = [], []
    for j, ((lo, hi), (t0, t1)) in enumerate(zip(lay.ranges, ctx.text_ranges())):
        vid, txt = ar(lo * T, hi * T), Sv + ar(t0, t1)
        extra = torch.cat([subj[i] for i in range(lay.n_scenes) if i != j])
        qs.append(vid)
        ks.append(torch.cat([vid, tok(lay.shared_frames(j)), txt, extra]))
        qs.append(txt)
        ks.append(torch.cat([vid, txt]))
    lq, lk = [len(x) for x in qs], [len(x) for x in ks]
    ctx._cache[key] = dict(q=torch.cat(qs), k=torch.cat(ks), cu_q=A.cu_seqlens(lq), cu_k=A.cu_seqlens(lk), max_q=max(lq), max_k=max(lk))
    return ctx._cache[key]


class SplitRefiner(torch.nn.Module):
    """the token refiner pools and attends over all valid text tokens, so it runs on each sentence copy alone"""

    def __init__(self, inner, ctx):
        super().__init__()
        self.inner, self.ctx = inner, ctx

    def forward(self, x, timestep, mask):
        B, L, C = x.shape
        n = self.ctx.layout.n_scenes
        out = self.inner(x.reshape(B * n, L // n, C), timestep.repeat_interleave(n), mask.reshape(B * n, L // n))
        return out.reshape(B, L, -1)


def load(device='cuda'):
    from diffusers import HunyuanVideo15Pipeline
    from onereel.utils import local_snapshot
    pipe = HunyuanVideo15Pipeline.from_pretrained(local_snapshot(REPO), torch_dtype=torch.bfloat16).to(device)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def install(pipe, ctx: Context, timing: bool = False):
    """OneReel attention on every block, per-sentence token refiner, token split / gather when parallel, chunked VAE"""
    tr = pipe.transformer
    for i, blk in enumerate(tr.transformer_blocks):
        blk.attn.set_processor(Processor(ctx, i))
    tr.context_embedder = SplitRefiner(tr.context_embedder, ctx)
    t0 = {}

    def pre(m, args, kw):
        ctx.fwd += 1
        if ctx.frame_memory is not None:
            ctx.frame_memory.begin_step(ctx.step())
        if timing:
            torch.cuda.synchronize()
            t0['f'] = time.perf_counter()

    def post(m, args, kw, out):
        if ctx.frame_memory is not None and ctx.cond():
            ctx.frame_memory.finish()
        if timing:
            torch.cuda.synchronize()
            ctx.forward_ms.append((time.perf_counter() - t0['f']) * 1e3)
    tr.register_forward_pre_hook(pre, with_kwargs=True)
    tr.register_forward_hook(post, with_kwargs=True)
    _observe(ctx, tr.transformer_blocks, lambda out: out[0])
    if D.world() > 1:
        def shard(m, args):
            return (args[0][:, local_rows(ctx)],) + tuple(args[1:])

        def gather(m, args, out):
            return gather_rows(ctx, out[0]), out[1]
        tr.transformer_blocks[0].register_forward_pre_hook(shard)
        tr.transformer_blocks[-1].register_forward_hook(gather)
    _chunked_vae_decode(pipe, ctx, timing)


def scene_rope_rows(ctx, device):
    """per video token row, the rotary row of the same spatial position at its frame index within its scene. Index 0 is
    the causal VAE's single-image latent (it decodes to one frame, every later latent to four), so scenes after the first
    start at index 1: at 0 their first latent is generated as an image latent and decodes, mid-video, to four ghosted frames"""
    lengths = list(ctx.rope_lengths or ctx.layout.lengths)
    key = ('scene_rope', tuple(lengths), ctx.T)
    if key not in ctx._cache:
        start = np.repeat(np.cumsum([0] + lengths[:-1]), lengths)   # first frame of each frame's scene
        local = np.arange(sum(lengths)) - start + (start > 0)
        rows = (local[:, None] * ctx.T + np.arange(ctx.T)[None]).reshape(-1)
        ctx._cache[key] = torch.as_tensor(rows, device=device)
    return ctx._cache[key]


def local_rows(ctx):
    """this rank's video rows: a token slice, or the ring's permuted rows"""
    a, b = ctx.own[D.rank()]
    return ctx.ring.perm[a:b] if ctx.run_mode == 'ring' else slice(a, b)


def gather_rows(ctx, x):
    """[B, local rows, C] -> [B, all tokens, C] in token order"""
    x = comm.gather_tokens(x, ctx.own)
    return x[:, ctx.ring.inv] if ctx.run_mode == 'ring' and D.world() > 1 else x


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


def _chunked_vae_decode(pipe, ctx, timing):
    """the VAE tiles only spatially and its decoder attends over all frames: latents longer than VAE_FULL_MAX frames
    are decoded in chunks of VAE_CHUNK with VAE_CTX preceding latents as causal context (their frames dropped)"""
    orig = pipe.vae.decode

    def decode(z, return_dict=True, **kw):
        if timing:
            torch.cuda.synchronize()
            t = time.perf_counter()
        F = z.shape[2]
        if F <= VAE_FULL_MAX:
            out = orig(z, return_dict=return_dict, **kw)
        else:
            outs = []
            for s in range(0, F, VAE_CHUNK):
                e = min(F, s + VAE_CHUNK)
                c = max(0, s - VAE_CTX)
                x = orig(z[:, :, c:e], return_dict=False)[0]
                outs.append(x if s == 0 else x[:, :, -4 * (e - s):])
                torch.cuda.empty_cache()
            x = torch.cat(outs, 2)
            out = (x,) if not return_dict else type('DecoderOutput', (), {'sample': x})()
        if timing:
            torch.cuda.synchronize()
            ctx.vae_ms.append((time.perf_counter() - t) * 1e3)
        return out
    pipe.vae.decode = decode


@torch.no_grad()
def _sentence_embeds(pipe, texts):
    es, ms = [], []
    for t in texts:
        e, m = pipe._get_mllm_prompt_embeds(text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, prompt=t, device='cuda',
                                            tokenizer_max_length=pipe.tokenizer_max_length, system_message=pipe.system_message,
                                            crop_start=pipe.prompt_template_encode_start_idx)
        es.append(e)
        ms.append(m)
    return torch.cat(es, 1), torch.cat(ms, 1), [int(m.sum()) for m in ms]


def _subject_tokens(pipe, sentences, subject):
    """per sentence, the MLLM token positions (after the template crop) overlapping the subject phrase"""
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


def _plan_parallel(ctx: Context, heads: int):
    """rank token ranges: the work-balanced partition (Algorithm 1) for K/V gathering, equal ranges for Ulysses; 'auto'
    takes the gathering path when Eq. 2 says it moves less (and when Ulysses cannot split the frames or heads)"""
    n, lay = D.world(), ctx.layout
    ctx.ring = None
    if n == 1:
        ctx.run_mode, ctx.own, ctx.path = 'scene', [(0, lay.n_frames * ctx.T)], {}
        return
    if ctx.mode == 'ring':                       # baseline: USP's Ring with every scene split evenly over the ranks
        assert not ctx.memory and not ctx.subject_links, 'the ring path runs the halo / frame-memory scene attention'
        ctx.ring = R.RingPlan(lay, n, ctx.T)
        ctx.run_mode, ctx.own = 'ring', ctx.ring.own
        ctx.path = dict(path='ring', V_R=ctx.ring.volume(), V_U=4 * lay.n_frames * ctx.T * (n - 1) / n ** 2)
        return
    extra = candidate_frames(lay)[0] if ctx.memory else ()
    gather = comm.partition(lay, n, ctx.T, 'scene', ctx.counts[True], ctx.memory, ctx.partition)
    ctx.path = comm.choose_path(lay, gather, ctx.T, ctx.theta, extra)
    ctx.path['ulysses_ok'] = lay.n_frames * ctx.T % n == 0 and heads % n == 0
    ctx.run_mode = ctx.mode if ctx.mode != 'auto' else ctx.path['path'] if ctx.path['ulysses_ok'] else 'scene'
    assert ctx.run_mode != 'ulysses' or ctx.path['ulysses_ok'], f'Ulysses needs video tokens and heads divisible by {n}'
    ctx.own = gather if ctx.run_mode == 'scene' else comm.partition(lay, n, ctx.T, 'ulysses')


def generate(pipe, ctx: Context, sentences: list[str], steps: int = 20, guidance: float = 6.0, seed: int = 0,
             subject: str = '', prompt: str | None = None, output_type: str = 'np'):
    """one video of ctx.layout.n_frames latent frames; returns frames as a float numpy array [N, H, W, 3]"""
    assert len(sentences) == ctx.layout.n_scenes
    ctx.cfg = guidance > 1
    if ctx.cfg:
        pipe.guider.enable()
        pipe.guider.guidance_scale = guidance
    else:
        pipe.guider.disable()
    vs = pipe.vae_scale_factor_spatial
    ctx.T = (ctx.height // vs) * (ctx.width // vs)
    ctx.fwd, ctx._cache = 0, {}
    if ctx.memory:
        assert ctx.layout.halo == 0, 'the bounded memory replaces the halo frames: use halo = 0'
        assert not ctx.subject_links, 'the bounded memory and subject links are separate variants'
        ctx._mem = MemoryAttention(BoundaryMemory(ctx.memory))
    if ctx.frame_memory is not None:
        assert ctx.layout.halo == 0 and not ctx.memory, 'frame memory replaces the halo frames: use halo = 0'
        ctx.frame_memory.reset(ctx.layout, len(pipe.transformer.transformer_blocks))
    ctx.forward_ms, ctx.vae_ms = [], []
    if ctx.subject_links:
        assert subject, 'subject links need the subject phrase'
        ctx.subject = SubjectMask(ctx.subject_step, ctx.subject_layers, ctx.subject_top, (ctx.height // vs, ctx.width // vs))
        ctx.subj_tokens = _subject_tokens(pipe, sentences, subject)
    e, m, ctx.counts[True] = _sentence_embeds(pipe, sentences)
    prompt = prompt or ' '.join(sentences)
    e2, m2 = pipe._get_byt5_prompt_embeds(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2, prompt=prompt,
                                          device='cuda', tokenizer_max_length=pipe.tokenizer_2_max_length)
    assert int(m2.sum()) == 0, 'quoted glyph text is not supported'
    ne, nm, ctx.counts[False] = _sentence_embeds(pipe, [''] * len(sentences))
    _plan_parallel(ctx, pipe.transformer.config.num_attention_heads)
    frames = 4 * (ctx.layout.n_frames - 1) + 1
    g = torch.Generator('cuda').manual_seed(seed)
    return pipe(prompt_embeds=e, prompt_embeds_mask=m, prompt_embeds_2=e2, prompt_embeds_mask_2=m2, negative_prompt_embeds=ne,
                negative_prompt_embeds_mask=nm, negative_prompt_embeds_2=e2, negative_prompt_embeds_mask_2=m2, height=ctx.height,
                width=ctx.width, num_frames=frames, num_inference_steps=steps, generator=g, output_type=output_type).frames[0]
