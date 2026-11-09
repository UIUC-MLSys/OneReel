"""MiniMax-H3 (t2va): per-sentence text, scene attention over the packed [text | audio | video] sequence, subject links.

Every sentence is encoded alone by the conditioner and refined alone by the token refiner. Scene j's queries (its
sentence, its audio latents and its video frames) read scene j's rows plus, as extra keys, `halo` video frames of each
adjacent scene and the audio latents covering the same time; or, with the bounded memory, M representatives per head
of the other scenes' boundary video frames (onereel.memory). The pipeline's 5-15 s duration check is a software limit
and can be lifted (allow_any_duration). On one process the text encoder and transformer sit on two GPUs (or one).

Parallel (torchrun; every rank holds the whole pipeline on its own GPU): the packed rows are split per rank by a
pre-hook on every block (rows, AdaLN indices, RoPE angles) and gathered by a hook on the last block. 'scene' gives each
rank its range of the work-balanced partition (paper Algorithm 1 over the scenes' query rows in time order; a boundary
may fall at any row) and gathers the K/V rows its scenes read; 'ulysses' splits the packed sequence evenly (padded) and
redistributes heads; 'auto' picks per request by Eq. 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from onereel import attention as A
from onereel.frame_memory import FrameMemory
from onereel.layout import SceneLayout
from onereel.memory import BoundaryMemory, MemoryAttention, candidate_frames, make_plan
from onereel.parallel import comm, dist as D
from onereel.parallel.comm import runs
from onereel.subject import SubjectMask, key_frames, logit_score

REPO = 'MiniMaxAI/MiniMax-H3'
FPS = 24


@dataclass
class Context:
    halo: int = 2
    lengths: list | None = None          # latent frames per scene; None: frames split evenly over the sentences
    subject_links: int = 0
    subject_step: int = 7                # transformer forward (1-based) at which the subject mask is built
    subject_layers: tuple = (16, 34)
    subject_top: float = 0.2
    grid: tuple = (17, 30)               # latent rows x columns per frame
    counts: list = field(default_factory=list)       # text tokens per sentence
    subj_tokens: list = field(default_factory=list)
    subject: SubjectMask = None
    fwd: int = 0
    plan: dict = None
    rows: dict = None
    layout: SceneLayout = None
    mode: str = 'scene'                  # 'scene' (one process or K/V gathering) | 'ulysses' | 'auto' (Eq. 2)
    theta: float = 1.0
    memory: int = 0                      # M > 0: bounded cross-scene memory (halo = 0)
    frame_memory: FrameMemory = None     # representative frames chosen once per step (halo = 0)
    attn_override: object = None         # baselines: (layer, q, k, v) on every packed row of the local heads after the
                                         # Ulysses all-to-all -> output, or None = the dense plan
    offload_text_encoder: bool = False   # one GPU holds both models: move the text encoder to CPU after encoding, so the
                                         # transformer's activations fit (auto offload only frees memory for weights)
    n_layers: int = 0
    run_mode: str = 'scene'
    path: dict = field(default_factory=dict)
    par: dict = None                     # rank rows, exchange and attention plans of the current request
    _par: dict = field(default_factory=dict)
    _mem: MemoryAttention = None


def allow_any_duration(min_seconds: float = 0.0, max_seconds: float = 1e9):
    """replace the pipeline's 5-15 s check (frames still snap to 17n + 5)"""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    MiniMaxH3ModularPipeline.min_duration = property(lambda self: min_seconds)
    MiniMaxH3ModularPipeline.max_duration = property(lambda self: max_seconds)


def snap_frames(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def build_plan(ctx: Context, video_idx, audio_idx, text_idx, n_frames):
    """per scene: query rows and key rows (global packed indices)"""
    lay = ctx.layout                     # one layout per request: the frame memory replaces its extra frames every step
    if lay is None or lay.n_frames != n_frames:
        lay = SceneLayout(ctx.lengths, halo=ctx.halo) if ctx.lengths else SceneLayout.even(n_frames, len(ctx.counts), halo=ctx.halo)
        assert lay.n_frames == n_frames, (lay.lengths, n_frames)
        ctx.layout = lay
        if ctx.frame_memory is not None:
            assert ctx.halo == 0 and not ctx.memory, 'frame memory replaces the halo frames (halo = 0)'
            assert not ctx.frame_memory.per_layer, 'H3 builds one plan per forward: per-layer frame memory is not supported'
            ctx.frame_memory.reset(lay, ctx.n_layers)
    if ctx.frame_memory is not None:
        ctx.frame_memory.begin_step(ctx.fwd - 1)
    dev = video_idx.device
    K = lay.n_scenes
    vid = video_idx.view(n_frames, -1)
    n_audio = audio_idx.numel() // 2
    aud = audio_idx.view(2, n_audio)
    vseg = lay.scene_of_frame().to(dev)
    if lay.is_even:
        aseg = torch.arange(n_audio, device=dev) * K // n_audio
    else:
        aseg = vseg[torch.arange(n_audio, device=dev) * n_frames // n_audio]
    t_off = np.cumsum([0] + ctx.counts)
    frames = lay.frames
    row_frame = torch.full((video_idx.numel() + audio_idx.numel() + text_idx.numel(),), -1, dtype=torch.long, device=dev)
    row_frame[vid.flatten()] = torch.arange(n_frames, device=dev).repeat_interleave(vid.shape[1])
    ctx.rows = {'vid': vid, 'vseg': vseg, 'text': text_idx, 't_off': t_off, 'aud': aud, 'frame': row_frame}
    subj_rows = None
    if ctx.subject_links and ctx.subject is not None and ctx.subject.mask is not None:
        subj_rows = [torch.cat([vid[f][ctx.subject.mask[f]] for f in key_frames(fr, ctx.subject_links)]) for fr in frames]
    qs, ks = [], []
    for j in range(K):
        q = torch.cat([text_idx[t_off[j]:t_off[j + 1]], aud[:, aseg == j].flatten(), vid[vseg == j].flatten()])
        extra = []
        for lo, hi in runs(lay.shared_frames(j)):   # halo / frame-memory frames: their video rows and the audio of that time
            extra.append(vid[lo:hi].flatten())
            extra.append(aud[:, lo * n_audio // n_frames:hi * n_audio // n_frames].flatten())
        if subj_rows is not None:
            extra += [subj_rows[i] for i in range(K) if i != j]
        qs.append(q)
        ks.append(torch.cat([q] + extra))
    lq, lk = [len(x) for x in qs], [len(x) for x in ks]
    cf, owner = candidate_frames(lay)
    return dict(q=torch.cat(qs), k=torch.cat(ks), cu_q=A.cu_seqlens(lq, dev), cu_k=A.cu_seqlens(lk, dev), max_q=max(lq),
                max_k=max(lk), scene_q=qs, scene_k=ks, cand=vid[cf].flatten(),
                cand_owner=torch.tensor(owner, device=dev).repeat_interleave(vid.shape[1]))


def _parallel(ctx: Context, pl: dict, N: int, heads: int) -> dict:
    """rank rows of the packed sequence, the exchange and the attention plans of this rank (cached per request shape)"""
    n, r, lay = D.world(), D.rank(), ctx.layout
    key = (N, tuple(lay.lengths), tuple(ctx.counts), tuple((j, tuple(v)) for j, v in sorted(lay.extra.items())))
    if key in ctx._par:
        return ctx._par[key]
    dev = pl['q'].device
    out = {}
    mem_plan = lambda pieces, cand: make_plan(pieces, cand, pl['cand_owner'], dev)
    if n == 1:
        if ctx.memory:
            out['mplan'] = mem_plan([(j, q, k, True) for j, (q, k) in enumerate(zip(pl['scene_q'], pl['scene_k']))], pl['cand'])
        ctx.run_mode, ctx._par[key] = 'scene', out
        return out
    # Algorithm 1 over the scenes' query rows, each scene's rows in time order (sentence first, then audio and video
    # by frame); the per-query cost counts the scene's key rows (plus the memory budget). Boundaries fall at any row.
    sq, sk = [x.cpu() for x in pl['scene_q']], [x.cpu() for x in pl['scene_k']]
    vid, aud = ctx.rows['vid'].cpu(), ctx.rows['aud'].cpu()
    F, n_audio = lay.n_frames, aud.shape[1]
    when = torch.full((N,), -1, dtype=torch.long)
    when[vid.flatten()] = torch.arange(F).repeat_interleave(vid.shape[1])
    when[aud] = (torch.arange(n_audio) * F // n_audio)[None].expand(2, -1)
    order = [q[when[q].argsort(stable=True)] for q in sq]
    cost = [comm.C_LIN + comm.C_ATT * comm.EFF_SCENE * (len(k) + ctx.memory) for k in sk]
    own = comm.balanced([len(q) for q in order], cost, n)
    owner = torch.full((N,), -1, dtype=torch.long)
    seq = torch.cat(order)
    for p, (a, b) in enumerate(own):
        owner[seq[a:b]] = p
    assert bool((owner >= 0).all()), 'a packed row belongs to no scene'
    cand = pl['cand'].cpu()
    need = []
    for p in range(n):
        m = torch.zeros(N, dtype=torch.bool)
        for q, k in zip(sq, sk):
            if bool((owner[q] == p).any()):
                m[k] = True
        if ctx.memory:
            m[cand] = True
        need.append((m & (owner != p)).nonzero()[:, 0])
    ctx.path = comm.eq2(max(len(x) for x in need), N, n, ctx.theta)
    ctx.path['ulysses_ok'] = heads % n == 0
    ctx.path['own_queries'] = own
    ctx.run_mode = ctx.mode if ctx.mode != 'auto' else ctx.path['path'] if ctx.path['ulysses_ok'] else 'scene'
    if ctx.run_mode == 'scene':
        ex = comm.row_plan(owner, need, r, dev)
        pos = torch.full((N,), -1, dtype=torch.long)
        pos[ex['local']] = torch.arange(len(ex['local']))
        bp = ex['bufpos']
        pieces = [(j, pos[q[owner[q] == r]], bp[k], True) for j, (q, k) in enumerate(zip(sq, sk))]
        assert all(bool((x[2] >= 0).all()) for x in pieces if len(x[1]))
        out['ex'] = ex
        if ctx.memory:
            out['mplan'] = mem_plan(pieces, bp[cand].to(dev))
        else:
            live = [x for x in pieces if len(x[1])]
            lq, lk = [len(x[1]) for x in live], [len(x[2]) for x in live]
            out['gplan'] = dict(q=torch.cat([x[1] for x in live]).to(dev), k=torch.cat([x[2] for x in live]).to(dev),
                                cu_q=A.cu_seqlens(lq, dev), cu_k=A.cu_seqlens(lk, dev), max_q=max(lq), max_k=max(lk))
        rows = [(owner == p).nonzero()[:, 0] for p in range(n)]
    else:
        assert ctx.path['ulysses_ok'], f'Ulysses needs heads divisible by {n}'
        m = -(-N // n)
        rows = [torch.arange(p * m, min((p + 1) * m, N)) for p in range(n)]
        if ctx.memory:
            out['mplan'] = mem_plan([(j, q, k, True) for j, (q, k) in enumerate(zip(pl['scene_q'], pl['scene_k']))], pl['cand'])
        out['pad'] = m
        out['valid'] = (torch.arange(r * m, (r + 1) * m) < N).to(dev)
    local = rows[r] if ctx.run_mode == 'scene' else torch.arange(r * m, (r + 1) * m).clamp(max=N - 1)
    out.update(local=local.to(dev), rows=[x.to(dev) for x in rows], width=max(len(x) for x in rows) if ctx.run_mode == 'scene' else m,
               N=N, fwd=-1)
    ctx._par[key] = out
    return out


class Processor:
    def __init__(self, ctx: Context, layer: int):
        self.ctx, self.layer = ctx, layer

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        ctx = self.ctx
        if attn.fused_projections:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = _apply_rotary_emb(q, *rotary_emb), _apply_rotary_emb(k, *rotary_emb)
        q, k, v = (x[0].to(torch.bfloat16) for x in (q, k, v))
        if ctx.subject is not None and ctx.subject.active(ctx.fwd) and ctx.rows is not None:
            assert D.world() == 1, 'subject links run on one process'
            r = ctx.rows
            score = torch.zeros(r['vid'].shape, device=q.device)
            for j in range(len(ctx.counts)):
                ksub = k[r['text'][[r['t_off'][j] + t for t in ctx.subj_tokens[j]]]]
                fr = (r['vseg'] == j).nonzero().flatten()
                score[fr] = logit_score(q[r['vid'][fr].flatten()], ksub).view(len(fr), -1)
            ctx.subject.add(self.layer, score)
        par = ctx.par
        if ctx.run_mode == 'ulysses' or D.world() == 1:
            q, k, v = comm.a2a_heads(q), comm.a2a_heads(k), comm.a2a_heads(v)
            o = ctx.attn_override(self.layer, q, k, v) if ctx.attn_override is not None else None
            if o is None and ctx.memory:
                o = ctx._mem(self.layer, q, k, v, par['mplan'])
            elif o is None:
                p = ctx.plan
                o = torch.zeros_like(q)
                o[p['q']] = A.varlen(q[p['q']], k[p['k']], v[p['k']], p['cu_q'], p['cu_k'], p['max_q'], p['max_k'])
            o = comm.a2a_tokens(o)
        else:                                     # K/V gathering: this rank's rows; the rows its scenes read gathered
            ex = comm.RowExchange(par['ex'], k, v)
            ex.wait()
            kb, vb = ex.buffer()
            if ctx.memory:
                o = ctx._mem(self.layer, q, kb, vb, par['mplan'])
            else:
                p = par['gplan']
                o = torch.zeros_like(q)
                o[p['q']] = A.varlen(q[p['q']], kb[p['k']], vb[p['k']], p['cu_q'], p['cu_k'], p['max_q'], p['max_k'])
        o = o.flatten(1, 2)[None].type_as(hidden_states)
        return attn.to_out[1](attn.to_out[0](o))


class SplitRefiner(torch.nn.Module):
    def __init__(self, inner, ctx):
        super().__init__()
        self.inner, self.ctx = inner, ctx

    def forward(self, x):
        return torch.cat([self.inner(c) for c in x.split(self.ctx.counts, dim=1)], dim=1)


def load(text_device='cuda:1', transformer_device='cuda:0'):
    """(conditioner, rest of the t2va workflow), each with automatic CPU offload on its own GPU"""
    from diffusers import ComponentsManager, ModularPipeline
    workflow = ModularPipeline.from_pretrained(REPO).blocks.get_workflow('t2va')
    tm = ComponentsManager()
    tm.enable_auto_cpu_offload(device=text_device)
    conditioner = workflow.sub_blocks.pop('text_encoder').init_pipeline(REPO, components_manager=tm)
    conditioner.load_components(dtype=torch.bfloat16)
    mm = tm if text_device == transformer_device else ComponentsManager()      # one GPU: one manager offloads both
    if mm is not tm:
        mm.enable_auto_cpu_offload(device=transformer_device)
    rest = workflow.init_pipeline(REPO, components_manager=mm)
    rest.load_components(dtype=torch.bfloat16)
    return conditioner, rest


def install(transformer, ctx: Context):
    blocks = transformer.transformer_blocks
    for i, b in enumerate(blocks):
        b.attn.processor = Processor(ctx, i)
    ctx.n_layers = len(blocks)

    def pre(m, args, kwargs):
        ctx.fwd += 1
        n_frames = kwargs['hidden_states'].shape[1] // (ctx.grid[0] * ctx.grid[1])     # grid of this request's size
        ctx.plan = build_plan(ctx, kwargs['video_indices'], kwargs['audio_indices'], kwargs['text_indices'], n_frames)
        ctx.par = _parallel(ctx, ctx.plan, kwargs['position_ids'].shape[0], blocks[0].attn.heads)
    transformer.register_forward_pre_hook(pre, with_kwargs=True)

    def post(m, args, kwargs, out):
        if ctx.frame_memory is not None:
            ctx.frame_memory.finish()
    transformer.register_forward_hook(post, with_kwargs=True)

    def hooks(i):                        # frame memory: per-layer extra frames; pool an observed block's video rows
        def pre(m, args):
            if ctx.frame_memory is not None:
                ctx.frame_memory.use_layer(i)

        def post(m, args, out):
            fm = ctx.frame_memory
            if fm is None or not fm.observes(i):
                return
            frame = ctx.rows['frame']
            if D.world() > 1:
                frame = frame[ctx.par['local']].masked_fill(~ctx.par.get('valid', torch.ones_like(ctx.par['local'], dtype=torch.bool)), -1)
            fm.collect(frame, out[0], i)
        return pre, post
    for i, b in enumerate(blocks):
        pre, post = hooks(i)
        b.register_forward_pre_hook(pre)
        b.register_forward_hook(post)
    transformer.token_refiner = SplitRefiner(transformer.token_refiner, ctx)
    if D.world() > 1:
        def shard(m, args):                       # block(hidden_states, temb, adaln_indices, rotary_emb)
            hs, temb, ada, rot = args
            par = ctx.par
            if par['fwd'] != ctx.fwd:             # once per forward: this rank's AdaLN rows and RoPE angles
                par['ada'], par['rot'] = ada.index_select(0, par['local']), tuple(x.index_select(0, par['local']) for x in rot)
                par['fwd'] = ctx.fwd
            if m is blocks[0]:
                hs = hs.index_select(1, par['local'])
            return hs, temb, par['ada'], par['rot']

        def gather(m, args, out):
            par = ctx.par
            buf = out.new_zeros(1, par['width'], out.shape[2])
            buf[:, :out.shape[1]] = out
            allb = [torch.empty_like(buf) for _ in range(D.world())]
            torch.distributed.all_gather(allb, buf, group=D.group())
            full = out.new_empty(1, par['N'], out.shape[2])
            for rows, x in zip(par['rows'], allb):
                full[:, rows] = x[:, :len(rows)]
            return full
        for b in blocks:
            b.register_forward_pre_hook(shard)
        blocks[-1].register_forward_hook(gather)


def _subject_tokens(tokenizer, sentences, subject):
    out = []
    for s in sentences:
        a = s.lower().index(subject.lower())
        enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
        out.append([i for i, (x, y) in enumerate(enc['offset_mapping']) if x < a + len(subject) and y > a])
    return out


def sentence_state(conditioner, ctx: Context, sentences, subject=''):
    """each sentence encoded alone; prompt_embeds / text_token_tags concatenated along the token axis"""
    states = [conditioner(prompt=s) for s in sentences]
    st = states[0]
    st.set('prompt_embeds', torch.cat([s.get('prompt_embeds') for s in states], dim=-2))
    tags = [s.get('text_token_tags') for s in states]
    st.set('text_token_tags', torch.cat(tags, dim=-1))
    ctx.counts = [int(t.shape[-1]) for t in tags]
    if subject:
        ctx.subj_tokens = _subject_tokens(conditioner.tokenizer, sentences, subject)
    return st


def encode(conditioner, ctx: Context, requests):
    """text states of several requests (each a list of sentences) with one text-encoder residency: all encoded, then
    the encoder leaves the GPU once (ctx.offload_text_encoder). Returns [(state, counts)] for generate(encoded=...)"""
    out = []
    for sentences in requests:
        state = sentence_state(conditioner, ctx, sentences)
        out.append((state, ctx.counts))
    if ctx.offload_text_encoder:
        conditioner.text_encoder.to('cpu')
        torch.cuda.empty_cache()
    return out


def generate(conditioner, rest, ctx: Context, sentences, num_frames: int, height: int = 544, width: int = 960, seed: int = 0,
             subject: str = '', steps: int | None = None, encoded=None):
    """returns the pipeline output dict with 'videos', 'audio', 'sampling_rate'; encoded: this request's
    (state, counts) from encode(), so the text encoder is not brought back"""
    ctx.grid = (height // 32, width // 32)
    ctx.fwd, ctx.plan, ctx.rows, ctx.par, ctx._par, ctx.layout = 0, None, None, None, {}, None
    if ctx.memory:
        assert ctx.halo == 0 and not ctx.subject_links, 'the bounded memory replaces the halo frames (halo = 0)'
        ctx._mem = MemoryAttention(BoundaryMemory(ctx.memory))
    ctx.subject = SubjectMask(ctx.subject_step, ctx.subject_layers, ctx.subject_top, ctx.grid) if ctx.subject_links else None
    if ctx.subject_links:
        assert subject, 'subject links need the subject phrase'
    if encoded is not None:
        state, ctx.counts = encoded
    else:
        state = sentence_state(conditioner, ctx, sentences, subject if ctx.subject_links else '')
        if ctx.offload_text_encoder:
            conditioner.text_encoder.to('cpu')        # the offload hook brings it back for the next request
            torch.cuda.empty_cache()
    kw = {} if steps is None else {'num_inference_steps': steps}          # None: the pipeline's 50
    return rest(state=state, generator=torch.Generator().manual_seed(seed), output=['videos', 'audio', 'sampling_rate'],
                height=height, width=width, num_frames=num_frames, **kw)
