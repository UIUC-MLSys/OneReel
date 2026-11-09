"""Per-sentence scene attention for MiniMax-H3 (t2va, two GPUs as in experiments/minimax_h3_two_gpu.py).

Packed t2va sequence: [text rows | audio rows (channel-major, N latents per channel) | video rows (F latent frames x
R rows)]. With K sentences the video frames and the audio latents are split evenly in time into K scenes.
Settings
  baseline           the original model and processor
  check_dense        every sentence encoded alone, but one dense attention document through this script's processor
                     (the processor path's numerical check is --check, which compares it with the original on one forward)
  joint_sep_grope    every sentence encoded alone by the conditioner and refined alone by the token refiner; scene j's
                     video, audio and sentence-j text attend only to each other (one flash_attn varlen sequence per scene)
  joint_sep_grope_adj{m}   plus, as extra keys, the m video frames of each adjacent scene nearest to scene j and the audio
                     latents covering the same time
  joint_sep_grope_adj{m}_subjA{n}   plus subject links: at forward SUBJ_STEP the logits of every scene's video queries to
                     the tokens of its sentence's subject phrase are averaged over layers SUBJ_LAYERS, smoothed 3x3 per
                     frame, and the top SUBJ_TOP of each latent frame becomes the subject mask; from the next forward
                     scene j also reads the subject rows of n evenly spaced key frames of every other scene
  separate           every sentence generated alone and its first CLIP_FRAMES frames kept (3 sentences; otherwise the
                     --frames split evenly) (H3 needs >= 120 frames, so each clip is generated at >= 124 frames), then
                     concatenated
  solprof_dense      the whole prompt, dense attention, timing dense vs Sol-Attn attention on the real q / k / v
                     (experiments/official_warmup/sol_h3_profile.py); writes --probe-out/solprof_<id>.json
  capture_dense      the whole prompt, dense attention, with the tail-compression error probe
                     (experiments/tail_compression.py) on 3 layers / 3 steps; writes --probe-out/h3_<id>.json
  probe_dense        the whole prompt, dense attention through this script's processor, with block attention statistics
                     (experiments/block_attention_stats.py) on PROBE_LAYERS at every forward; writes --probe-out/<id>.json
                     and <id>_maps.pt instead of a video
Saves OUT/<setting>/<id>/{video.mp4, audio.wav, meta.json}.

Usage: data/envs/h3/bin/python experiments/minimax_h3_scene.py --out DIR --settings joint_sep_grope_adj2 [--ids ...] [--check]
"""
import argparse
import math
import os
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

BENCH = Path(__file__).resolve().parents[1] / 'dataset' / 'event_order_bench.json'
REPO = 'MiniMaxAI/MiniMax-H3'


SUBJ_STEP, SUBJ_LAYERS, SUBJ_TOP = 7, (16, 34), 0.2
CLIP_FRAMES, KEEP_FRAC = (43, 42, 39), (0.0, 0.2, 0.4, 0.6, 0.8)
H3_MIN_FRAMES = 120      # H3 only generates 5-15 s, so a separate clip is generated at >= 5 s and its first frames kept


class St:
    setting = 'baseline'
    counts = []          # text tokens per sentence, in packed order
    plan = None
    fwd = 0
    subject = ''
    subj_tok = []        # per sentence, local token positions of the subject phrase
    subj_score = subj_mask = None
    grid = (17, 30)
    rows = None          # dict: video rows [F, R], text offsets, for subject scoring
    probe = None         # probe_dense: BlockStats
    capture = None       # capture_dense: experiments/tail_compression.TailProbe
    plan_full = None     # _vfull: plan whose video queries read every video row
    capture_full = None  # solprof_dense: callable(step, layer, q, k, v, St) on the full packed sequence
    vid_idx = None


SETTING_RE = r'joint_sep_grope(?:_adj(\d+))?(?:_subjA(\d+))?(_vfull(?:L(\d+))?)?(_spat)?'


def setting_parts():
    m = re.fullmatch(SETTING_RE, St.setting)
    return (int(m.group(1)) if m and m.group(1) else 0), (int(m.group(2)) if m and m.group(2) else 0)


def variant_parts():
    """(video queries see all video keys, only in layers < L (None: all layers), same-position keys of out-of-scope frames)"""
    m = re.fullmatch(SETTING_RE, St.setting)
    if not m:
        return False, None, False
    return bool(m.group(3)), (int(m.group(4)) if m.group(4) else None), bool(m.group(5))


def subject_tokens(tokenizer, sentences, subject):
    out = []
    for s in sentences:
        a = s.lower().index(subject.lower())
        enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
        out.append([i for i, (x, y) in enumerate(enc['offset_mapping']) if x < a + len(subject) and y > a])
    return out


def sentence_state(conditioner, sentences):
    """encode each sentence alone and concatenate prompt_embeds / text_token_tags along the token axis"""
    states = [conditioner(prompt=s) for s in sentences]
    pe = [s.get('prompt_embeds') for s in states]
    tt = [s.get('text_token_tags') for s in states]
    st = states[0]
    st.set('prompt_embeds', torch.cat(pe, dim=-2))
    st.set('text_token_tags', torch.cat(tt, dim=-1))
    St.counts = [int(t.shape[-1]) for t in tt]
    if St.subject:
        St.subj_tok = subject_tokens(conditioner.tokenizer, sentences, St.subject)
        assert all(len(conditioner.tokenizer(x, add_special_tokens=False)['input_ids']) == c for x, c in zip(sentences, St.counts))
    return st


class SplitRefiner(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if St.setting == 'baseline' or not St.counts:
            return self.inner(x)
        return torch.cat([self.inner(c) for c in x.split(St.counts, dim=1)], dim=1)


def build_plan(seq_len, video_idx, audio_idx, text_idx, n_frames, n_seg, halo, vfull=False):
    """per scene: query rows and key rows (global indices in the packed sequence)"""
    dev = video_idx.device
    rows_per_frame = video_idx.numel() // n_frames
    vid = video_idx.view(n_frames, rows_per_frame)
    n_audio = audio_idx.numel() // 2
    aud = audio_idx.view(2, n_audio)
    vseg = torch.arange(n_frames, device=dev) * n_seg // n_frames
    aseg = torch.arange(n_audio, device=dev) * n_seg // n_audio
    t_off = np.cumsum([0] + St.counts)
    frames = [(vseg == j).nonzero().flatten().tolist() for j in range(n_seg)]
    St.rows = {'vid': vid, 'vseg': vseg, 'text': text_idx, 't_off': t_off}
    _, n_kf = setting_parts()
    subj_rows = None
    if n_kf and St.subj_mask is not None:
        subj_rows = []
        for fr in frames:
            kfs = [fr[int(round(x))] for x in np.linspace(0, len(fr) - 1, n_kf + 2)[1:-1]]
            subj_rows.append(torch.cat([vid[f][St.subj_mask[f]] for f in kfs]))
    qs, ks = [], []
    scope = torch.zeros(n_frames, n_frames, dtype=torch.bool, device=dev)    # [query frame, key frame] inside the scope
    for j in range(n_seg):
        q = torch.cat([text_idx[t_off[j]:t_off[j + 1]], aud[:, aseg == j].flatten(), vid[vseg == j].flatten()])
        sc = list(frames[j])
        for i in (j - 1, j + 1):
            if halo and 0 <= i < n_seg:
                sc += frames[i][-halo:] if i < j else frames[i][:halo]
        scope[torch.tensor(frames[j], device=dev)[:, None], torch.tensor(sc, device=dev)[None]] = True
        extra = []
        if halo:
            for i in (j - 1, j + 1):
                if 0 <= i < n_seg:
                    hf = frames[i][-halo:] if i < j else frames[i][:halo]
                    extra.append(vid[hf].flatten())
                    lo, hi = min(hf) * n_audio // n_frames, (max(hf) + 1) * n_audio // n_frames
                    extra.append(aud[:, lo:hi].flatten())
        if subj_rows is not None:
            extra += [subj_rows[i] for i in range(n_seg) if i != j]
        if vfull:                       # text / audio queries keep the scope; video queries also read every video row
            ta = torch.cat([text_idx[t_off[j]:t_off[j + 1]], aud[:, aseg == j].flatten()])
            qs += [ta, vid[vseg == j].flatten()]
            ks += [torch.cat([q] + extra), torch.cat([ta] + [x for x in extra if x.numel() and not bool(torch.isin(x[:1], video_idx))] + [video_idx])]
        else:
            qs.append(q)
            ks.append(torch.cat([q] + extra))
    lq, lk = [len(x) for x in qs], [len(x) for x in ks]
    cu = lambda x: torch.tensor([0] + np.cumsum(x).tolist(), dtype=torch.int32, device=dev)
    return dict(q=torch.cat(qs), k=torch.cat(ks), cu_q=cu(lq), cu_k=cu(lk), max_q=max(lq), max_k=max(lk),
                pairs=float(sum(a * b for a, b in zip(lq, lk))), seq=seq_len, scope=scope, vid=vid)


def subj_detect(layer, q, k):
    """accumulate video-to-subject-token logits at forward SUBJ_STEP over SUBJ_LAYERS; build the mask after the last"""
    _, n_kf = setting_parts()
    if not n_kf or St.subj_mask is not None or St.fwd != SUBJ_STEP or St.rows is None:
        return
    if SUBJ_LAYERS[0] <= layer < SUBJ_LAYERS[1]:
        vid, vseg, text, t_off = St.rows['vid'], St.rows['vseg'], St.rows['text'], St.rows['t_off']
        score = torch.zeros(vid.shape, device=q.device)
        for j in range(len(St.counts)):
            ks = k[text[[t_off[j] + t for t in St.subj_tok[j]]]].float()
            fr = (vseg == j).nonzero().flatten()
            qq = q[vid[fr].flatten()].float()
            score[fr] = (torch.einsum('shd,nhd->s', qq, ks) / (ks.shape[0] * ks.shape[1])).view(len(fr), -1)
        St.subj_score = score if St.subj_score is None else St.subj_score + score
    if layer == SUBJ_LAYERS[1] - 1:
        h, w = St.grid
        sc = torch.nn.functional.avg_pool2d(St.subj_score.view(-1, 1, h, w), 3, 1, 1, count_include_pad=False).view(St.subj_score.shape)
        St.subj_mask = sc >= sc.quantile(1 - SUBJ_TOP, dim=1, keepdim=True)


def spat_merge(o, lse, p, q, k, v):
    """add, for every video row, the keys at the same spatial position in all out-of-scope frames; exact LSE merge"""
    lse_full = torch.empty(q.shape[1], q.shape[0], device=q.device, dtype=torch.float32)
    lse_full[:, p['q']] = lse.float()
    vid, scope = p['vid'], p['scope']                               # [F, R], [F, F]
    qv, kv, vv = (x[vid].float() for x in (q, k, v))                # [F, R, H, D]
    lg = torch.einsum('frhd,grhd->rhfg', qv, kv) / math.sqrt(q.shape[-1])
    lg = lg.masked_fill(scope[None, None], float('-inf'))
    lse_b = torch.logsumexp(lg, -1)                                 # [R, H, F]
    ob = torch.einsum('rhfg,grhd->frhd', torch.softmax(lg, -1).nan_to_num(), vv)
    lse_a = lse_full[:, vid].permute(2, 0, 1)                       # [R, H, F]
    m = torch.maximum(lse_a, lse_b)
    wa, wb = (lse_a - m).exp(), (lse_b - m).exp().nan_to_num()
    wa, wb = wa.permute(2, 0, 1)[..., None], wb.permute(2, 0, 1)[..., None]
    o = o.clone()
    o[vid] = ((o[vid].float() * wa + ob * wb) / (wa + wb)).to(o.dtype)
    return o


def install(transformer):
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    class Proc:
        def __init__(self, layer):
            self.layer = layer

        def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
            if attn.fused_projections:
                q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
            q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in (q, k, v))
            q, k = attn.norm_q(q), attn.norm_k(k)
            if rotary_emb is not None:
                q, k = _apply_rotary_emb(q, *rotary_emb), _apply_rotary_emb(k, *rotary_emb)
            q, k, v = (x[0].to(torch.bfloat16) for x in (q, k, v))
            subj_detect(self.layer, q, k)
            if St.probe is not None:
                St.probe(St.fwd - 1, self.layer, q[St.vid_idx], k[St.vid_idx], St.rows_per_frame)
            if St.capture is not None:
                St.capture(St.fwd - 1, self.layer, q[St.vid_idx], k[St.vid_idx], v[St.vid_idx], St.rows_per_frame)
            if St.capture_full is not None:
                St.capture_full(St.fwd - 1, self.layer, q, k, v, St)
            if St.setting in ('check_dense', 'probe_dense', 'capture_dense', 'solprof_dense'):
                o = flash_attn_func(q[None], k[None], v[None])[0]
            else:
                vfull, vlayers, spat = variant_parts()
                p = St.plan_full if vfull and (vlayers is None or self.layer < vlayers) else St.plan
                oq, lse, _ = flash_attn_varlen_func(q[p['q']], k[p['k']], v[p['k']], p['cu_q'], p['cu_k'], p['max_q'],
                                                    p['max_k'], return_attn_probs=True)
                o = torch.zeros_like(q)
                o[p['q']] = oq
                if spat:
                    o = spat_merge(o, lse, p, q, k, v)
            o = o.flatten(1, 2)[None].type_as(hidden_states)
            return attn.to_out[1](attn.to_out[0](o))
    procs = {'orig': [b.attn.processor for b in transformer.transformer_blocks],
             'ours': [Proc(i) for i in range(len(transformer.transformer_blocks))]}

    def pre(m, args, kwargs):
        if St.setting == 'baseline':
            for b, p in zip(m.transformer_blocks, procs['orig']):
                b.attn.processor = p
            return
        St.fwd += 1
        for b, p in zip(m.transformer_blocks, procs['ours']):
            b.attn.processor = p
        St.vid_idx = kwargs['video_indices']
        if St.setting not in ('check_dense', 'probe_dense'):
            halo, _ = setting_parts()
            n_frames = kwargs['hidden_states'].shape[1] // St.rows_per_frame
            St.plan = build_plan(kwargs['position_ids'].shape[0], kwargs['video_indices'], kwargs['audio_indices'],
                                 kwargs['text_indices'], n_frames, len(St.counts), halo)
            if variant_parts()[0]:
                St.plan_full = build_plan(kwargs['position_ids'].shape[0], kwargs['video_indices'], kwargs['audio_indices'],
                                          kwargs['text_indices'], n_frames, len(St.counts), halo, vfull=True)
    transformer.register_forward_pre_hook(pre, with_kwargs=True)
    transformer.token_refiner = SplitRefiner(transformer.token_refiner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--settings', default='joint_sep_grope_adj2')
    ap.add_argument('--ids', default='L1_CAB')
    ap.add_argument('--height', type=int, default=544)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--frames', type=int, default=124)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--probe-out', type=Path, default=None)
    ap.add_argument('--min-seconds', type=float, default=None,
                    help='lower the pipeline\'s 5 s minimum (a software check, not a model limit) so separate clips are '
                         'generated at their own length, snapped up to 17n+5 frames, instead of >= 124 frames and cut')
    ap.add_argument('--bench', type=Path, default=BENCH, help='prompt file: {"prompts": [{id, prompt, sentences}]}')
    ap.add_argument('--check', action='store_true', help='compare check_dense with the original processor on one forward')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from diffusers import ComponentsManager, ModularPipeline
    min_frames = H3_MIN_FRAMES
    if a.min_seconds is not None:
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
        MiniMaxH3ModularPipeline.min_duration = property(lambda self: a.min_seconds)
        min_frames = 1
    from diffusers.utils import export_to_video
    bench = {r['id']: r for r in json.loads(a.bench.read_text())['prompts']}
    workflow = ModularPipeline.from_pretrained(REPO).blocks.get_workflow('t2va')
    # default: text encoder on cuda:1, transformer on cuda:0. H3_SINGLE_GPU=1: everything on cuda:0 under one manager,
    # which offloads the text encoder to CPU when the transformer needs the memory (end-to-end time on one GPU)
    single = os.environ.get('H3_SINGLE_GPU') == '1'
    tm = ComponentsManager()
    tm.enable_auto_cpu_offload(device='cuda:0' if single else 'cuda:1')
    conditioner = workflow.sub_blocks.pop('text_encoder').init_pipeline(REPO, components_manager=tm)
    conditioner.load_components(dtype=torch.bfloat16)
    mm = tm if single else ComponentsManager()
    if not single:
        mm.enable_auto_cpu_offload(device='cuda:0')
    rest = workflow.init_pipeline(REPO, components_manager=mm)
    rest.load_components(dtype=torch.bfloat16)
    tr = rest.transformer
    St.rows_per_frame = (a.height // 32) * (a.width // 32)
    St.grid = (a.height // 32, a.width // 32)
    install(tr)
    fwd = []
    def t_pre(m, *_):
        torch.cuda.synchronize('cuda:0')
        m._t0 = time.perf_counter()

    def t_post(m, *_):
        torch.cuda.synchronize('cuda:0')
        fwd.append(time.perf_counter() - m._t0)
    tr.register_forward_pre_hook(t_pre)
    tr.register_forward_hook(t_post)
    kw = dict(height=a.height, width=a.width, num_frames=a.frames)
    if a.check:
        cap = {}

        def grab(m, args, kwargs, out):
            cap['o'] = out[0].float().clone() if isinstance(out, tuple) else out.sample.float().clone()
            raise StopIteration
        h = tr.register_forward_hook(grab, with_kwargs=True)
        r = bench[a.ids.split(',')[0]]
        outs = {}
        for s in ('baseline', 'check_dense', 'joint_sep_grope'):
            St.setting = s
            state = sentence_state(conditioner, r['sentences']) if s != 'baseline' else conditioner(prompt=' '.join(r['sentences']))
            if s == 'baseline':
                St.counts = []
            try:
                rest(state=state, generator=torch.Generator().manual_seed(a.seed), output=['videos'], **kw)
            except StopIteration:
                pass
            outs[s] = cap['o']
        h.remove()
        rel = lambda x, y: float((x - y).norm() / y.norm())
        print(f"[check] separate-text dense vs joint_sep_grope rel {rel(outs['joint_sep_grope'], outs['check_dense']):.3e}; "
              f"baseline vs separate-text dense (different text, not a numeric check) rel {rel(outs['check_dense'], outs['baseline']):.3e}", flush=True)
        St.setting = 'check_dense'
        state = conditioner(prompt=' '.join(r['sentences']))
        St.counts = [int(state.get('text_token_tags').shape[-1])]
        h = tr.register_forward_hook(grab, with_kwargs=True)
        try:
            rest(state=state, generator=torch.Generator().manual_seed(a.seed), output=['videos'], **kw)
        except StopIteration:
            pass
        h.remove()
        print(f"[check] numeric: our processor (dense) vs original, same text: rel {rel(cap['o'], outs['baseline']):.3e}", flush=True)
        return
    for setting in a.settings.split(','):
        St.setting = setting
        for pid in a.ids.split(','):
            d = a.out / setting / pid
            if (d / 'video.mp4').exists():
                continue
            r = bench[pid]
            if setting == 'probe_dense':
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from block_attention_stats import BlockStats
                L = len(tr.transformer_blocks)
                a.probe_out.mkdir(parents=True, exist_ok=True)
                if (a.probe_out / f'{pid}.json').exists():
                    continue
                St.fwd, St.rows = 0, None
                state = conditioner(prompt=r['prompt'])
                St.counts = [int(state.get('text_token_tags').shape[-1])]
                n_steps = 49
                St.probe = BlockStats([round(i * (L - 1) / 6) for i in range(7)], {round(f * (n_steps - 1)) for f in KEEP_FRAC})
                rest(state=state, generator=torch.Generator().manual_seed(a.seed), output=['videos'], **kw)
                res = St.probe.result()
                res.update({'model': 'h3', 'id': pid, 'steps': St.fwd, 'F': St.vid_idx.numel() // St.rows_per_frame, 'T': St.rows_per_frame})
                (a.probe_out / f'{pid}.json').write_text(json.dumps(res))
                torch.save(St.probe.maps, a.probe_out / f'{pid}_maps.pt')
                print(f'[probe] h3 {pid}: {len(St.probe.need)} maps, {St.fwd} forwards', flush=True)
                St.probe = None
                continue
            if setting == 'solprof_dense':
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent / 'official_warmup'))
                from sol_h3_profile import SolProfile
                from tail_compression import StopGeneration
                a.probe_out.mkdir(parents=True, exist_ok=True)
                St.fwd, St.rows = 0, None
                state = conditioner(prompt=r['prompt'])
                St.counts = [int(state.get('text_token_tags').shape[-1])]
                St.capture_full = SolProfile()
                try:
                    rest(state=state, generator=torch.Generator().manual_seed(a.seed), output=['videos'], **kw)
                except StopGeneration:
                    pass
                (a.probe_out / f'solprof_{pid}.json').write_text(json.dumps(St.capture_full.result(), indent=1))
                print(f'[solprof] {pid}: {len(St.capture_full.rows)} rows', flush=True)
                St.capture_full = None
                continue
            if setting == 'capture_dense':
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from tail_compression import StopGeneration, TailProbe
                a.probe_out.mkdir(parents=True, exist_ok=True)
                St.fwd, St.rows = 0, None
                state = conditioner(prompt=r['prompt'])
                St.counts = [int(state.get('text_token_tags').shape[-1])]
                St.capture = TailProbe((8, 24, 41), (0, 10, 29), len(r['sentences']))
                try:
                    rest(state=state, generator=torch.Generator().manual_seed(a.seed), output=['videos'], **kw)
                except StopGeneration:
                    pass
                (a.probe_out / f'h3_{pid}.json').write_text(json.dumps({'model': 'h3', 'id': pid, 'frames': a.frames,
                                                                         'scenes': len(r['sentences']), 'rows': St.capture.rows}, indent=1))
                print(f'[tail] h3 {pid}: {len(St.capture.rows)} rows', flush=True)
                St.capture = None
                continue
            fwd.clear()
            St.fwd, St.subj_score, St.subj_mask, St.rows = 0, None, None, None
            St.subject = r.get('subject', '') if setting_parts()[1] else ''
            torch.cuda.synchronize('cuda:0')
            t0 = time.perf_counter()
            if setting == 'separate':
                St.setting, St.counts = 'baseline', []
                vids, auds = [], []
                K = len(r['sentences'])
                clip_frames = CLIP_FRAMES if K == 3 else [a.frames // K + (j < a.frames % K) for j in range(K)]
                for sent, n in zip(r['sentences'], clip_frames):
                    m = max(n, min_frames)
                    while m % 17 != 5:
                        m += 1
                    o = rest(state=conditioner(prompt=sent), generator=torch.Generator().manual_seed(a.seed),
                             output=['videos', 'audio', 'sampling_rate'], **{**kw, 'num_frames': m})
                    v = o['videos'][0] if isinstance(o['videos'], (list, tuple)) else o['videos']
                    vids += (list(v) if not torch.is_tensor(v) else list(v.float().cpu().numpy()))[:n]
                    x = o['audio'][0] if isinstance(o['audio'], (list, tuple)) else o['audio']
                    x = (x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)).squeeze()
                    sr = int(o['sampling_rate'])
                    auds.append(x[..., :int(round(n / 24 * sr))])
                St.setting = 'separate'
                out = {'videos': [vids], 'audio': [np.concatenate(auds, -1)], 'sampling_rate': sr}
            else:
                if setting == 'baseline':
                    St.counts = []
                    state = conditioner(prompt=r['prompt'])
                else:
                    state = sentence_state(conditioner, r['sentences'])
                out = rest(state=state, generator=torch.Generator().manual_seed(a.seed), output=['videos', 'audio', 'sampling_rate'], **kw)
            torch.cuda.synchronize('cuda:0')
            total = time.perf_counter() - t0
            v = out['videos'][0] if isinstance(out['videos'], (list, tuple)) else out['videos']
            frames = list(v) if not torch.is_tensor(v) else list(v.float().cpu().numpy())
            d.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(d / 'video.mp4'), fps=24)
            try:
                import soundfile as sf
                x = out['audio'][0] if isinstance(out['audio'], (list, tuple)) else out['audio']
                x = (x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)).squeeze()
                sf.write(str(d / 'audio.wav'), x.T if x.ndim == 2 and x.shape[0] <= 2 else x, int(out['sampling_rate']))
            except Exception as ex:
                print('[audio] not saved:', ex, flush=True)
            p = St.plan
            meta = {'model': 'h3', 'setting': setting, 'id': pid, 'seed': a.seed, 'frames': len(frames), 'height': a.height,
                    'width': a.width, 'seconds': total, 'denoise_s': sum(fwd), 'forwards': len(fwd),
                    'per_forward_s': float(np.median(fwd)) if fwd else None, 'text_tokens': St.counts,
                    'attn_pairs_fraction': (p['pairs'] / p['seq'] ** 2) if (p and setting.startswith('joint')) else 1.0,
                    'subject_mask_fraction': float(St.subj_mask.float().mean()) if St.subj_mask is not None else None}
            (d / 'meta.json').write_text(json.dumps(meta, indent=1))
            print(f"[done] h3 {setting}/{pid} {total:.0f}s denoise {sum(fwd):.0f}s attn fraction {meta['attn_pairs_fraction']:.3f}", flush=True)


if __name__ == '__main__':
    main()
