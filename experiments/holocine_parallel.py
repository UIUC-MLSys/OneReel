"""HoloCine (Wan2.2-T2V-A14B fine-tuned for multi-shot video, sparse inter-shot attention: each shot's queries read
their shot and the first latent frame of every shot) on N GPUs, one request per method:
  official   HoloCine's own pipeline on one GPU (the released code has no working multi-GPU path for the sparse
             model: its xDiT / USP hook drops the shot cuts and the shot text mask)
  ulysses    sequence parallel by heads: equal token shards, all-to-all to head shards around HoloCine's shot
             attention and back
  onereel   K/V gathering: Algorithm 1 token partition (onereel.parallel.comm.balanced, per-query cost from the shot
             and shared keys it reads), one all_to_all fetching only the K/V rows a GPU's queries read (their shots and
             every shot's first frame), then HoloCine's packed varlen attention on the GPU's own queries
  tokens     K/V gathering with an equal-token partition (ablation)
Token-wise layers (projections, cross-attention with the shot text mask, FFN) run on each GPU's own tokens; the head
output is gathered once per forward. Every rank holds the whole model (both experts and the T5 / VAE, bf16).

Correctness: rank 0 saves the inputs and output of each parallel method's first forward (OUT/check_<id>_<method>.pt);
`--methods check` (one GPU) runs HoloCine's model_fn on those inputs and records the relative error.
Timing: a request of --steps denoising steps (CFG: 2 forwards per step), every forward CUDA-synchronised; per-forward
time = mean over the steps after the first; e2e estimate = text + 100 forwards (50 steps) + VAE.
--save: rank 0 also writes the decoded video to OUT/<id>_<method>.mp4 (15 fps), e.g. with --steps 50 for samples.
Output: one JSON line per method to OUT/holocine_parallel.jsonl, printed as [hcpar] ...
Usage: torchrun --nproc_per_node 2 experiments/holocine_parallel.py --id holo_1 --methods ulysses,onereel,tokens
       python experiments/holocine_parallel.py --id holo_1 --methods official
"""
import argparse
import json
import os
import sys
import time
from itertools import accumulate
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))
from holocine_gen import CKPT, HC, NEG, TOKENIZER, holocine_prompt, shot_cuts, transformers_compat  # noqa: E402

sys.path[:0] = [str(ROOT / 'experiments/stubs'), str(HC), str(ROOT / 'data/envs/holocine_extra')]
from onereel.parallel.comm import C_ATT, C_LIN, balanced, clip, merge  # noqa: E402

transformers_compat()

# the cost model was fitted on HunyuanVideo-1.5 (hidden 2048); token-wise work per token grows with hidden^2 and
# attention per query-key pair with hidden, so the attention weight scales by 2048 / 5120 for Wan2.2-A14B
C_ATT_WAN = C_ATT * 2048 / 5120
CTX = {}


def plan(cuts, per_g, n, mode):
    """per rank (lo, hi) over the video tokens, and the K/V rows every rank reads (sorted disjoint intervals)"""
    shots = list(zip(cuts[:-1], cuts[1:]))
    S = cuts[-1]
    if mode in ('ulysses', 'tokens'):
        own = [(r * S // n, (r + 1) * S // n) for r in range(n)]
    else:
        G = sum(min(per_g, b - a) for a, b in shots)
        own = balanced([b - a for a, b in shots], [C_LIN + C_ATT_WAN * (b - a + G) for a, b in shots], n)
    shared = [(a, a + min(per_g, b - a)) for a, b in shots]
    need = [merge([s for s in shots if s[0] < hi and s[1] > lo] + shared) for lo, hi in own]
    return shots, own, need


def a2a(x):
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    return out


def ulysses_attn(q, k, v, H):
    """q, k, v [1, s, H*d] on equal token shards -> attention over the whole sequence on H / n heads -> [1, s, H*d]"""
    from diffsynth.models.wan_video_dit import attention_per_batch_with_shots
    n, s = dist.get_world_size(), q.shape[1]
    d = q.shape[2] // H

    def to_heads(t):                     # [n, s, H/n, d], dim 0 = head group; after a2a dim 0 = source token shard
        return a2a(t.view(s, n, H // n, d).transpose(0, 1).contiguous()).reshape(1, n * s, H // n * d)
    o = attention_per_batch_with_shots(to_heads(q), to_heads(k), to_heads(v), [CTX['cuts']], num_heads=H // n,
                                       per_g=CTX['per_g'])
    o = a2a(o.view(n, s, H // n, d).contiguous())          # dim 0 = head group of my token shard
    return o.transpose(0, 1).reshape(1, s, H * d)


def gather_attn(q, k, v, H):
    """K/V gathering: fetch the K/V rows my queries read, then HoloCine's packed varlen shot attention on my queries"""
    from diffsynth.models.wan_video_dit import flash_attn_varlen_func
    me, n = dist.get_rank(), dist.get_world_size()
    shots, own, need = CTX['shots'], CTX['own'], CTX['need']
    lo, hi = own[me]
    hd = k.shape[2]
    kv = torch.stack([k[0], v[0]], 1)                       # [s, 2, H*d]
    send = [torch.cat([kv[a - lo:b - lo] for a, b in clip(need[r], lo, hi)] or [kv[:0]]) if r != me else kv[:0]
            for r in range(n)]
    recv_iv = [clip(need[me], *own[j]) if j != me else [] for j in range(n)]
    recv = torch.empty(sum(b - a for iv in recv_iv for a, b in iv), 2, hd, dtype=kv.dtype, device=kv.device)
    dist.all_to_all_single(recv, torch.cat(send), [sum(b - a for a, b in iv) for iv in recv_iv],
                           [t.shape[0] for t in send])
    buf = CTX.setdefault('buf', {}).get(kv.shape[1:])
    if buf is None or buf.shape[0] < CTX['S']:
        buf = CTX['buf'][kv.shape[1:]] = torch.empty(CTX['S'], 2, hd, dtype=kv.dtype, device=kv.device)
    buf[lo:hi] = kv
    p = 0
    for iv in recv_iv:
        for a, b in iv:
            buf[a:b] = recv[p:p + b - a]
            p += b - a
    d = hd // H
    glob = torch.cat([buf[a:a + min(CTX['per_g'], b - a)] for a, b in shots])
    mine = [(a, b) for a, b in shots if a < hi and b > lo]
    kv_p = torch.cat([t for a, b in mine for t in (buf[a:b], glob)]).view(-1, 2, H, d)
    nq = [min(b, hi) - max(a, lo) for a, b in mine]
    nk = [b - a + glob.shape[0] for a, b in mine]
    cu = lambda x: torch.tensor([0] + list(accumulate(x)), dtype=torch.int32, device=q.device)
    o = flash_attn_varlen_func(q[0].view(-1, H, d), kv_p[:, 0], kv_p[:, 1], cu(nq), cu(nk), max(nq), max(nk),
                               softmax_scale=None, causal=False)
    if isinstance(o, tuple):
        o = o[0]
    return o.reshape(1, -1, H * d)


def self_attn_forward(self, x, freqs, shot_latent_indices=None, per_g=0):
    from diffsynth.models.wan_video_dit import rope_apply
    q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
    k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
    v = self.v(x)
    f = ulysses_attn if CTX['mode'] == 'ulysses' else gather_attn
    return self.o(f(q, k, v, self.num_heads))


def model_fn_parallel(dit, latents=None, timestep=None, context=None, shot_indices=None, text_cut_positions=None,
                      shot_mask_type=None, **kwargs):
    """model_fn_wan_video of HoloCine (T2V path: no VACE / TeaCache / reference / image input) with the blocks on this
    rank's tokens"""
    import diffsynth.pipelines.wan_video_holocine as P
    assert shot_mask_type is None and not kwargs.get('fuse_vae_embedding_in_latents'), 'T2V path only'
    x = latents
    t = dit.time_embedding(P.sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    x, (f, h, w) = dit.patchify(x, None)
    if dit.shot_embedding is not None and shot_indices is not None:
        x = x + dit.shot_embedding(shot_indices.repeat_interleave(h * w, dim=1))
    freqs = torch.cat([dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                       dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                       dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    attn_mask = text_mask(x, context, shot_indices, text_cut_positions, h * w)
    cuts = [int(c) for c in P.labels_to_cuts(shot_indices.repeat_interleave(h * w, dim=1))[0]]
    key = (tuple(cuts), h * w, CTX['mode'])
    if CTX.get('key') != key:
        CTX['shots'], CTX['own'], CTX['need'] = plan(cuts, h * w, dist.get_world_size(), CTX['mode'])
        CTX.update(key=key, cuts=cuts, per_g=h * w, S=cuts[-1])
    lo, hi = CTX['own'][dist.get_rank()]
    x, freqs = x[:, lo:hi], freqs[lo:hi]
    if attn_mask is not None:
        attn_mask = attn_mask[:, :, lo:hi]
    for block in dit.blocks:
        x = block(x, context, t_mod, freqs, attn_mask, None, h * w)
    x = dit.head(x, t)
    sizes = [b - a for a, b in CTX['own']]
    parts = [torch.empty(1, m, x.shape[2], dtype=x.dtype, device=x.device) for m in sizes]
    xs = x.contiguous()
    if len(set(sizes)) == 1:
        dist.all_gather(parts, xs)
    else:                                                    # uneven shards: pad to the largest
        m = max(sizes)
        pad = torch.zeros(1, m, x.shape[2], dtype=x.dtype, device=x.device)
        pad[:, :xs.shape[1]] = xs
        tmp = [torch.empty_like(pad) for _ in sizes]
        dist.all_gather(tmp, pad)
        parts = [t[:, :s] for t, s in zip(tmp, sizes)]
    return dit.unpatchify(torch.cat(parts, 1), (f, h, w))


def text_mask(x, context, shot_indices, text_cut_positions, hw):
    """HoloCine's window cross-attention bias (each video token reads the global caption and its shot's caption)"""
    if text_cut_positions is None or text_cut_positions.get('global') is None:
        return None
    L = context.shape[1]
    g0, g1 = map(int, text_cut_positions['global'])
    ranges = text_cut_positions['shots']
    dev = x.device
    glob = torch.zeros(L, dtype=torch.bool, device=dev)
    glob[g0:g1 + 1] = True
    if ranges:
        table = torch.zeros(len(ranges), L, dtype=torch.bool, device=dev)
        for sid, (s0, s1) in enumerate(ranges):
            table[sid, int(s0):int(s1) + 1] = True
        allow = table[shot_indices.repeat_interleave(hw, dim=1)] | glob.view(1, 1, L)
    else:
        allow = glob.view(1, 1, L)
    end = max([g1] + [int(r[1]) for r in ranges]) + 1
    pad = torch.zeros(L, dtype=torch.bool, device=dev)
    if end < L:
        pad[end:] = True
    allow = allow | pad.view(1, 1, L)
    return torch.zeros(allow.shape, dtype=x.dtype, device=dev).masked_fill(~allow, -1e4).unsqueeze(1)


def load():
    from diffsynth.pipelines.wan_video_holocine import ModelConfig, WanVideoHoloCinePipeline
    pipe = WanVideoHoloCinePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=str(CKPT / 'Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth')),
                       ModelConfig(path=str(CKPT / 'HoloCine_dit/sparse/sparse_high_noise.safetensors')),
                       ModelConfig(path=str(CKPT / 'HoloCine_dit/sparse/sparse_low_noise.safetensors')),
                       ModelConfig(path=str(CKPT / 'Wan2.2-T2V-A14B/Wan2.1_VAE.pth'))],
        tokenizer_config=ModelConfig(path=str(TOKENIZER)))
    pipe.dit.use_sparse_self_attn = pipe.dit2.use_sparse_self_attn = True
    pipe.to('cuda')
    dt = {n: str(next(getattr(pipe, n).parameters()).dtype) for n in ('dit', 'dit2', 'text_encoder', 'vae')}
    print(f'[hcpar-load] dtypes {dt} allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB', flush=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default='holo_1')
    ap.add_argument('--methods', default='ulysses,onereel,tokens')
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/holocine_parallel')
    ap.add_argument('--save', action='store_true', help='write the decoded video (rank 0)')
    a = ap.parse_args()
    par = 'WORLD_SIZE' in os.environ
    if par:
        dist.init_process_group('nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    rank, n = (dist.get_rank(), dist.get_world_size()) if par else (0, 1)
    r = {x['id']: x for x in json.load(open(ROOT / 'dataset/multishot_bench.json'))['prompts']}[a.id]
    pipe = load()
    import diffsynth.pipelines.wan_video_holocine as P
    official = P.model_fn_wan_video
    times, first = [], {}

    def timed(fn):
        def run(**kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(**kw)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            if 'out' not in first:
                first.update(out=out.detach().clone(), kw={k: v for k, v in kw.items()})
            return out
        return run

    kwargs = dict(prompt=holocine_prompt(r), negative_prompt=NEG, seed=0, tiled=True, height=480, width=832,
                  num_inference_steps=a.steps, num_frames=r['frames'], shot_cut_frames=shot_cuts(r))
    a.out.mkdir(parents=True, exist_ok=True)
    for method in a.methods.split(','):
        if method == 'check':
            assert not par
            pipe.text_encoder.to('cpu')
            pipe.vae.to('cpu')
            for f in sorted(a.out.glob(f'check_{a.id}_*.pt')):
                c = torch.load(f, map_location='cuda', weights_only=False)
                other = 'dit2' if c['kw']['dit'] == 'dit' else 'dit'
                getattr(pipe, other).to('cpu')
                torch.cuda.empty_cache()
                kw = dict(c['kw'], dit=getattr(pipe, c['kw']['dit']))
                kw.update({m: None for m in ('motion_controller', 'vace') if m not in kw})
                with torch.no_grad():
                    ref = official(**kw).float()
                err = ((c['out'].cuda().float() - ref).norm() / ref.norm()).item()
                row = dict(id=a.id, method=f.stem[len(f'check_{a.id}_'):], check='rel_err vs model_fn_wan_video', rel_err=err)
                print('[hcpar]', json.dumps(row), flush=True)
                with open(a.out / 'holocine_parallel.jsonl', 'a') as fh:
                    fh.write(json.dumps(row) + '\n')
                del ref, c, kw
                torch.cuda.empty_cache()
            continue
        if method == 'official':
            assert not par, 'official runs on one GPU'
            pipe.model_fn = timed(official)
        else:
            assert par, f'{method} needs torchrun'
            CTX.clear()
            CTX['mode'] = method
            for dit in (pipe.dit, pipe.dit2):
                for b in dit.blocks:
                    b.self_attn.forward = self_attn_forward.__get__(b.self_attn)
            pipe.model_fn = timed(model_fn_parallel)
        times.clear()
        first.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video = pipe(**kwargs)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        fwd = times[2:] if len(times) > 2 else times
        per_fwd = sum(fwd) / len(fwd)
        rest = total - sum(times)
        row = dict(id=a.id, method=method, gpus=n, frames=r['frames'], lengths=r['lengths'], steps=a.steps,
                   forwards=len(times), forward_s=per_fwd, rest_s=rest, e2e_est_s=rest + 100 * per_fwd,
                   own=CTX.get('own'), seconds=total)
        if a.save and rank == 0:
            from diffsynth import save_video
            save_video(video, str(a.out / f'{a.id}_{method}.mp4'), fps=15, quality=5)
            row['video'] = f'{a.id}_{method}.mp4'
        if method != 'official' and rank == 0:               # checked against HoloCine's model_fn by --methods check
            kw = {k: v for k, v in first['kw'].items() if not isinstance(v, torch.nn.Module)}
            kw['dit'] = 'dit2' if first['kw']['dit'] is pipe.dit2 else 'dit'
            torch.save(dict(kw=kw, out=first['out'].cpu()), a.out / f'check_{a.id}_{method}.pt')
        if par:
            dist.barrier()
        if rank == 0:
            print('[hcpar]', json.dumps(row), flush=True)
            with open(a.out / 'holocine_parallel.jsonl', 'a') as fh:
                fh.write(json.dumps(row) + '\n')
    if par:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
