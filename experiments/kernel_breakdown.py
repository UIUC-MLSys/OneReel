"""Kernel-level breakdown of OneReel's scene attention on the scopes it runs (one GPU, FlashAttention-3, no model).

Scopes at 720p, scenes of F_SCENE latent frames, as in the latency study:
  hunyuan   T = 3600 tokens per frame (720 x 1280), 16 heads, joint text (TEXT tokens per sentence). Video queries of
            scene j read its frames, the frame memory (MEM representative frames shared by all scenes, those outside j)
            and sentence j; sentence j's queries read scene j's frames and sentence j. Memory frames are scattered.
  wan       T = 880 (704 x 1280), 24 heads, self-attention only. Scene j reads its frames and the first HALO frames of
            the next scene (one contiguous key range).
Layouts: K equal scenes (K in --scenes) and 'uneven' (12 scenes of 4-10 frames, 84 in all).

Implementations (outputs checked against an fp32 masked reference on sampled query rows):
  dense         one FA3 call over all tokens (computes every pair; the throughput reference)
  flex          FlexAttention in the original token order, block mask from the scope (built once per request:
                mask_ms, not in ms); mask_mod evaluated on partial blocks only
  fi_bsr        FlashInfer 0.3.1 block-sparse attention, 128 x 128 blocks in the original order: full blocks on its fa3
                backend without mask, partial blocks with the token mask (fa2; fa3 has no custom mask), merged by LSE
                (plan once per request, not in ms) -- the fastest sparse path of Table 2
  per_scene     the packed segments, one FA3 call per segment (2K calls for Hunyuan, K for Wan): prep + kernel
  packed        OneReel: queries cut per segment, the scope's K/V rows copied per segment (prep), one FA3 varlen call
                (kernel), outputs written back to their token rows (post)
  zero_copy     Wan only: every scene's keys are one contiguous range, read in place by FA3 varlen through
                cu_seqlens_k offsets and seqused_k (no copy)
Per implementation: ms (prep + kernel + post), prep_ms, kernel_ms, post_ms, err, useful TFLOP/s (FLOPs of the scope's
pairs) and eff = useful TFLOP/s / dense FA3 TFLOP/s measured in the same layout. Writes OUT/kernel_breakdown.json
after every layout.

Usage (onereel env, FlashInfer 0.3.1 on PYTHONPATH; FLASHINFER_WORKSPACE_BASE for its JIT cache):
  python experiments/kernel_breakdown.py --out DIR [--models hunyuan,wan] [--scenes 3,6,12,24] [--no-uneven]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import flash_attn_interface as fa3
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

MODELS = {'hunyuan': dict(T=3600, H=16, text=100, scope='memory'),
          'wan': dict(T=880, H=24, text=0, scope='halo')}
D, F_SCENE, MEM, HALO, BLOCK = 128, 7, 4, 2, 128
UNEVEN = [4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5]
flex = torch.compile(flex_attention, dynamic=False)
first = lambda x: x[0] if isinstance(x, tuple) else x


def timed(fn, n):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


class Scope:
    """token layout [video frames | sentence 0 | sentence 1 | ...]; units = frames, then sentences"""

    def __init__(self, model, lengths):
        cfg = MODELS[model]
        self.T, self.H, self.L, self.K = cfg['T'], cfg['H'], cfg['text'], len(lengths)
        s = np.cumsum([0] + lengths)
        self.F = int(s[-1])
        self.frames = [list(range(int(s[j]), int(s[j + 1]))) for j in range(self.K)]
        if cfg['scope'] == 'memory':
            mem = sorted({int(f) for f in np.linspace(0, self.F - 1, MEM).round()})
            self.keys = [fr + [f for f in mem if f not in fr] for fr in self.frames]
        else:
            self.keys = [fr + (self.frames[j + 1][:HALO] if j + 1 < self.K else []) for j, fr in enumerate(self.frames)]
        self.S = self.F * self.T + self.K * self.L
        U = self.F + (self.K if self.L else 0)
        allow = torch.zeros(U, U, dtype=torch.bool)
        for j in range(self.K):
            allow[torch.tensor(self.frames[j])[:, None], torch.tensor(self.keys[j])[None]] = True
            if self.L:
                t = self.F + j
                allow[torch.tensor(self.frames[j]), t] = True
                allow[t, torch.tensor(self.frames[j])] = True
                allow[t, t] = True
        self.allow = allow.cuda()
        unit = torch.arange(self.F).repeat_interleave(self.T)
        if self.L:
            unit = torch.cat([unit, (self.F + torch.arange(self.K)).repeat_interleave(self.L)])
        self.unit = unit.cuda()
        dev = 'cuda'
        rows = lambda fl: torch.cat([torch.arange(f * self.T, (f + 1) * self.T) for f in fl]).to(dev)
        text = lambda j: torch.arange(self.F * self.T + j * self.L, self.F * self.T + (j + 1) * self.L, device=dev)
        self.segs = []                                     # (query rows as (lo, hi), key row index)
        for j in range(self.K):
            lo, hi = self.frames[j][0] * self.T, (self.frames[j][-1] + 1) * self.T
            self.segs.append(((lo, hi), torch.cat([rows(self.keys[j])] + ([text(j)] if self.L else []))))
            if self.L:
                t0 = self.F * self.T + j * self.L
                self.segs.append(((t0, t0 + self.L), torch.cat([rows(self.frames[j]), text(j)])))
        self.useful = sum(4.0 * (hi - lo) * len(k) * self.H * D for (lo, hi), k in self.segs)
        self.dense = 4.0 * self.S * self.S * self.H * D

    def mask_rows(self, r):
        return self.allow[self.unit[r]][:, self.unit]


def run_layout(sc, name, n):
    torch.manual_seed(0)
    S, H = sc.S, sc.H
    q, k, v = (torch.randn(S, H, D, device='cuda', dtype=torch.bfloat16) for _ in range(3))
    sample = torch.randperm(S, device='cuda')[:256]
    s_ = torch.einsum('qhd,khd->hqk', q[sample].float(), k.float()) / math.sqrt(D)
    ref = torch.einsum('hqk,khd->qhd', s_.masked_fill(~sc.mask_rows(sample)[None], float('-inf')).softmax(-1), v.float())
    ref_dense = torch.einsum('hqk,khd->qhd', s_.softmax(-1), v.float())
    del s_
    err = lambda o, r=ref: float((o[sample].float() - r).norm() / r.norm())
    res = dict(layout=name, tokens=S, scenes=sc.K, frames=sc.F, density=sc.useful / sc.dense, methods={})
    out = res['methods']

    def record(m, parts, o, reference=ref):
        tot = sum(parts.values())
        out[m] = dict(ms=tot, **{f'{p}_ms': t for p, t in parts.items()}, err=err(o, reference))
        print(f'[kernel] {name} {m:10s} {tot:9.2f} ms  ' + ' '.join(f'{p} {t:.2f}' for p, t in parts.items()) +
              f'  err {out[m]["err"]:.2e}', flush=True)

    o = first(fa3.flash_attn_func(q[None], k[None], v[None]))[0]
    record('dense', dict(kernel=timed(lambda: fa3.flash_attn_func(q[None], k[None], v[None]), n)), o, ref_dense)
    del o
    # OneReel packed: prep (query cut + K/V copy), one varlen call, write-back
    lq = [hi - lo for (lo, hi), _ in sc.segs]
    lk = [len(kk) for _, kk in sc.segs]
    cu = lambda xs: torch.tensor([0] + np.cumsum(xs).tolist(), dtype=torch.int32, device='cuda')
    cq, ck = cu(lq), cu(lk)
    kidx = torch.cat([kk for _, kk in sc.segs])

    def prep():
        return torch.cat([q[lo:hi] for (lo, hi), _ in sc.segs]), k.index_select(0, kidx), v.index_select(0, kidx)
    qp, kp, vp = prep()

    def kern():
        return first(fa3.flash_attn_varlen_func(qp, kp, vp, cq, ck, max(lq), max(lk)))
    op = kern()
    full = torch.empty_like(q)

    def post():
        for ((lo, hi), _), oj in zip(sc.segs, op.split(lq)):
            full[lo:hi] = oj
    post()
    record('packed', dict(prep=timed(prep, n), kernel=timed(kern, n), post=timed(post, n)), full)
    # one FA3 call per segment on the same copied K/V
    ks, vs = kp.split(lk), vp.split(lk)

    def loop():
        return [first(fa3.flash_attn_func(q[None, lo:hi], kj[None], vj[None]))[0]
                for ((lo, hi), _), kj, vj in zip(sc.segs, ks, vs)]
    full2 = torch.empty_like(q)
    for ((lo, hi), _), oj in zip(sc.segs, loop()):
        full2[lo:hi] = oj
    record('per_scene', dict(prep=out['packed']['prep_ms'], kernel=timed(loop, n)), full2)
    del qp, kp, vp, op, ks, vs, full, full2
    torch.cuda.empty_cache()
    if MODELS_SCOPE[name.split('/')[0]] == 'halo':          # contiguous key ranges: read in place
        T = sc.T
        kr = [(sc.keys[j][0] * T, (sc.keys[j][-1] + 1) * T) for j in range(sc.K)]
        assert all(len(sc.keys[j]) == sc.keys[j][-1] - sc.keys[j][0] + 1 for j in range(sc.K))
        off = torch.tensor([a for a, _ in kr] + [S], dtype=torch.int32, device='cuda')
        used = torch.tensor([b - a for a, b in kr], dtype=torch.int32, device='cuda')
        cq0 = cu([(fr[-1] + 1 - fr[0]) * T for fr in sc.frames])
        zc = lambda: first(fa3.flash_attn_varlen_func(q, k, v, cq0, off, max(lq), max(used.tolist()), seqused_k=used))
        record('zero_copy', dict(kernel=timed(zc, n)), zc())
    try:
        flex_run(sc, q, k, v, n, record, out)
    except Exception as e:
        print(f'[kernel] {name} flex failed: {e!r}', flush=True)
        out['flex'] = dict(error=repr(e))
    torch.cuda.empty_cache()
    try:
        fi_bsr(sc, q, k, v, n, record, out)
    except Exception as e:                                   # keep the other results
        print(f'[kernel] {name} fi_bsr failed: {e!r}', flush=True)
        out['fi_bsr'] = dict(error=repr(e))
    dense_tps = sc.dense / out['dense']['ms']
    for m, x in out.items():
        if 'ms' in x:
            x['useful_tflops'] = sc.useful / x['ms'] / 1e9
            x['eff'] = (sc.useful / x['ms']) / dense_tps
    del q, k, v
    torch.cuda.empty_cache()
    return res


def flex_run(sc, q, k, v, n, record, out):
    """FlexAttention in the original order; block mask built once per request (mask_ms)"""
    S = sc.S
    last = S - 1
    allow, unit = sc.allow, sc.unit
    mm = lambda b, h, qi, ki: allow[unit[qi.clamp(max=last)], unit[ki.clamp(max=last)]] & (qi <= last) & (ki <= last)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    bm = create_block_mask(mm, None, None, S, S, device='cuda', _compile=True)
    t1.record()
    torch.cuda.synchronize()
    qt, kt, vt = (x[None].transpose(1, 2) for x in (q, k, v))
    fl = lambda: flex(qt, kt, vt, block_mask=bm)[0].transpose(0, 1)
    record('flex', dict(kernel=timed(fl, n)), fl())
    out['flex']['mask_ms'] = t0.elapsed_time(t1)
    del bm, qt, kt, vt


def fi_bsr(sc, q, k, v, n, record, out):
    """full blocks on FlashInfer's fa3 backend, partial blocks masked (fa2), merged by LSE"""
    import flashinfer
    from flashinfer.sparse import BlockSparseAttentionWrapper
    import flashinfer.sparse

    def convert(mask, indptr):                               # int64 offsets (32-bit overflow in 0.3.1)
        nnz, R, C = mask.shape
        ip = indptr.long().tolist()
        o = torch.empty(nnz * R * C, dtype=mask.dtype, device=mask.device)
        for i in range(len(ip) - 1):
            o[ip[i] * R * C:ip[i + 1] * R * C] = mask[ip[i]:ip[i + 1]].transpose(0, 1).reshape(-1)
        return o
    flashinfer.sparse.convert_bsr_mask_layout = convert
    S, H, B = sc.S, sc.H, BLOCK
    nb = math.ceil(S / B)
    Sp = nb * B
    pad = lambda x: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, Sp - S))
    qp, kp, vp = pad(q), pad(k), pad(v)
    unit_p = torch.cat([sc.unit, sc.unit.new_full((Sp - S,), -1)])
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    per_row = []
    ublk = unit_p.view(nb, B)
    for rb in range(nb):
        uq = ublk[rb]
        valid_q = uq >= 0
        m = torch.zeros(B, Sp, dtype=torch.bool, device='cuda')
        m[valid_q, :S] = sc.allow[uq[valid_q]][:, sc.unit]
        blk = m.view(B, nb, B)
        cnt = blk.sum((0, 2))
        cols = (cnt > 0).nonzero().flatten()
        full = cnt[cols] == B * B
        per_row.append((cols, full, blk[:, cols[~full]].permute(1, 0, 2).contiguous()))   # masks of partial blocks

    def wrappers(select, with_mask, backend):
        groups, cur, bits = [], [], 0
        for rb in range(nb):
            nbits = int(select(per_row[rb][1]).sum()) * B * B if with_mask else 0
            if cur and bits + nbits > 2 ** 30:
                groups.append(cur)
                cur, bits = [], 0
            cur.append(rb)
            bits += nbits
        groups.append(cur)
        ws = []
        for g in groups:
            ind, msk, indptr = [], [], [0]
            for rb in g:
                cols, full, mk = per_row[rb]
                sel = select(full)
                ind.append(cols[sel].int())
                if with_mask:
                    msk.append(mk)
                indptr.append(indptr[-1] + int(sel.sum()))
            if indptr[-1] == 0:
                continue
            w = BlockSparseAttentionWrapper(torch.empty(256 * 2 ** 20, dtype=torch.uint8, device='cuda'), backend=backend)
            need = indptr[-1] * B + 1                      # fa3 expands blocks to token indices (128M fixed)
            if backend == 'fa3' and need > w._vector_sparse_indices_buffer.numel():
                w._vector_sparse_indices_buffer = torch.empty(need, dtype=torch.int32, device='cuda')
            w.plan(torch.tensor(indptr, dtype=torch.int32, device='cuda'), torch.cat(ind), len(g) * B, Sp, B, B, H, H, D,
                   mask=torch.cat(msk) if with_mask else None, q_data_type=torch.bfloat16)
            ws.append((g[0] * B, (g[-1] + 1) * B, w))
        return ws
    full_w = wrappers(lambda f: f, False, 'fa3')
    part_w = wrappers(lambda f: ~f, True, 'fa2')
    t1.record()
    torch.cuda.synchronize()

    def part(ws):
        """one wrapper set over all block rows -> (out [Sp, H, D], lse [Sp, H]); rows without blocks stay empty"""
        o = qp.new_zeros(Sp, H, D)
        lse = torch.full((Sp, H), float('-inf'), device='cuda')
        for lo, hi, w in ws:
            o[lo:hi], lse[lo:hi] = w.run(qp[lo:hi], kp, vp, return_lse=True)
        return o, lse

    def run():
        of, lf = part(full_w)
        op_, lp = part(part_w)
        return flashinfer.merge_state(of, lf, op_, lp)[0][:S]
    record('fi_bsr', dict(kernel=timed(run, n)), run())
    out['fi_bsr']['plan_ms'] = t0.elapsed_time(t1)
    out['fi_bsr']['blocks'] = dict(full=sum(int(f.sum()) for _, f, _ in per_row),
                                   partial=sum(int((~f).sum()) for _, f, _ in per_row))


MODELS_SCOPE = {m: c['scope'] for m, c in MODELS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--models', default='hunyuan,wan')
    ap.add_argument('--scenes', default='3,6,12,24')
    ap.add_argument('--no-uneven', action='store_true')
    ap.add_argument('--iters', type=int, default=10)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    for model in a.models.split(','):
        layouts = [(f'{model}/K{K}', [F_SCENE] * K) for K in map(int, a.scenes.split(','))]
        if not a.no_uneven:
            layouts.append((f'{model}/uneven', UNEVEN))
        for name, lengths in layouts:
            r = run_layout(Scope(model, lengths), name, a.iters)
            r['model'] = model
            res.append(r)
            (a.out / 'kernel_breakdown.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
