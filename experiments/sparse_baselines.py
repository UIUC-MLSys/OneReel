"""Training-free sparse-attention baselines on Hunyuan / Wan with and without their dense warm-up steps.

Reimplementations in this repo's attention hooks (St.override), not the official kernels (the official repos target
HunyuanVideo / Wan2.1). All sparse calls run flex attention on 128-token blocks; dense steps run the normal FA2 path.
  svg        Sparse VideoGen: per (step, layer, CFG branch, head) choose a spatial or a temporal band by the MSE of 64
             sampled query rows against full attention (profiling bands of 1.5 / 2 frames, as the official hyvideo /
             wan get_attention_mask); the runtime band keeps SVG_SPARSITY of the attention (official sparsity_to_width:
             width from the kept fraction, the text rows and columns counted as full), the temporal band in
             position-major token order; Wan adds the official first-frame sink (keys of the first T tokens). Official
             720p scripts: kept 0.25 / 0.3, warm-up 10% / 20% of steps, 3% of layers (hyvideo / wan)
  topp       dynamic block top-p (SVG2 / XAttention style without k-means): block-mean q.k proxy, per query block the
             smallest key-block set holding P_TOP of the proxy mass, rebuilt every step (warm-up as SVG2: 10% / 20%)
  radial     Radial Attention style static mask: frame distance d keeps a spatial window of T >> floor(log2 d)
             tokens (full for d <= 1), plus frame 0 (official: HunyuanVideo dense for 12 of 50 steps = 24%; Wan2.2
             11 of 40 = 27.5% and layer 0)
  sol        Sol-Attn (arXiv 2607.24027): 64x64 blocks, proxy s = mean(Q_i) . mean(K_j), key block kept if
             s > mu_i + beta sigma_i of its proxy row; skipped blocks are not dropped but approximated in the softmax by
             exp(q . mean(K_j)) times the block's count / value sum (zeroth-order Taylor); beta is calibrated once per
             model on SOL_CALIB for SOL_DENSITY; warm-up 20% of steps and the first layer (paper Sec. 4.1). Routing is
             exact at 64 blocks, but flex in torch 2.5 has no 64-blocks, so kept 64-blocks are a mask inside 128-blocks
             and the correction is separate matmuls: its timing is not representative
  solk       the official Sol-Attn kernel (NVIDIA sol-engine, third_party/Sana, CuTe DSL, SM90/SM100), called as in its
             HunyuanVideo integration (config/hunyuan_video/sol_only.toml): tau 1.0, thresh_type diag, kv_splits auto,
             no dense steps or layers; video queries run the kernel with the valid text tokens as an exact KV sink,
             text queries run dense SDPA; on Wan (self-attention has no text) the kernel runs on the video tokens,
             with the released Wan configuration's dense guards (first 20% of the steps and layer 0).
             Needs torch >= 2.8 and nvidia-cutlass-dsl (onereel env)
  <m>_nowarm the same method sparse from the first step (dense layers unchanged)
  flexdense  every block kept through the same flex path (numerical floor of the comparison)
Text tokens (Hunyuan) are always attended and attend to everything. Videos: OUT/<model>/<method>/<id>/video.mp4 with
meta.json (per-step times, warm-up steps, kept block fraction per step).

Usage (onereel_metrics env; solk: onereel env): python experiments/sparse_baselines.py --model hunyuan --methods svg,svg_nowarm
    --out DIR [--ids ...]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.nn.attention.flex_attention import BlockMask

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_order_attention as E  # noqa: E402

B, P_TOP, SAMPLED_ROWS, TEXT_BLOCKS, STEPS = 128, 0.9, 64, 2, 20
SOL_DENSITY, SOL_CALIB, SB = 0.15, 'W2_ABC', 64
SOL_BETA = {}
KEEP64 = {}
SOLK = dict(tau=1.0, thresh_type='diag', kv_splits='auto')
SOLK_RUN = None                # Sana's _run_sol_attn_bthd, imported in main() when a solk method is requested
# MiniMax-H3 ('h3', 50 steps): Sol-Attn's released H3 configuration (10 dense steps, 2 dense layers, text and audio as an
# exact prefix sink with dense prefix queries); SVG and Radial have no H3 release, so they keep their HunyuanVideo shares
# CogVideoX1.5 ('cogvideox', 42 layers, frames = token frames of 2 latent frames): SVG's released CogVideoX-v1.5 settings
# (cog_inference.py: first 20% of steps and layer 0 dense, sparsity 0.25, temporal mask multiplier 2); Radial and Sol-Attn
# have no CogVideoX release, so they keep their Wan2.x warm-ups
WARM = {'solk': {'hunyuan': 0.0, 'wan': 0.20, 'h3': 0.20, 'cogvideox': 0.20}, 'sol': {'hunyuan': 0.20, 'wan': 0.20}, 'svg': {'hunyuan': 0.10, 'wan': 0.20, 'h3': 0.10, 'cogvideox': 0.20}, 'topp': {'hunyuan': 0.10, 'wan': 0.20}, 'radial': {'hunyuan': 0.24, 'wan': 0.275, 'h3': 0.24, 'cogvideox': 0.275}}
SVG_SPARSITY = {'hunyuan': 0.25, 'wan': 0.3, 'h3': 0.25, 'cogvideox': 0.25}       # fraction of attention kept (official scripts' --sparsity)
SVG_PROFILE_FRAMES = {'hunyuan': 1.5, 'wan': 2.0, 'h3': 1.5, 'cogvideox': 2.0}   # band of the head-classification masks (get_attention_mask)
DENSE_LAYERS = {'svg': {'hunyuan': 0.03, 'wan': 0.03, 'h3': 0.03, 'cogvideox': 0.025}, 'topp': {'hunyuan': 0.03, 'wan': 0.03}}
RADIAL_DENSE_LAYERS = {'hunyuan': 0, 'wan': 1, 'h3': 0, 'cogvideox': 1}
# official warm-ups (steps as a fraction, dense layers): Radial scripts hunyuan_t2v 12/50, 0 layers, wan_22_t2v 11/40, 1 layer;
# Sol-Attn released configs hunyuan_video/sol_only (no dense steps or layers), wan21_t2v_*/sol_only (10/50 steps, layer 0)
SOLK_DENSE_LAYERS = {'hunyuan': 0, 'wan': 1, 'h3': 2, 'cogvideox': 1}
# the released Sol-Attn kernel takes head dim 128 only (Sana sol_attn_supported); on other shapes Sana's adapter silently
# runs dense SDPA, so such models are refused here. CogVideoX1.5 has head dim 64.
SOLK_HEAD_DIM = {'hunyuan': 128, 'wan': 128, 'h3': 128, 'cogvideox': 64}
VALID = {}                     # Sp -> key validity buffer, updated in place so the compiled flex graph is reused


def valid_mod(b, h, q, k):
    return VALID['buf'][k]


def sol_mod(b, h, q, k):
    return VALID['buf'][k] & KEEP64['buf'][h, q // SB, k // SB]


IDS = [f'{g}_{o}' for g in ('W1', 'A4', 'L1', 'O3') for o in ('ABC', 'BCA', 'CAB')]


class Sparse:
    def __init__(self, model, method, n_layers, steps=STEPS):
        self.model, self.method = model, method
        base = method.replace('_nowarm', '')
        self.base = base
        self.warm = 0 if method.endswith('_nowarm') or base == 'flexdense' else round(WARM[base][model] * steps)
        if base == 'radial':
            self.dense_layers = RADIAL_DENSE_LAYERS[model]
        elif base == 'solk':
            if SOLK_HEAD_DIM[model] != 128:
                raise ValueError(f"Sol-Attn kernel needs head dim 128; {model} has {SOLK_HEAD_DIM[model]}")
            self.dense_layers = SOLK_DENSE_LAYERS[model]
        elif base == 'flexdense':
            self.dense_layers = 0
        elif base == 'sol':
            self.dense_layers = 1
        else:
            self.dense_layers = math.floor(DENSE_LAYERS[base][model] * n_layers)
        self.layouts, self.layout, self.static = {}, None, {}
        self.kept = {}
        self.calib = []

    # ---------------------------------------------------------------- layout and block masks
    def setup(self, Sv, n_text):
        T, F = E.St.T, E.St.F
        nbv = math.ceil(Sv / B)
        ntb = max(TEXT_BLOCKS, math.ceil(n_text / B)) if n_text else 0          # long multi-shot prompts need more
        assert n_text <= ntb * B, f'{n_text} text tokens exceed {ntb} blocks'
        Svp = nbv * B
        Sp = Svp + ntb * B
        valid = torch.zeros(Sp, dtype=torch.bool, device='cuda')
        valid[:Sv] = True
        valid[Svp:Svp + n_text] = True
        partial = torch.zeros(nbv + ntb, dtype=torch.bool, device='cuda')
        partial[nbv - 1] = Sv % B != 0
        if ntb:
            partial[-1] = True
        pos = torch.arange(Sv, device='cuda')
        perm = (pos % T) * F + pos // T                               # frame-major -> position-major
        self.layout = dict(Sv=Sv, n=n_text, nbv=nbv, ntb=ntb, Svp=Svp, Sp=Sp, valid=valid, partial=partial,
                           perm=torch.empty_like(perm).scatter_(0, perm, pos), inv=perm)

    def block_mask(self, keep_v, H):
        """keep_v [H, nbv, nbv] video-video blocks -> BlockMask over [video | text] (text rows and columns full)"""
        L = self.layout
        nbv, ntb = L['nbv'], L['ntb']
        NB = nbv + ntb
        keep = torch.ones(H, NB, NB, dtype=torch.bool, device='cuda')
        keep[:, :nbv, :nbv] = keep_v
        part = keep & L['partial'][None, None]
        full = keep & ~L['partial'][None, None]
        idx = lambda m: m.to(torch.int8).argsort(dim=-1, descending=True, stable=True).to(torch.int32)[None]
        return BlockMask.from_kv_blocks(part.sum(-1, dtype=torch.int32)[None], idx(part), full.sum(-1, dtype=torch.int32)[None],
                                        idx(full), BLOCK_SIZE=B, mask_mod=valid_mod)

    def svg_width(self):
        """official sparsity_to_width in tokens, rounded to 128 as the official mask_mods (hyvideo floor, wan ceil)"""
        L = self.layout
        ctx = L['n']                                              # text tokens (Hunyuan); 0 on Wan (official too)
        seq = ctx + L['Sv']
        s = (SVG_SPARSITY[self.model] * seq ** 2 - 2 * seq * ctx) / seq ** 2
        w = seq * (1 - math.sqrt(1 - s))
        return (math.ceil if self.model == 'wan' else math.floor)(w / B) * B

    def band(self, H):
        """runtime mask at block level: hyvideo |i - j| < W / B, wan |i - j| <= W / B plus the first-frame sink"""
        if 'band' not in self.static:
            nbv, wb = self.layout['nbv'], self.svg_width() // B
            i = torch.arange(nbv, device='cuda')
            d = (i[:, None] - i[None]).abs()
            keep = d <= wb if self.model == 'wan' else d < wb
            if self.model == 'wan':
                keep = keep | (i[None] < math.ceil(E.St.T / B))
            keep = keep[None].expand(H, -1, -1)
            self.static['band'] = (self.block_mask(keep, H), keep.float().mean().item())
        return self.static['band']

    def radial(self, H):
        if 'radial' not in self.static:
            T, nbv = E.St.T, self.layout['nbv']
            c = torch.arange(nbv, device='cuda') * B + B // 2
            f, s = c // T, c % T
            d = (f[:, None] - f[None]).abs()
            k = torch.where(d <= 1, torch.zeros_like(d), torch.floor(torch.log2(d.clamp_min(1).float())).long())
            width = T / (2.0 ** k)
            keep = ((s[:, None] - s[None]).abs().float() <= width / 2 + B) | (f[None] == 0) | (d <= 1)
            keep = keep[None].expand(H, -1, -1)
            self.static['radial'] = (self.block_mask(keep, H), keep.float().mean().item())
        return self.static['radial']

    def topp(self, qv, kv):
        L, H = self.layout, qv.shape[1]
        pad = lambda x: Fn.pad(x, (0, 0, 0, 0, 0, L['Svp'] - L['Sv'])).view(L['nbv'], B, H, -1)
        cnt = torch.full((L['nbv'],), B, device='cuda')
        cnt[-1] = L['Sv'] - (L['nbv'] - 1) * B
        qb = pad(qv).float().sum(1) / cnt[:, None, None]
        kb = pad(kv).float().sum(1) / cnt[:, None, None]
        p = (torch.einsum('qhd,khd->hqk', qb, kb) / math.sqrt(qv.shape[-1]) + cnt.log()[None, None]).softmax(-1)
        v, i = p.sort(-1, descending=True)
        keep_sorted = (v.cumsum(-1) - v) < P_TOP
        keep = torch.zeros_like(keep_sorted).scatter_(-1, i, keep_sorted)
        keep |= torch.eye(L['nbv'], dtype=torch.bool, device='cuda')[None]
        return self.block_mask(keep, H), keep.float().mean().item()

    # ---------------------------------------------------------------- Sol-Attn
    def sol(self, q, k, v, eq, ek, ev):
        L, H, D = self.layout, q.shape[1], q.shape[2]
        Sv, Svp, Sp, n = L['Sv'], L['Svp'], L['Sp'], L['n']
        nbv, NB = Svp // SB, Sp // SB
        pad = lambda x: Fn.pad(x, (0, 0, 0, 0, 0, Svp - Sv)).view(nbv, SB, H, D).float()
        cnt = L['valid'][:Svp].view(nbv, SB).sum(1).float()
        live = cnt > 0
        qb, kb = pad(q).sum(1) / cnt.clamp_min(1)[:, None, None], pad(k).sum(1) / cnt.clamp_min(1)[:, None, None]
        vsum = pad(v).sum(1)
        sc = torch.einsum('qhd,khd->hqk', qb, kb) / math.sqrt(D)
        m = live[None, None].float()
        mu = (sc * m).sum(-1, keepdim=True) / live.sum()
        sd = (((sc - mu) ** 2) * m).sum(-1, keepdim=True).div(live.sum()).sqrt().clamp_min(1e-6)
        z = (sc - mu) / sd
        beta = SOL_BETA.get(self.model)
        if beta is None:
            zz = z[:, live][:, :, live].flatten()
            zz = zz[torch.randint(0, zz.numel(), (min(zz.numel(), 2 ** 22),), device=zz.device)]
            beta = zz.float().quantile(1 - SOL_DENSITY).item()
            self.calib.append(beta)
        keep = (z > beta) & live[None, None]
        k64 = torch.ones(H, NB, NB, dtype=torch.bool, device='cuda')
        k64[:, :nbv, :nbv] = keep
        if KEEP64.get('buf') is None or KEEP64['buf'].shape != k64.shape:
            KEEP64['buf'] = k64.clone()
        KEEP64['buf'].copy_(k64)
        k128 = k64.view(H, NB // 2, 2, NB // 2, 2).any(4).any(2)
        idx = k128.to(torch.int8).argsort(dim=-1, descending=True, stable=True).to(torch.int32)[None]
        bm = BlockMask.from_kv_blocks(k128.sum(-1, dtype=torch.int32)[None], idx, BLOCK_SIZE=B, mask_mod=sol_mod)

        def lay(x, xt):
            out = x.new_zeros(Sp, H, D)
            out[:Sv] = x
            if n:
                out[Svp:Svp + n] = xt[:n]
            return out.transpose(0, 1)[None].to(torch.bfloat16)
        o, lse = E.flex(lay(q, eq), lay(k, ek), lay(v, ev), block_mask=bm, return_lse=True)
        o, lse = torch.nan_to_num(o[0].transpose(0, 1).float()), lse[0].float()
        ov = torch.empty(Sv, H, D, device='cuda')
        blk = torch.arange(Sv, device='cuda') // SB
        for a in range(0, Sv, 8192):
            t = slice(a, min(Sv, a + 8192))
            s_ = torch.einsum('thd,khd->htk', q[t].float(), kb) / math.sqrt(D)
            skip = ~keep[:, blk[t]] & live[None, None]
            s_ = s_.masked_fill(~skip, float('-inf')) + cnt.clamp_min(1e-9).log()[None, None]
            l0 = lse[:, t]
            mx = torch.maximum(l0, s_.amax(-1))
            w = (s_ - mx[..., None]).exp()                                  # cnt_j exp(q . mean K_j) / e^mx
            den = (l0 - mx).exp() + w.sum(-1)
            num = o[t].transpose(0, 1) * (l0 - mx).exp()[..., None] + torch.einsum('htk,khd->htd', w / cnt.clamp_min(1)[None, None], vsum)
            ov[t] = (num / den[..., None]).transpose(0, 1)
        ot = None
        if n:
            ot = o.new_zeros(eq.shape[0], H, D)
            ot[:n] = o[Svp:Svp + n]
        return ov, ot, keep.float().sum().item() / (H * live.sum().item() ** 2)

    # ---------------------------------------------------------------- SVG head selection
    def svg_pick(self, q, k, v):
        """q, k, v [S, H, D] in natural order ([video | valid text]); returns [H] bool, True = temporal"""
        L = self.layout
        Sv, H, D = L['Sv'], q.shape[1], q.shape[2]
        rows = torch.randint(0, Sv, (SAMPLED_ROWS,), device='cuda')
        w = math.floor(SVG_PROFILE_FRAMES[self.model] * E.St.T) // B
        kidx = torch.arange(q.shape[0], device='cuda')
        is_text = kidx >= Sv
        blk_nat = torch.where(is_text, -10 ** 6, kidx // B)
        blk_tmp = torch.where(is_text, -10 ** 6, L['inv'][kidx.clamp_max(Sv - 1)] // B)
        logits = torch.einsum('nhd,khd->hnk', q[rows].float(), k.float()) / math.sqrt(D)
        gold = logits.softmax(-1) @ v.float().transpose(0, 1)
        mses = []
        sink_nat = ~is_text & (kidx < E.St.T)                                    # wan profiling masks keep frame 0
        sink_tmp = ~is_text & (L['inv'][kidx.clamp_max(Sv - 1)] < E.St.T)
        for blk, sink in ((blk_nat, sink_nat), (blk_tmp, sink_tmp)):
            m = ((blk[rows][:, None] - blk[None]).abs() < w) | is_text[None]
            if self.model == 'wan':
                m = m | sink[None]
            o = logits.masked_fill(~m[None], float('-inf')).softmax(-1) @ v.float().transpose(0, 1)
            mses.append(((o - gold) ** 2).mean((1, 2)))
        return mses[1] < mses[0]

    # ---------------------------------------------------------------- hook
    def solk(self, q, k, v, eq, ek, ev, n):
        """official kernel on [video, valid text] with the text as exact KV sink; dense text queries"""
        Sv = q.shape[0]
        qa, ka, va = (torch.cat([a, b[:n]])[None].to(torch.bfloat16).contiguous() for a, b in ((q, eq), (k, ek), (v, ev)))
        o = SOLK_RUN(qa, ka, va, sink_start=Sv, sink_tokens=n, **SOLK)[0]
        ot = o.new_zeros(eq.shape[0], *o.shape[1:])
        if n:
            ot[:n] = Fn.scaled_dot_product_attention(qa[:, Sv:].transpose(1, 2), ka.transpose(1, 2),
                                                     va.transpose(1, 2))[0].transpose(0, 1)
        return o[:Sv].to(q.dtype), ot.to(q.dtype)

    def __call__(self, kind, layer, q, k, v, eq=None, ek=None, ev=None, amask=None):
        step = E.St.step()
        if step < self.warm or layer < self.dense_layers:
            return None
        Sv, H = q.shape[0], q.shape[1]
        n = int(amask.sum()) if amask is not None else 0
        if self.base == 'solk':
            if kind == 'wan':
                qa, ka, va = (x[None].to(torch.bfloat16).contiguous() for x in (q, k, v))
                return SOLK_RUN(qa, ka, va, **SOLK)[0].to(q.dtype).flatten(1, 2)
            ov, ot = self.solk(q, k, v, eq, ek, ev, n)
            return ov.flatten(1, 2), ot.flatten(1, 2)
        if (Sv, n) not in self.layouts:
            self.setup(Sv, n)
            self.layouts[(Sv, n)] = (self.layout, {})
        self.layout, self.static = self.layouts[(Sv, n)]
        L = self.layout
        if VALID.get('buf') is None or VALID['buf'].shape[0] != L['Sp']:
            VALID['buf'] = L['valid'].clone()
        VALID['buf'].copy_(L['valid'])
        if n:
            qa, ka, va = (torch.cat([a, b[:n]]) for a, b in ((q, eq), (k, ek), (v, ev)))
        else:
            qa, ka, va = q, k, v
        temporal = None
        if self.base == 'sol':
            ov, ot, dens = self.sol(q, k, v, eq, ek, ev)
            self.kept.setdefault(step, []).append(dens)
            if kind == 'wan':
                return ov.flatten(1, 2)
            return ov.flatten(1, 2), ot.flatten(1, 2)
        if self.base == 'svg':
            temporal = self.svg_pick(qa, ka, va)
            bm, dens = self.band(H)
        elif self.base == 'topp':
            bm, dens = self.topp(q, k)
        elif self.base == 'radial':
            bm, dens = self.radial(H)
        else:
            bm, dens = self.block_mask(torch.ones(H, L['nbv'], L['nbv'], dtype=torch.bool, device='cuda'), H), 1.0
        self.kept.setdefault(step, []).append(dens)

        def lay(x, xt):
            vid = x
            if temporal is not None:
                vid = torch.where(temporal[None, :, None], x[L['perm']], x)
            out = x.new_zeros(L['Sp'], H, x.shape[-1])
            out[:Sv] = vid
            if n:
                out[L['Svp']:L['Svp'] + n] = xt[:n]
            return out.transpose(0, 1)[None].to(torch.bfloat16)
        o = E.flex(lay(q, eq), lay(k, ek), lay(v, ev), block_mask=bm)[0].transpose(0, 1)
        ov = o[:Sv]
        if temporal is not None:
            ov = torch.where(temporal[None, :, None], ov[L['inv']], ov)
        ov = ov.flatten(1, 2)
        if kind == 'wan':
            return ov
        ot = o.new_zeros(eq.shape[0], H, o.shape[-1])
        ot[:n] = o[L['Svp']:L['Svp'] + n]
        return ov, ot.flatten(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['hunyuan', 'wan'], required=True)
    ap.add_argument('--methods', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ids', default=','.join(IDS))
    ap.add_argument('--sol-beta', type=float, default=None, help='skip the calibration run')
    ap.add_argument('--cfg', type=float, default=None, help='guidance scale (default: Hunyuan 6.0, Wan 1.0 = no CFG)')
    ap.add_argument('--bench', type=Path, default=E.BENCH, help='prompt set; rows with a "frames" field use that length (default 81)')
    a = ap.parse_args()
    bench = {r['id']: r for r in json.loads(a.bench.read_text())['prompts']}
    E.St.sdpa_dense, E.St.compile = False, False
    if a.model == 'hunyuan':
        from diffusers import HunyuanVideo15Pipeline
        pipe = HunyuanVideo15Pipeline.from_pretrained(E.local(E.HUNYUAN['id']), torch_dtype=torch.bfloat16).to('cuda')
        pipe.vae.enable_tiling()
        E.chunked_vae_decode(pipe)
        cfg = a.cfg or 6.0
        if cfg > 1:
            pipe.guider.guidance_scale = cfg
        else:
            pipe.guider.disable()
        E.St.cfg, E.St.impl = cfg > 1, 'varlen'
        E.hunyuan_setup(pipe)
        prep, gen, fps = E.hunyuan_prepare, E.hunyuan_generate, E.HUNYUAN['fps']
    else:
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(E.local(E.WAN['id']), torch_dtype=torch.bfloat16).to('cuda')
        cfg = a.cfg or 1.0
        E.St.cfg, E.St.impl = cfg > 1, 'fast'                          # St.cfg: two forwards per step
        E.wan_setup(pipe)
        prep, gen, fps = E.wan_prepare, E.wan_generate, E.WAN['fps']
    if any(m.startswith('solk') for m in a.methods.split(',')):
        import os
        global SOLK_RUN
        os.environ['SOL_ATTN_STRICT'] = '1'                             # a kernel failure raises instead of going dense
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/Sana'))
        from techniques.sparse_backends.sol_attn_backend import _run_sol_attn_bthd as SOLK_RUN
    tr = pipe.transformer
    tr.register_forward_pre_hook(lambda m, args: setattr(E.St, 'fwd', E.St.fwd + 1))
    fwd = []

    def t_pre(m, *_):
        torch.cuda.synchronize()
        m._t0 = time.perf_counter()

    def t_post(m, *_):
        torch.cuda.synchronize()
        fwd.append((E.St.step(), time.perf_counter() - m._t0))
    tr.register_forward_pre_hook(t_pre)
    tr.register_forward_hook(t_post)
    if any(m.startswith('sol') for m in a.methods.split(',')):
        bf = a.out / f'sol_beta_{a.model}.json'
        if a.sol_beta is not None:
            SOL_BETA[a.model] = a.sol_beta
        elif bf.exists():
            SOL_BETA[a.model] = json.loads(bf.read_text())['beta']
        else:
            r = {x['id']: x for x in json.loads(E.BENCH.read_text())['prompts']}[SOL_CALIB]
            sp = Sparse(a.model, 'sol', E.St.n_layers)
            E.St.override = sp
            E.St.setting, E.St.assign, E.St.nseg = 'baseline', False, len(r['sentences'])
            E.St.calls, E.St.x0_prev, E.St.x0_log, E.St.dyn_cache = [], None, [], {}
            E.St.scene_prev, E.St.scene_log, E.St.latent_path, E.St.edges = None, [], None, set()
            prep(pipe, r['sentences'], 81)
            E.St.fwd = 0
            gen(pipe, r['prompt'], 81, STEPS, 0, cfg)
            E.St.override = None
            SOL_BETA[a.model] = float(np.mean(sp.calib))
            a.out.mkdir(parents=True, exist_ok=True)
            bf.write_text(json.dumps({'beta': SOL_BETA[a.model], 'density': SOL_DENSITY, 'calib_id': SOL_CALIB, 'calls': len(sp.calib)}))
        print(f'[sol] {a.model} beta {SOL_BETA[a.model]:.3f}', flush=True)
    for method in a.methods.split(','):
        for pid in a.ids.split(','):
            d = a.out / a.model / method / pid
            if (d / 'video.mp4').exists():
                continue
            r = bench[pid]
            sp = None if method == 'dense' else Sparse(a.model, method, E.St.n_layers)
            E.St.override = sp
            E.St.setting, E.St.assign, E.St.nseg = 'baseline', False, len(r['sentences'])
            E.St.calls, E.St.x0_prev, E.St.x0_log, E.St.dyn_cache = [], None, [], {}
            E.St.scene_prev, E.St.scene_log, E.St.latent_path, E.St.edges = None, [], None, set()
            n_frames = r.get('frames', 81)
            prep(pipe, r['sentences'], n_frames)
            E.St.fwd = 0
            fwd.clear()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            video = gen(pipe, r['prompt'], n_frames, STEPS, 0, cfg)
            torch.cuda.synchronize()
            total = time.perf_counter() - t0
            E.St.override = None
            per_step = [0.0] * STEPS
            for s, t in fwd:
                per_step[s] += t
            warm = sp.warm if sp else 0
            d.mkdir(parents=True, exist_ok=True)
            E.export_to_video(list(video), str(d / 'video.mp4'), fps=fps)
            meta = {'model': a.model, 'setting': method, 'id': pid, 'seed': 0, 'cfg': cfg, 'frames': len(video), 'steps': STEPS, 'seconds': total,
                    'denoise_s': sum(per_step), 'step_s': per_step, 'warm_steps': warm,
                    'warm_s': sum(per_step[:warm]), 'dense_layers': sp.dense_layers if sp else None,
                    'kept_fraction': {s: float(np.mean(v)) for s, v in sp.kept.items()} if sp else None}
            (d / 'meta.json').write_text(json.dumps(meta, indent=1))
            print(f"[done] {a.model} {method}/{pid} {total:.0f}s denoise {meta['denoise_s']:.0f}s warm {meta['warm_s']:.0f}s "
                  f"kept {np.mean(list(meta['kept_fraction'].values())) if sp and sp.kept else 1:.3f}", flush=True)


if __name__ == '__main__':
    main()
