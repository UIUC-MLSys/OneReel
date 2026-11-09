"""OneReel with intra-scene Sol-Attn (HunyuanVideo-1.5): scene-structured attention whose scene-local part is sparse.

Runs as the attention override of OneReel's Ulysses path (onereel.models.hunyuan15.Processor.ulysses), where every
GPU holds all tokens of its heads, so every scene is complete. Per layer and scene j:
  video queries  Sol-Attn (official kernel, Sana's _run_sol_attn_bthd, released HunyuanVideo settings: tau 1.0, diag
                 threshold, no dense steps or layers) over [scene j's video | its shared frames, sentence j]; the
                 suffix is the kernel's exact KV sink, so the shared frames and the text stay exact and only the
                 scene-local video attention is sparsified
  text queries   dense, scene j's video and sentence j (as OneReel), one varlen call for all scenes
The kernel needs equal query and key lengths, so the sink positions also run as queries; their outputs are discarded
(shared-frame and text outputs come from their own scenes and from the text call). dense=True runs the same layout
with dense attention instead of the kernel (tests: it must equal OneReel's scene attention).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from onereel import attention as A
from onereel.parallel import comm

ROOT = Path(__file__).resolve().parents[1]
SOLK = dict(tau=1.0, thresh_type='diag', kv_splits='auto')       # Sol-Attn's HunyuanVideo configuration
_RUN = None


def sol_kernel():
    global _RUN
    if _RUN is None:
        os.environ['SOL_ATTN_STRICT'] = '1'                       # raise instead of a silent dense fallback
        sys.path.insert(0, str(ROOT / 'third_party/Sana'))
        from techniques.sparse_backends.sol_attn_backend import _run_sol_attn_bthd
        _RUN = _run_sol_attn_bthd
    return _RUN


class IntraSceneSol:
    """attention override (layer, q, k, v, eq, ek, ev) -> (video out, text out) on all tokens of the local heads"""

    def __init__(self, ctx, dense: bool = False):
        self.ctx, self.dense = ctx, dense
        self.run = None if dense else sol_kernel()

    def __call__(self, layer, q, k, v, eq, ek, ev):
        ctx, T, lay = self.ctx, self.ctx.T, self.ctx.layout
        rows = lambda x, frames: torch.cat([x[a:b] for a, b in comm.frame_tokens(frames, T)]) if frames else x[:0]
        ov, ot = torch.empty_like(q), torch.zeros_like(eq)
        text = []
        for j, ((lo, hi), (t0, t1)) in enumerate(zip(lay.ranges, ctx.text_ranges())):
            a, b, other = lo * T, hi * T, lay.shared_frames(j)
            qa, ka, va = (torch.cat([x[a:b], rows(x, other), e[t0:t1]]) for x, e in ((q, eq), (k, ek), (v, ev)))
            if self.dense:
                o = A.packed([(qa[:b - a], ka, va)])[0]
            else:
                o = self.run(*(x[None].to(torch.bfloat16).contiguous() for x in (qa, ka, va)),
                             sink_start=b - a, sink_tokens=qa.shape[0] - (b - a), **SOLK)[0][:b - a]
            ov[a:b] = o.to(q.dtype)
            text.append((eq[t0:t1], torch.cat([k[a:b], ek[t0:t1]]), torch.cat([v[a:b], ev[t0:t1]])))
        for (t0, t1), o in zip(ctx.text_ranges(), A.packed(text)):
            ot[t0:t1] = o
        return ov, ot
