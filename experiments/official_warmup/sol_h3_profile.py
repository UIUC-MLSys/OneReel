"""Profile Sol-Attn on MiniMax-H3's real attention inputs (hook of experiments/minimax_h3_scene.py, setting solprof_dense).

At STEPS x LAYERS of the dense trajectory, on the full packed [text | audio | video] q / k / v:
  dense_ms   FlashAttention over the whole sequence
  sol_ms     the official H3 Sol-Attn setting (config/minimax_h3/kernel_sol.toml: tau 1.0, diag threshold, the text +
             audio prefix as exact KV sink, dense prefix queries): Sol-Attn kernel on the sequence plus FlashAttention
             for the prefix queries
  exact      share of video-query x video-key blocks Sol-Attn computes exactly (routing of the official Triton
             reference, sol_attn.triton_ref.preprocess.prepare)
Raises StopGeneration after the last step.
"""
import math
import sys
from pathlib import Path

import torch
from flash_attn import flash_attn_func

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'third_party' / 'Sana' / 'techniques' / 'sparse_backends'))
import sol_attn  # noqa: E402
from sol_attn.triton_ref.preprocess import prepare  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tail_compression import StopGeneration  # noqa: E402

STEPS, LAYERS, TAU = (0, 10, 25, 40), (8, 24, 41), 1.0


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


class SolProfile:
    def __init__(self):
        self.rows, self.kv_splits = [], None

    def __call__(self, step, layer, q, k, v, St):
        if step > max(STEPS):
            raise StopGeneration
        if step not in STEPS or layer not in LAYERS:
            return
        q, k, v = (x[None].contiguous() for x in (q, k, v))                     # [1, S, H, D]
        P = int(St.vid_idx.min())                                               # rows before the first video row
        S = q.shape[1]
        if self.kv_splits is None:
            self.kv_splits = 1
        dense_ms = timed(lambda: flash_attn_func(q, k, v))

        def sol():
            o = sol_attn.sol_attn(q, k, v, tau=TAU, thresh_type='diag', kv_splits=self.kv_splits, sink_start=0, sink_tokens=P)
            o[:, :P] = flash_attn_func(q[:, :P], k, v)
            return o
        sol_ms = timed(sol)
        bs = 64
        nt = math.ceil(S / bs)
        kc, _, thr = prepare(q, k, v, tau=TAU, scale=q.shape[-1] ** -0.5, thresh_type='diag')
        kc, thr = kc[:, :nt], thr.reshape(1, -1, q.shape[2])[:, :nt]
        pad = nt * bs - S
        qp = torch.nn.functional.pad(q.float(), (0, 0, 0, 0, 0, pad)).view(1, nt, bs, *q.shape[2:])
        cnt = torch.full((nt,), float(bs), device=q.device)
        cnt[-1] = S - (nt - 1) * bs
        qm = qp.sum(2) / cnt[None, :, None, None]
        sc = torch.einsum('bihd,bjhd->bhij', qm, kc.float()) * q.shape[-1] ** -0.5 * 1.4426950408889634
        idx = torch.arange(nt, device=q.device)
        exact = (sc > thr.float().permute(0, 2, 1)[..., None]) | ((idx[:, None] - idx[None]).abs() <= 1)
        v0 = math.ceil(P / bs)                                                  # first block holding only video rows
        self.rows.append({'step': step, 'layer': layer, 'seq': S, 'prefix': P, 'dense_ms': dense_ms, 'sol_ms': sol_ms,
                          'exact': exact[..., v0:, v0:].float().mean().item()})
        print(f"[solprof] step {step} layer {layer}: dense {dense_ms:.1f} ms sol {sol_ms:.1f} ms exact {self.rows[-1]['exact']:.3f}", flush=True)

    def result(self):
        return {'steps': STEPS, 'layers': LAYERS, 'tau': TAU, 'rows': self.rows}
