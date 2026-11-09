"""shared_pos = 'adjacent' (HunyuanVideo-1.5): a shared frame's key, rotated at its own frame and moved by _move_shared,
must equal the same key rotated at its adjacent position (3D RoPE table built as the model's: temporal, height, width
frequencies on separate dimension groups, interleaved real pairs).
  python tests/test_shared_pos.py"""
import sys
from pathlib import Path

import torch
from diffusers.models.embeddings import apply_rotary_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from onereel.layout import SceneLayout  # noqa: E402
from onereel.models import hunyuan15 as HY  # noqa: E402

F, Hh, Ww, D, H = 30, 4, 6, 128, 2
T = Hh * Ww
f, y, x = torch.meshgrid(torch.arange(F), torch.arange(Hh), torch.arange(Ww), indexing='ij')
groups = [(f, 16), (y, 24), (x, 24)]                                   # pairs per axis: 16 + 24 + 24 = 64 = D / 2
ang = torch.cat([p.reshape(-1, 1).float() * (1e4 ** (-torch.arange(n).float() / n)) for p, n in groups], 1)
cos, sin = ang.cos().repeat_interleave(2, 1), ang.sin().repeat_interleave(2, 1)
lay = SceneLayout([8, 7, 9, 6], halo=0)
lay.extra = {0: [12, 20, 27], 1: [2, 20, 27], 2: [2, 12, 27], 3: [2, 12, 20]}
ctx = HY.Context(lay, T=T, shared_pos='adjacent')
ctx._rot_full = (cos, sin)
rot = lambda k, rows: apply_rotary_emb(k[None], (cos[rows], sin[rows]), sequence_dim=1)[0]
for j in range(lay.n_scenes):
    frames = lay.key_frames(j)
    k0 = torch.randn(len(frames) * T, H, D, dtype=torch.float64).float()
    src = torch.cat([torch.arange(fr * T, (fr + 1) * T) for fr in frames])
    dst_frames = lay.frames[j] + HY.adjacent_positions(lay, j)
    dst = torch.cat([torch.arange(fr * T, (fr + 1) * T) for fr in dst_frames])
    moved = HY._move_shared(ctx, j, rot(k0, src))
    err = (moved - rot(k0, dst)).abs().max().item()
    print(f'scene {j} {lay.ranges[j]}: shared {lay.shared_frames(j)} -> {HY.adjacent_positions(lay, j)}, max err {err:.1e}')
    assert err < 1e-4, err

# scene-local RoPE: own frames at their index within the scene (+1 after scene 0), shared frames just after the scene
ctx = HY.Context(lay, T=T, shared_pos='adjacent', scene_rope=True)
ctx._rot_full = (cos, sin)
for j in range(lay.n_scenes):
    frames = lay.key_frames(j)
    k0 = torch.randn(len(frames) * T, H, D)
    src = torch.cat([torch.arange(HY.frame_position(ctx, fr) * T, (HY.frame_position(ctx, fr) + 1) * T) for fr in frames])
    dst_frames = [HY.frame_position(ctx, fr) for fr in lay.frames[j]] + HY.shared_positions(ctx, j)
    dst = torch.cat([torch.arange(fr * T, (fr + 1) * T) for fr in dst_frames])
    moved = HY._move_shared(ctx, j, rot(k0, src))
    err = (moved - rot(k0, dst)).abs().max().item()
    print(f'scene-local scene {j}: own {dst_frames[:len(lay.frames[j])][:2]}.., shared {lay.shared_frames(j)} -> '
          f'{HY.shared_positions(ctx, j)}, max err {err:.1e}')
    assert err < 1e-4, err
