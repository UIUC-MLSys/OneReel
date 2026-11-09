"""Scene layout over the latent frames of one video."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class SceneLayout:
    """lengths: latent frames per scene, in order. A scene also reads `halo` frames of the neighbouring scenes nearest
    to it: both neighbours ('adj'), only the previous ('prev') or only the next ('next'), and the frames in extra[j]
    (frame memory: representative frames of other scenes, replaced every denoising step)."""
    lengths: list[int]
    halo: int = 2
    sides: str = 'adj'
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.sides in ('adj', 'prev', 'next'), self.sides
        assert all(n > 0 for n in self.lengths), self.lengths

    @classmethod
    def even(cls, n_frames: int, n_scenes: int, **kw) -> 'SceneLayout':
        """scene of frame f = f * n_scenes // n_frames"""
        sc = np.arange(n_frames) * n_scenes // n_frames
        return cls([int((sc == j).sum()) for j in range(n_scenes)], **kw)

    @property
    def n_frames(self) -> int:
        return sum(self.lengths)

    @property
    def n_scenes(self) -> int:
        return len(self.lengths)

    @property
    def is_even(self) -> bool:
        return self.lengths == SceneLayout.even(self.n_frames, self.n_scenes).lengths

    @property
    def ranges(self) -> list[tuple[int, int]]:
        s = np.cumsum([0] + self.lengths)
        return [(int(s[j]), int(s[j + 1])) for j in range(self.n_scenes)]

    @property
    def frames(self) -> list[list[int]]:
        return [list(range(lo, hi)) for lo, hi in self.ranges]

    def scene_of_frame(self) -> torch.Tensor:
        return torch.repeat_interleave(torch.arange(self.n_scenes), torch.tensor(self.lengths))

    def halo_frames(self, j: int) -> list[int]:
        """frames of other scenes that scene j reads: the previous neighbour's last frames, then the next one's first"""
        fr, k, out = self.frames, self.halo, []
        if k == 0:
            return out
        if j > 0 and self.sides in ('adj', 'prev'):
            out += fr[j - 1][-k:]
        if j + 1 < self.n_scenes and self.sides in ('adj', 'next'):
            out += fr[j + 1][:k]
        return out

    def shared_frames(self, j: int) -> list[int]:
        """frames of other scenes that scene j reads: its halo frames, then its extra frames in order"""
        lo, hi = self.ranges[j]
        out = self.halo_frames(j)
        return out + [f for f in sorted(self.extra.get(j, ())) if not lo <= f < hi and f not in out]

    def key_frames(self, j: int) -> list[int]:
        """scene j's own frames, then its shared frames (the key order of the packed varlen layout)"""
        return self.frames[j] + self.shared_frames(j)

    def key_range(self, j: int) -> tuple[int, int]:
        """scene j's keys as one contiguous frame range (scene plus halo are adjacent)"""
        kf = self.key_frames(j)
        return min(kf), max(kf) + 1
