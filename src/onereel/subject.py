"""Subject links: a per-frame subject mask from video-to-subject-token logits, and the key frames that share it."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SubjectMask:
    """Accumulates, at one forward over a range of layers, the mean logit of every video token to the subject tokens of
    its own sentence; after the last layer smooths it 3x3 per frame and keeps the top `top` fraction of each frame."""

    def __init__(self, step: int, layers: tuple[int, int], top: float, grid: tuple[int, int]):
        self.step, self.layers, self.top, self.grid = step, layers, top, grid
        self.score = None
        self.mask = None                       # [F, T] bool once built

    def reset(self):
        self.score = self.mask = None

    def active(self, step: int) -> bool:
        return self.mask is None and step == self.step

    def add(self, layer: int, score: torch.Tensor):
        """score [F, T] for this layer"""
        if self.layers[0] <= layer < self.layers[1]:
            self.score = score if self.score is None else self.score + score
        if layer == self.layers[1] - 1:
            h, w = self.grid
            sc = F.avg_pool2d(self.score.view(-1, 1, h, w), 3, 1, 1, count_include_pad=False).view(self.score.shape)
            self.mask = sc >= sc.quantile(1 - self.top, dim=1, keepdim=True)


def key_frames(frames: list[int], n: int) -> list[int]:
    """n evenly spaced interior frames of a scene"""
    return [frames[int(round(x))] for x in np.linspace(0, len(frames) - 1, n + 2)[1:-1]]


def logit_score(q_rows: torch.Tensor, k_subj: torch.Tensor) -> torch.Tensor:
    """mean logit of every query row [N, H, D] to the subject keys [n, H, D] over heads and keys"""
    k_subj = k_subj.float()
    return torch.einsum('shd,nhd->s', q_rows.float(), k_subj) / (k_subj.shape[0] * k_subj.shape[1])
