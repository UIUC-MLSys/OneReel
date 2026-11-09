"""Frame memory: cross-scene context from representative frames, chosen once per denoising step (or per layer).

During the conditional pass of step s, one layer's video hidden states are mean-pooled per latent frame (on several GPUs
each rank pools its own tokens and the per-frame sums are all-reduced, so every rank sees the same features). After the
pass, the candidate frames (every frame, or the first and last frame of every scene; sink frames excluded) are
clustered by k-means on the L2-normalised features into budget - len(sink) clusters (cosine distance, farthest-point
initialisation, deterministic); each cluster's medoid frame becomes a representative. At step s + 1, in every layer and
head, scene j reads the sink frames and the representatives that belong to other scenes as extra key frames
(SceneLayout.extra), through the same packed path as the halo frames. Step 0 has no features yet and reads the fallback
halo frames (and the sink frames) instead.

per_layer (ablation): every layer pools its own hidden states and gets its own representatives; the engine switches
SceneLayout.extra before each block (use_layer).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from onereel.layout import SceneLayout
from onereel.parallel import dist as D


class FrameMemory:
    def __init__(self, budget: int = 4, candidates: str = 'all', layer: int | None = None, fallback=(2, 'adj'),
                 sink=(), per_layer: bool = False, iters: int = 20):
        assert candidates in ('all', 'boundary'), candidates
        assert len(sink) < budget, 'the sink frames are part of the budget'
        self.budget, self.candidates, self.layer, self.fallback, self.iters = budget, candidates, layer, fallback, iters
        self.sink, self.per_layer = tuple(sink), per_layer
        self.layout, self.step, self.pending, self.sums, self.counts = None, -1, None, {}, {}
        self.history, self.extra = [], None                  # history: per step the representatives in use

    def reset(self, layout: SceneLayout, n_layers: int):
        """a new request: fallback halo frames (and the sink) at step 0"""
        self.layout, self.step, self.pending, self.sums, self.counts, self.history = layout, -1, None, {}, {}, []
        self.n_layers = n_layers
        if self.layer is None:
            self.layer = n_layers // 2
        h, sides = self.fallback
        halo = SceneLayout(layout.lengths, halo=h, sides=sides)
        first = self._extra([], halo)
        layout.extra = first
        self.extra = {l: first for l in range(n_layers)} if self.per_layer else first

    def _extra(self, reps, halo=None) -> dict:
        """scene -> frames of other scenes it reads: the sink, the representatives (and the fallback halo frames)"""
        out = {}
        for j, (lo, hi) in enumerate(self.layout.ranges):
            fr = set(self.sink) | set(reps) | set(halo.halo_frames(j) if halo else ())
            out[j] = sorted(f for f in fr if not lo <= f < hi)
        return out

    def observes(self, layer: int) -> bool:
        return self.per_layer or layer == self.layer

    def begin_step(self, step: int):
        """at every transformer forward: a new step applies the representatives chosen in the previous one"""
        if step == self.step:
            return
        self.step = step
        if self.pending is not None:
            reps, self.pending = self.pending, None
            self.history.append(reps)
            if self.per_layer:
                self.extra = {l: self._extra(r) for l, r in reps.items()}
            else:
                self.extra = self._extra(reps)
                self.layout.extra = self.extra

    def use_layer(self, layer: int):
        """per_layer: the extra frames of this layer (called before each block)"""
        if self.per_layer and self.extra is not None:
            self.layout.extra = self.extra[layer]

    def collect(self, frame: torch.Tensor, hs: torch.Tensor, layer: int | None = None):
        """frame [N] latent frame of each local row (-1: not video); hs [N, C] hidden states of an observed layer"""
        layer = self.layer if layer is None else layer
        n_frames = self.layout.n_frames
        if layer not in self.sums:
            self.sums[layer] = hs.new_zeros(n_frames, hs.shape[-1], dtype=torch.float32)
            self.counts[layer] = hs.new_zeros(n_frames, dtype=torch.float32)
        keep = frame >= 0
        self.sums[layer].index_add_(0, frame[keep], hs[keep].float())
        self.counts[layer].index_add_(0, frame[keep], torch.ones_like(frame[keep], dtype=torch.float32))

    @torch.no_grad()
    def finish(self):
        """after the conditional pass: all-reduce the frame features and choose the next step's representatives"""
        if not self.sums:
            return
        layers = sorted(self.sums)
        sums, counts = torch.stack([self.sums[l] for l in layers]), torch.stack([self.counts[l] for l in layers])
        D.all_reduce(sums)                                      # one collective for every observed layer
        D.all_reduce(counts)
        self.sums, self.counts = {}, {}
        feats = F.normalize(sums / counts.clamp_min(1)[..., None], dim=-1)
        if self.per_layer:
            self.pending = {l: self.select(f) for l, f in zip(layers, feats)}
        else:
            self.pending = self.select(feats[0])

    def candidate_frames(self) -> list[int]:
        if self.candidates == 'all':
            cand = range(self.layout.n_frames)
        else:
            cand = sorted({f for lo, hi in self.layout.ranges for f in (lo, hi - 1)})
        return [f for f in cand if f not in self.sink]

    def select(self, feats: torch.Tensor) -> list[int]:
        """k-means (cosine) over the candidate frames; the medoid (nearest member, lowest index on ties) of each cluster"""
        cand = torch.tensor(self.candidate_frames(), device=feats.device)
        x = feats[cand]
        k = min(self.budget - len(self.sink), len(cand))
        centre = [0]                                                     # farthest-point initialisation
        for _ in range(k - 1):
            d = (1 - x @ x[centre].T).min(1).values
            centre.append(int(d.argmax()))
        c = x[centre].clone()
        for _ in range(self.iters):
            z = (x @ c.T).argmax(1)
            new = torch.stack([F.normalize(x[z == i].mean(0), dim=0) if (z == i).any() else c[i] for i in range(k)])
            if torch.equal(new, c):
                break
            c = new
        z = (x @ c.T).argmax(1)
        reps = []
        for i in range(k):
            m = (z == i).nonzero().flatten()
            if len(m):
                reps.append(int(cand[m[(x[m] @ c[i]).argmax()]]))       # argmax returns the first maximum
        return sorted(reps)
