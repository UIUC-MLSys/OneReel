"""Frame memory: per-step representative frames (CPU)."""
import torch

from onereel.frame_memory import FrameMemory
from onereel.layout import SceneLayout
from onereel.parallel.comm import needed


def test_shared_frames_and_exchange_include_extra():
    lay = SceneLayout([3, 4, 5], halo=0)
    lay.extra = {0: [9, 1, 5], 1: [0], 2: [10]}
    assert lay.shared_frames(0) == [5, 9]            # own frames dropped, sorted
    assert lay.key_frames(1) == [3, 4, 5, 6, 0]
    assert lay.shared_frames(2) == []
    T = 10
    assert needed(lay, T, (0, 30)) == [(50, 60), (90, 100)]


def test_fallback_then_next_step_applies_selection():
    lay = SceneLayout([3, 3, 3], halo=0)
    fm = FrameMemory(budget=2, fallback=(1, 'adj'))
    fm.reset(lay, n_layers=8)
    assert fm.layer == 4 and lay.extra == {0: [3], 1: [2, 6], 2: [5]}
    fm.begin_step(0)
    g = torch.Generator().manual_seed(0)
    base = torch.randn(2, 16, generator=g)
    feats = torch.cat([base[0].expand(5, -1), base[1].expand(4, -1)]) + 0.01 * torch.randn(9, 16, generator=g)
    frame = torch.arange(9).repeat_interleave(4)
    fm.collect(frame, feats.repeat_interleave(4, 0))
    fm.finish()
    assert lay.extra == {0: [3], 1: [2, 6], 2: [5]}  # same step: unchanged
    fm.begin_step(1)
    reps = fm.history[-1]
    assert len(reps) == 2 and min(reps) < 5 <= max(reps)   # one medoid per group
    for j, (lo, hi) in enumerate(lay.ranges):
        assert lay.extra[j] == [f for f in reps if not lo <= f < hi]


def test_select_is_deterministic_medoid_and_boundary_candidates():
    lay = SceneLayout([4, 4, 4], halo=0)
    g = torch.Generator().manual_seed(1)
    feats = torch.nn.functional.normalize(torch.randn(12, 8, generator=g), dim=-1)
    a, b = FrameMemory(budget=3), FrameMemory(budget=3)
    a.reset(lay, 4)
    b.reset(lay, 4)
    assert a.select(feats) == b.select(feats)
    c = FrameMemory(budget=3, candidates='boundary')
    c.reset(lay, 4)
    assert c.candidate_frames() == [0, 3, 4, 7, 8, 11]
    assert set(c.select(feats)) <= {0, 3, 4, 7, 8, 11}


def test_sink_is_read_by_every_scene_and_counts_in_budget():
    lay = SceneLayout([3, 3, 3], halo=0)
    fm = FrameMemory(budget=3, fallback=(0, 'adj'), sink=(0,))
    fm.reset(lay, n_layers=4)
    assert lay.extra == {0: [], 1: [0], 2: [0]}
    assert 0 not in fm.candidate_frames()
    feats = torch.nn.functional.normalize(torch.randn(9, 8, generator=torch.Generator().manual_seed(3)), dim=-1)
    reps = fm.select(feats)
    assert len(reps) == 2 and 0 not in reps


def test_per_layer_selection_switches_extra_per_block():
    lay = SceneLayout([3, 3, 3], halo=0)
    fm = FrameMemory(budget=1, fallback=(0, 'adj'), per_layer=True)
    fm.reset(lay, n_layers=2)
    assert fm.observes(0) and fm.observes(1)
    fm.begin_step(0)
    frame = torch.arange(9)
    a = torch.eye(9)[[0, 0, 0, 0, 0, 0, 0, 0, 8]] + 0.01 * torch.arange(9)[:, None]
    fm.collect(frame, torch.eye(9), 0)
    fm.collect(frame, a, 1)
    fm.finish()
    fm.begin_step(1)
    r0, r1 = fm.history[-1][0], fm.history[-1][1]
    fm.use_layer(0)
    e0 = dict(lay.extra)
    fm.use_layer(1)
    e1 = dict(lay.extra)
    assert len(r0) == len(r1) == 1
    assert e0 == {j: [f for f in r0 if not lo <= f < hi] for j, (lo, hi) in enumerate(lay.ranges)}
    assert e1 == {j: [f for f in r1 if not lo <= f < hi] for j, (lo, hi) in enumerate(lay.ranges)}
