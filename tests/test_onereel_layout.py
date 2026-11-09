import torch

from onereel.layout import SceneLayout
import numpy as np

from onereel.memory import candidate_frames
from onereel.parallel.comm import balanced, choose_path, clip, frame_tokens, merge, needed, partition, query_cost, runs


def test_even_matches_floor_split():
    lay = SceneLayout.even(21, 3)
    assert lay.lengths == [7, 7, 7]
    sc = torch.arange(20) * 3 // 20
    assert SceneLayout.even(20, 3).scene_of_frame().tolist() == sc.tolist()


def test_halo_order_and_sides():
    lay = SceneLayout([3, 5, 7], halo=2)
    assert lay.halo_frames(0) == [3, 4]
    assert lay.halo_frames(1) == [1, 2, 8, 9]
    assert lay.key_frames(1) == [3, 4, 5, 6, 7, 1, 2, 8, 9]
    assert lay.key_range(1) == (1, 10)
    assert SceneLayout([3, 5, 7], halo=2, sides='next').halo_frames(1) == [8, 9]
    assert SceneLayout([3, 5, 7], halo=2, sides='prev').halo_frames(1) == [1, 2]
    assert SceneLayout([3, 5], halo=0).halo_frames(0) == []


def test_partition_and_exchange_sets():
    T = 1590
    lay = SceneLayout([8] * 6, halo=2)
    own = partition(lay, 2, T, 'scene')
    assert own == [(0, 24 * T), (24 * T, 48 * T)]
    assert needed(lay, T, own[0]) == [(24 * T, 26 * T)]
    assert needed(lay, T, own[1]) == [(22 * T, 24 * T)]
    lay = SceneLayout([3, 5, 7, 9, 11, 13], halo=2)
    own = partition(lay, 2, T, 'scene')
    b = own[0][1]
    assert 24 * T < b < 35 * T and b % T != 0             # the boundary falls inside scene 4, inside a frame
    assert needed(lay, T, own[0]) == [(b, 37 * T)]       # the rest of scene 4 and its next halo
    assert needed(lay, T, own[1]) == [(22 * T, b)]       # scene 4's previous halo and its first part
    assert partition(lay, 2, T, 'ulysses') == [(0, 24 * T), (24 * T, 48 * T)]


def test_balanced_partition_p_ranks():
    lay = SceneLayout([3, 5, 7, 9, 11, 13, 2, 6], halo=2)
    T = 1590
    cost = query_cost(lay, T)
    per_token = np.concatenate([[c] * (m * T) for c, m in zip(cost, lay.lengths)])
    for n in (2, 3, 4, 8):
        own = partition(lay, n, T, 'scene')
        assert own[0][0] == 0 and own[-1][1] == lay.n_frames * T
        assert all(x[1] == y[0] and x[0] < x[1] for x, y in zip(own, own[1:]))
        cuts = sum(any(lo * T < x[0] < hi * T for x in own) for lo, hi in lay.ranges)
        assert cuts <= n - 1                              # at most n - 1 scenes are cut
        work = [per_token[a:b].sum() for a, b in own]
        assert max(abs(w - sum(work) / n) for w in work) <= per_token.max() + 1e-9   # within one query of the target
    assert balanced([1, 1], [1.0, 100.0], 2) == [(0, 1), (1, 2)]


def test_eq2_path_choice():
    T = 1590
    many = SceneLayout([4] * 12, halo=2)                 # many short scenes: scenes stay whole, little to gather
    own = partition(many, 4, T, 'scene')
    assert choose_path(many, own, T, 1.0)['path'] == 'scene'
    one = SceneLayout([40, 2, 2, 4], halo=2)             # one dominant scene spans every GPU
    own = partition(one, 4, T, 'scene')
    c = choose_path(one, own, T, 1.0)
    assert c['path'] == 'ulysses' and c['V_G'] > c['V_U']
    mem = SceneLayout([4] * 12, halo=0)                  # the memory also gathers every boundary candidate
    own = partition(mem, 4, T, 'scene', extra_keys=128)
    assert needed(mem, T, own[0], candidate_frames(mem)[0])[:2] == [(12 * T, 13 * T), (15 * T, 17 * T)]


def test_interval_helpers():
    assert merge([(5, 7), (0, 2), (2, 3), (6, 9)]) == [(0, 3), (5, 9)]
    assert clip([(0, 3), (5, 9)], 2, 6) == [(2, 3), (5, 6)]
    assert frame_tokens([3, 4, 1, 2], 10) == [(30, 50), (10, 30)]


def test_runs():
    assert runs([3, 4, 5, 1, 2, 8, 9]) == [(3, 6), (1, 3), (8, 10)]
