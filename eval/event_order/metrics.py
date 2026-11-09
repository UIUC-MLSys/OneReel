"""Per-video metrics for the event-order benchmark."""
import cv2
import numpy as np
from scipy.stats import spearmanr

CUT_RATIO = 2.0   # a boundary whose warp ratio reaches this counts as a hard cut


def segment_index(n, ranges):
    seg_of = np.zeros(n, int)
    for j, (lo, hi) in enumerate(ranges):
        seg_of[lo:min(hi, n)] = j
    return seg_of


def text_following(sc, frames, sentences, ranges):
    """CLIP text following plus the LPIPS / whole-frame DINO continuity terms (cached together in report/scores)"""
    n = len(frames)
    seg_of = segment_index(n, ranges)
    sim = sc.clip_sim(frames, sentences) * 100                       # [n, 3]
    M = np.stack([sim[seg_of == j].mean(0) for j in range(3)])      # [segment, sentence]
    win = sim.argmax(1)
    order = spearmanr(np.arange(n), win)[0] if len(set(win.tolist())) > 1 else 0.0
    lp = sc.lpips_consecutive(frames)
    b = seg_of[1:] != seg_of[:-1]
    emb = sc.dino_cls(frames)
    means = np.stack([emb[seg_of == j].mean(0) for j in range(3)])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    cs = means @ means.T
    return {'seg_acc': float((M.argmax(1) == np.arange(3)).mean()),
            'seg_margin': float(np.mean([M[j, j] - np.delete(M[j], j).max() for j in range(3)])),
            'frame_acc': float((win == seg_of).mean()), 'order': float(order),
            'boundary_jump': float(lp[b].mean() / lp[~b].mean()), 'boundary_lpips': float(lp[b].mean()),
            'within_lpips': float(lp[~b].mean()), 'cross_dino': float(cs[np.triu_indices(3, 1)].mean())}


def warp_errors(frames, size=(424, 240)):
    """[n-1] mean abs RGB error after warping frame t onto t+1 with Farneback flow (no occlusion mask, so cuts stay visible)"""
    g = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in g]
    xs, ys = np.meshgrid(np.arange(size[0], dtype=np.float32), np.arange(size[1], dtype=np.float32))
    err = []
    for t in range(len(g) - 1):
        fl = cv2.calcOpticalFlowFarneback(gray[t + 1], gray[t], None, 0.5, 4, 21, 3, 5, 1.1, 0)
        w = cv2.remap(g[t], xs + fl[..., 0], ys + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err.append(np.abs(w.astype(np.float32) - g[t + 1].astype(np.float32)).mean() / 255)
    return np.array(err)


def transition_ok(seg_correct, boundary_ratios, tau=CUT_RATIO):
    return float(np.mean([seg_correct[k] and seg_correct[k + 1] and boundary_ratios[k] < tau for k in range(len(boundary_ratios))]))


def continuity(sc, gd, frames, sentences, ranges, subject):
    """warp-ratio smoothness, subject-region DINO and transition success (cached in report/continuity)"""
    seg_of = segment_index(len(frames), ranges)
    sim = sc.clip_sim(frames, sentences)
    correct = [bool(sim[seg_of == j].mean(0).argmax() == j) for j in range(3)]
    err = warp_errors(frames)
    b = seg_of[1:] != seg_of[:-1]
    within = err[~b].mean()
    ratios = (err[b] / within).tolist()
    embs, det = [], 0
    for j in range(3):
        idx = np.flatnonzero(seg_of == j)
        pick = idx[np.linspace(2, len(idx) - 3, 4).round().astype(int)]
        e, k = gd.crop_emb([frames[i] for i in pick], subject)
        det += k
        if e is not None:
            m = e.mean(0)
            embs.append(m / np.linalg.norm(m))
    sd = float(np.mean([embs[i] @ embs[j] for i in range(len(embs)) for j in range(i + 1, len(embs))])) if len(embs) > 1 else float('nan')
    return {'seg_correct': correct, 'boundary_ratios': ratios, 'warp_ratio': float(np.mean(ratios)),
            'warp_boundary': float(err[b].mean()), 'warp_within': float(within), 'subject_dino': sd,
            'subject_det': det / 12, 'transition_ok': transition_ok(correct, ratios), 'n_subject_segments': len(embs)}
