"""Can the out-of-shot attention tail be compressed instead of dropped? Offline error of video->video attention outputs.

Called with one layer's video q, k, v [S, H, D] (frame-major, T tokens per latent frame) on the dense trajectory. Latent
frames are split evenly into K shots; the scope of shot j is its frames plus HALO frames of each adjacent shot. Variants:
  drop      scope keys only (the current method)
  pool{c}   scope keys exact + out-of-scope keys pooled per c consecutive tokens of one frame (c = 'frame': whole frame);
            a pooled key / value is the mean, its logit gets + log(n) so it stands for n keys; the two parts are merged
            exactly by log-sum-exp
Reported per (step, layer): relative L2 error ||o - o_dense|| / ||o_dense|| over all video queries, and keys per query
relative to dense. Text keys are not included in any variant (video->video only).
"""
import math

import torch
from flash_attn import flash_attn_func


class StopGeneration(Exception):
    pass


def pooled(k, v, idx, T, c):
    """mean-pooled keys / values of the token index list idx (whole frames), groups of c tokens inside a frame"""
    c = T if c == 'frame' else c
    kk, vv = k[idx].float(), v[idx].float()
    n_frames = len(idx) // T
    nb = math.ceil(T / c)
    pad = nb * c - T
    kk = torch.nn.functional.pad(kk.view(n_frames, T, *kk.shape[1:]), (0, 0, 0, 0, 0, pad))
    vv = torch.nn.functional.pad(vv.view(n_frames, T, *vv.shape[1:]), (0, 0, 0, 0, 0, pad))
    cnt = torch.full((nb,), float(c), device=k.device)
    cnt[-1] = T - (nb - 1) * c
    kp = kk.view(n_frames, nb, c, *kk.shape[2:]).sum(2) / cnt[None, :, None, None]
    vp = vv.view(n_frames, nb, c, *vv.shape[2:]).sum(2) / cnt[None, :, None, None]
    return kp.flatten(0, 1), vp.flatten(0, 1), cnt.log().repeat(n_frames)


class TailProbe:
    def __init__(self, layers, steps, n_scenes, halo=2, pools=(16, 64, 256, 'frame')):
        self.layers, self.steps, self.K, self.halo, self.pools = set(layers), set(steps), n_scenes, halo, pools
        self.rows, self.last = [], max(steps)

    def __call__(self, step, layer, q, k, v, T):
        if step > self.last:
            raise StopGeneration
        if step not in self.steps or layer not in self.layers:
            return
        S, H, D = q.shape
        F = S // T
        scene = torch.arange(F) * self.K // F
        frames = [torch.nonzero(scene == j).flatten().tolist() for j in range(self.K)]
        tok = lambda fs: torch.cat([torch.arange(f * T, (f + 1) * T) for f in fs]).to(q.device)
        ref = flash_attn_func(q[None], k[None], v[None])[0].float()
        err = {name: 0.0 for name in ['drop'] + [f'pool{c}' for c in self.pools]}
        keys = {name: 0.0 for name in err}
        scale, denom = 1 / math.sqrt(D), 0.0
        for j in range(self.K):
            sc = set(frames[j])
            if j > 0:
                sc |= set(frames[j - 1][-self.halo:])
            if j < self.K - 1:
                sc |= set(frames[j + 1][:self.halo])
            qi, ki = tok(frames[j]), tok(sorted(sc))
            out = sorted(set(range(F)) - sc)
            qj = q[qi]
            oa, lse_a, _ = flash_attn_func(qj[None], k[ki][None], v[ki][None], return_attn_probs=True)
            oa, lse_a = oa[0].float(), lse_a[0].float()                          # [Lq, H, D], [H, Lq]
            r = ref[qi]
            denom += r.pow(2).sum().item()
            err['drop'] += (oa - r).pow(2).sum().item()
            keys['drop'] += len(qi) * len(ki)
            for c in self.pools:
                if not out:
                    err[f'pool{c}'] += (oa - r).pow(2).sum().item()
                    keys[f'pool{c}'] += len(qi) * len(ki)
                    continue
                kp, vp, bias = pooled(k, v, tok(out), T, c)
                ob = torch.empty_like(oa)
                lse_b = torch.empty_like(lse_a)
                for h in range(H):                                              # per head keeps memory bounded
                    lg = (qj[:, h].float() @ kp[:, h].T) * scale + bias
                    lse_b[h] = torch.logsumexp(lg, -1)
                    ob[:, h] = torch.softmax(lg, -1) @ vp[:, h]
                m = torch.maximum(lse_a, lse_b)
                wa, wb = (lse_a - m).exp().T[..., None], (lse_b - m).exp().T[..., None]
                o = (oa * wa + ob * wb) / (wa + wb)
                err[f'pool{c}'] += (o - r).pow(2).sum().item()
                keys[f'pool{c}'] += len(qi) * (len(ki) + kp.shape[0])
        for name in err:
            self.rows.append({'step': step, 'layer': layer, 'variant': name, 'rel_err': math.sqrt(err[name] / denom),
                              'keys_vs_dense': keys[name] / (S * S)})
