"""Exact video->video block attention statistics, shared by the Hunyuan / Wan / MiniMax-H3 probes (torch only).

Blocks are BS consecutive tokens inside one latent frame (the last block of a frame may be shorter), so no block
straddles a frame. For a (step, layer) the exact softmax over all video keys is reduced to per-head block mass
maps [H, QB, KB], row-normalised per query block. From each map BlockStats keeps
  need[c]      oracle key-block fraction needed to cover a share c of the mass (COVER grid), averaged over head and query block
  reuse[s][t]  recall at step t of the oracle top-RHO mask chosen at step s (same layer), for every s < t
  maps         the fp16 maps at the steps in keep_steps (for cross-prompt static-mask analysis)
"""
import math

import torch
import torch.nn.functional as Fn

BS, RHO = 64, 0.15
COVER = [round(0.05 * i, 2) for i in range(1, 21)]


def block_map(q, k, T):
    """q, k [F*T, H, D] (video tokens, frame-major) -> exact block mass [H, F*nb, F*nb] (fp32)"""
    F, H, D = q.shape[0] // T, q.shape[1], q.shape[2]
    nb = math.ceil(T / BS)
    cnt = torch.full((nb,), BS, device=q.device)
    cnt[-1] = T - (nb - 1) * BS
    out = torch.empty(H, F * nb, F * nb, device=q.device)
    kt = k.transpose(0, 1)
    for f in range(F):
        qf = q[f * T:(f + 1) * T].transpose(0, 1)
        p = (torch.matmul(qf, kt.transpose(1, 2)).float() / math.sqrt(D)).softmax(-1)
        p = Fn.pad(p.view(H, T, F, T), (0, nb * BS - T)).view(H, T, F, nb, BS).sum(-1).flatten(2)
        p = Fn.pad(p, (0, 0, 0, nb * BS - T)).view(H, nb, BS, F * nb).sum(2)
        out[:, f * nb:(f + 1) * nb] = p / cnt[None, :, None]
        del p
    return out


def need_curve(ex):
    c = ex.sort(-1, descending=True).values.cumsum(-1)
    return [((c < x - 1e-6).sum(-1) + 1).float().mean().item() / ex.shape[-1] for x in COVER]


class BlockStats:
    def __init__(self, layers, keep_steps=()):
        self.layers, self.keep_steps = set(layers), set(keep_steps)
        self.reset()

    def reset(self):
        self.need, self.reuse, self.maps, self.masks = {}, {}, {}, {}

    def __call__(self, step, layer, q, k, T):
        if layer not in self.layers:
            return
        ex = block_map(q, k, T)
        n = round(RHO * ex.shape[-1])
        self.need[(step, layer)] = need_curve(ex)
        for (s, l), m in self.masks.items():
            if l == layer and s < step:
                self.reuse[(s, step, layer)] = (ex * m.to(ex.device)).sum(-1).mean().item()
        self.masks[(step, layer)] = torch.zeros_like(ex, dtype=torch.bool).scatter_(-1, ex.topk(n, -1).indices, True).cpu()
        self.reuse[(step, step, layer)] = (ex * self.masks[(step, layer)].to(ex.device)).sum(-1).mean().item()
        if step in self.keep_steps:
            self.maps[(step, layer)] = ex.half().cpu()

    def result(self):
        return {'need': {f'{s}|{l}': v for (s, l), v in self.need.items()},
                'reuse': {f'{s}|{t}|{l}': v for (s, t, l), v in self.reuse.items()}, 'cover': COVER, 'rho': RHO, 'bs': BS}
