# Modified by the OneReel authors from xDiT (Apache-2.0). See third_party/xDiT/MODIFICATIONS.md.
"""MiniMax-H3 transformer with Ulysses sequence parallelism.

MiniMax-H3 attends over one packed [text | audio | video] sequence with no mask, and every per-row input (AdaLN index,
timestep index, RoPE angles) is indexed by packed-sequence position. So the packed buffer is built on every rank,
zero-padded at the end to a multiple of the sequence-parallel size and split along the sequence; each attention
all-to-alls q / k / v to (full sequence, local heads), drops the padded keys and all-to-alls back; the two output heads
run on the local rows, which are gathered before the video / audio rows are selected. Needs diffusers with
MiniMaxH3Transformer3DModel (>= 0.41). Not registered in xfuser's model registry: import this module directly.
"""
from typing import Any

import torch
from flash_attn import flash_attn_func
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (MINIMAX_H3_MODALITY_NUM, MiniMaxH3Transformer3DModel,
                                                                  MiniMaxH3TransformerOutput, _apply_rotary_emb)

from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group
from xfuser.model_executor.layers.usp import _combined_qkv_all_to_all, _ft_c_output_all_to_all


class xFuserMiniMaxH3AttnProcessor:
    seq_len = None          # real (unpadded) packed length, set by the wrapper before the block stack

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attn.fused_projections:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = _apply_rotary_emb(q, *rotary_emb), _apply_rotary_emb(k, *rotary_emb)
        q, k, v = _combined_qkv_all_to_all(*(x.transpose(1, 2).contiguous() for x in (q, k, v)))  # [B, H/P, S_pad, D]
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))                                         # [B, S_pad, H/P, D]
        n = self.seq_len or k.shape[1]
        o = flash_attn_func(q, k[:, :n].contiguous(), v[:, :n].contiguous())
        o = _ft_c_output_all_to_all(o.transpose(1, 2).contiguous()).transpose(1, 2)              # [B, S_loc, H, D]
        o = o.flatten(2, 3).type_as(hidden_states)
        return attn.to_out[1](attn.to_out[0](o))


class xFuserMiniMaxH3Transformer3DWrapper(MiniMaxH3Transformer3DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._xfuser_proc = xFuserMiniMaxH3AttnProcessor()
        for block in self.transformer_blocks:                    # the text-only token refiner keeps its own processor
            block.attn.processor = self._xfuser_proc

    @staticmethod
    def _pad_chunk(x, pad, dim, rank, size):
        if pad:
            shape = list(x.shape)
            shape[dim] = pad
            x = torch.cat([x, x.new_zeros(shape)], dim=dim)
        return x.chunk(size, dim=dim)[rank]

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices, token_tags,
                position_ids, video_indices, audio_indices, text_indices, attention_kwargs: dict[str, Any] | None = None,
                return_dict: bool = True):
        S = position_ids.shape[0]
        rank, size = get_sequence_parallel_rank(), get_sequence_parallel_world_size()
        pad = (-S) % size
        cos, sin = self.rope(position_ids)

        video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
        text_embeds = self.token_refiner(text_embeds)
        x = text_embeds.new_zeros((text_embeds.shape[0], S, text_embeds.shape[-1]))
        x = x.index_copy(1, text_indices, text_embeds)
        x = x.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        x = x.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        x = self._pad_chunk(x, pad, 1, rank, size)
        rotary_emb = (self._pad_chunk(cos, pad, 0, rank, size), self._pad_chunk(sin, pad, 0, rank, size))
        adaln_local = self._pad_chunk(adaln_indices, pad, 0, rank, size)
        tidx_local = self._pad_chunk(timestep_indices, pad, 0, rank, size)
        self._xfuser_proc.seq_len = S
        for block in self.transformer_blocks:
            x = block(x, temb, adaln_local, rotary_emb)

        x = self.norm_out(x, temb, tidx_local).to(get_parameter_dtype(self.proj_out))
        video, audio = self.proj_out(x), self.audio_proj_out(x)
        if size > 1:
            video = get_sp_group().all_gather(video.contiguous(), dim=1)
            audio = get_sp_group().all_gather(audio.contiguous(), dim=1)
        video = video[:, :S].index_select(1, video_indices)
        audio = audio[:, :S].index_select(1, audio_indices)
        if not return_dict:
            return (video, audio)
        return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)
