# Modified by the OneReel authors from xDiT (Apache-2.0). See third_party/xDiT/MODIFICATIONS.md.
import torch
from typing import Optional

from diffusers.models.transformers.transformer_wan import WanAttnProcessor
from diffusers.models.transformers.transformer_wan import WanAttention

from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.model_executor.layers.attention_processor import (
    set_hybrid_seq_parallel_attn,
    xFuserAttentionProcessorRegister
)
from xfuser.envs import PACKAGES_CHECKER

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]

@xFuserAttentionProcessorRegister.register(WanAttnProcessor)
class xFuserWanAttnProcessor(WanAttnProcessor):

    def __init__(self):
        super().__init__()
        # qkv cache structure: store latest q/k/v for this processor instance
        # stored shapes are the query/key/value tensors after reshaping
        self._qkv_cache = None
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

        # Experimental cross-clip self-attention K/V injection.  Caching is
        # opt-in: the normal runtime path must not retain a full K/V copy for
        # every attention block.
        self._capture_cross_clip_kv: bool = False
        self._cross_clip_k: Optional[torch.Tensor] = None
        self._cross_clip_v: Optional[torch.Tensor] = None

        # Upstream K/V to inject, set by wan_executor before block forward.
        # Shape: same as _cross_clip_k/v above.
        self._upstream_k: Optional[torch.Tensor] = None
        self._upstream_v: Optional[torch.Tensor] = None
        self._upstream_hidden: Optional[torch.Tensor] = None
        self._upstream_token_stride: int = 1

        # Optional read-only diagnostics for text cross-attention.  The
        # executor enables this only on attn2 for selected layer/step pairs.
        self._text_diag_enabled: bool = False
        self._text_diag_token_groups: dict[str, list[int]] = {}
        self._text_diag_topk: int = 16
        self._text_diag_stats: Optional[dict] = None

        # Injection hyperparams (written by wan_executor per-forward).
        self._kv_inject_weight: float = 0.2
        self._kv_inject_mode: str = "concat"

    # ------------------------------------------------------------------
    # K/V injection API (called by wan_executor)
    # ------------------------------------------------------------------

    def configure_cross_clip_kv(
        self, *, capture: bool, weight: float = 0.2, mode: str = "concat"
    ) -> None:
        if mode not in {
            "concat", "replace", "weighted", "reproject", "same_layer_reproject"
        }:
            raise ValueError(f"Unsupported cross-clip K/V mode: {mode!r}")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("cross-clip K/V weight must be in [0, 1]")
        self._capture_cross_clip_kv = bool(capture)
        self._kv_inject_weight = float(weight)
        self._kv_inject_mode = mode

    def set_upstream_kv(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Set upstream K/V to be injected during the next __call__."""
        self._upstream_k = k
        self._upstream_v = v

    def set_upstream_hidden(
        self, hidden_states: torch.Tensor, token_stride: int = 1
    ) -> None:
        """Set upstream tokens for projection by this attention block."""
        self._upstream_hidden = hidden_states
        self._upstream_token_stride = token_stride

    def pop_cross_clip_kv(self):
        """
        Return (K, V) cached during last __call__ (post-RoPE, pre-transpose).
        Clears the cache. Returns (None, None) if nothing was cached.
        """
        k, v = self._cross_clip_k, self._cross_clip_v
        self._cross_clip_k = None
        self._cross_clip_v = None
        return k, v

    def configure_text_attention_diagnostic(
        self,
        *,
        enabled: bool,
        token_groups: Optional[dict[str, list[int]]] = None,
        topk: int = 16,
    ) -> None:
        """Configure read-only phrase-to-video token affinity capture."""
        self._text_diag_enabled = bool(enabled)
        self._text_diag_token_groups = token_groups or {}
        self._text_diag_topk = max(1, int(topk))
        self._text_diag_stats = None

    def pop_text_attention_stats(self) -> Optional[dict]:
        stats = self._text_diag_stats
        self._text_diag_stats = None
        return stats

    def _capture_text_attention_stats(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> None:
        """Summarize selected text-token QK affinity without changing attention.

        This intentionally computes only selected text columns.  The resulting
        spatial distribution is a phrase-affinity diagnostic, not the full
        cross-attention probability (whose denominator includes all text
        tokens).
        """
        if not self._text_diag_enabled or not self._text_diag_token_groups:
            return

        seq_len = key.shape[1]
        scale = query.shape[-1] ** -0.5
        groups = {}
        with torch.no_grad():
            query_float = query.detach().float()
            for name, raw_indices in self._text_diag_token_groups.items():
                indices = sorted({int(idx) for idx in raw_indices if 0 <= int(idx) < seq_len})
                if not indices:
                    continue
                selected_key = key[:, indices].detach().float()
                # [B, H, video_tokens, phrase_tokens]
                scores = torch.einsum(
                    "blhd,bshd->bhls", query_float, selected_key
                ) * scale
                # Aggregate phrase tokens, batch, and heads, then normalize
                # across video tokens to expose concentration/localization.
                affinity = torch.logsumexp(scores, dim=-1).mean(dim=(0, 1))
                distribution = torch.softmax(affinity, dim=0)
                k = min(self._text_diag_topk, distribution.numel())
                top_values, top_indices = torch.topk(distribution, k=k)
                entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum()
                entropy_scale = torch.log(
                    torch.tensor(float(distribution.numel()), device=entropy.device)
                ).clamp_min(1.0)
                normalized_entropy = entropy / entropy_scale
                groups[name] = {
                    "text_token_indices": indices,
                    "top_video_token_indices": top_indices.cpu().tolist(),
                    "top_video_token_weights": top_values.cpu().tolist(),
                    "topk_mass": float(top_values.sum().item()),
                    "normalized_entropy": float(normalized_entropy.item()),
                    "peak_weight": float(top_values[0].item()),
                }
        self._text_diag_stats = {
            "video_token_count": int(query.shape[1]),
            "text_token_count": int(key.shape[1]),
            "groups": groups,
        }

    def reset_kv_state(self) -> None:
        """
        Clear all K/V injection state at the start of each denoising step.
        Called from wan_executor.new_forward to prevent cross-step leakage.
        """
        self._cross_clip_k = None
        self._cross_clip_v = None
        # upstream_k/v should already be None after __call__ consumed it;
        # assert here to catch leaks early.
        assert self._upstream_k is None, \
            "[KV Injection] upstream_k leaked across timesteps — not consumed in __call__"
        assert self._upstream_v is None, \
            "[KV Injection] upstream_v leaked across timesteps — not consumed in __call__"
        assert self._upstream_hidden is None, \
            "[KV Injection] upstream_hidden leaked across timesteps — not consumed in __call__"

    # ------------------------------------------------------------------

    def _get_qkv_projections(self, attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # encoder_hidden_states is only passed for cross-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.fused_projections:
            if attn.cross_attention_dim_head is None:
                # In self-attention layers, we can fuse the entire QKV projection into a single linear
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                # In cross-attention layers, we can only fuse the KV projections into a single linear
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        return query, key, value

    def _get_added_kv_projections(self, attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
        if attn.fused_projections:
            key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
        else:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
        return key_img, value_img


    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = self._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Shape after unflatten: [B, L, num_heads, head_dim]
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # attn2 is configured by the executor; capture before the actual
        # attention call and leave query/key/value untouched.
        self._capture_text_attention_stats(query, key)

        upstream_key = upstream_value = None
        if self._upstream_hidden is not None:
            upstream_hidden = self._upstream_hidden.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            self._upstream_hidden = None
            _, upstream_key, upstream_value = self._get_qkv_projections(
                attn, upstream_hidden, None
            )
            upstream_key = attn.norm_k(upstream_key)
            upstream_key = upstream_key.unflatten(2, (attn.heads, -1))
            upstream_value = upstream_value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)
            if upstream_key is not None:
                # The memory clip uses the same local spatiotemporal grid as
                # the target clip, so encode its tokens on that grid once.
                upstream_rotary_emb = tuple(
                    freqs[:, ::self._upstream_token_stride] for freqs in rotary_emb
                )
                upstream_key = apply_rotary_emb(upstream_key, *upstream_rotary_emb)
                self._upstream_token_stride = 1

        # Cache the upstream key after its own positional encoding.  Injected
        # clips therefore retain their native spatiotemporal positions instead
        # of mixing an unrotated key with rotated downstream keys.
        if self._capture_cross_clip_kv:
            self._cross_clip_k = key.detach()
            self._cross_clip_v = value.detach()

        # ── K/V injection from upstream ───────────────────────────────
        # upstream_k/v are set by wan_executor before this block runs.
        # Upstream keys already carry the upstream clip's positional encoding;
        # do not apply the downstream RoPE a second time.
        if self._upstream_k is not None:
            with torch.no_grad():
                uk = self._upstream_k.to(device=key.device, dtype=key.dtype)
                uv = self._upstream_v.to(device=value.device, dtype=value.dtype)
                self._upstream_k = None   # consume immediately
                self._upstream_v = None

                if self._kv_inject_mode == "concat":
                    # Concat along sequence dim (dim=1): [B, L+L_up, H, D/H]
                    # Scale values, not keys: scaling keys changes attention
                    # logits/temperature rather than contribution magnitude.
                    key = torch.cat([key, uk], dim=1)
                    value = torch.cat([value, uv * self._kv_inject_weight], dim=1)
                elif self._kv_inject_mode == "replace":
                    key = uk
                    value = uv
                elif self._kv_inject_mode == "weighted":
                    if uk.shape != key.shape or uv.shape != value.shape:
                        raise ValueError(
                            "weighted cross-clip K/V requires matching local/upstream shapes"
                        )
                    weight = self._kv_inject_weight
                    key = (1.0 - weight) * key + weight * uk
                    value = (1.0 - weight) * value + weight * uv
                # attention_mask extension: USP does not accept an explicit mask,
                # so no mask adjustment is needed for the concat case.

        if upstream_key is not None:
            key = torch.cat([key, upstream_key], dim=1)
            value = torch.cat(
                [value, upstream_value * self._kv_inject_weight], dim=1
            )

        self._qkv_cache = {
            "q": query,  # shape: (batch, seq/heads, ...)
            "k": key,
            "v": value,
        }

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = self._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))


            hidden_states_img = USP(query.transpose(1, 2), key_img.transpose(1, 2), value_img.transpose(1, 2)).transpose(1, 2)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = USP(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def get_qkv_cache(self):
        """Return stored qkv cache for this processor (or None)."""
        return self._qkv_cache

    # Compatibility alias for the earlier experiment API.
    def pop_pre_rope_kv(self):
        return self.pop_cross_clip_kv()
