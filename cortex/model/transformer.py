"""The CORTEX-0 decoder-only Transformer.

The module layout deliberately mirrors Llama (pre-norm RMSNorm, RoPE,
grouped-query attention, SwiGLU, tied embeddings) so that checkpoints can be
converted to ``transformers.LlamaForCausalLM`` with a pure key remapping.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..config import CortexConfig
from .attention import CortexAttention, KVCache
from .layers import RMSNorm, RotaryEmbedding, SwiGLU


class CortexDecoderLayer(nn.Module):
    def __init__(self, config: CortexConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CortexAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, cache = self.self_attn(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, cache


class CortexModel(nn.Module):
    """Base transformer producing hidden states."""

    def __init__(self, config: CortexConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [CortexDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def _prepare_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        q_len: int,
        kv_len: int,
        past_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an additive ``[B, 1, q_len, kv_len]`` causal (+padding) mask."""
        q_positions = torch.arange(past_len, past_len + q_len, device=device)
        k_positions = torch.arange(kv_len, device=device)
        mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_len)
        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape != (batch_size, kv_len):
                raise ValueError(
                    f"attention_mask must be [B, {kv_len}], got {tuple(attention_mask.shape)}"
                )
            mask = mask & attention_mask[:, None, None, :].bool()
        additive = torch.zeros(mask.shape, dtype=dtype, device=device)
        return additive.masked_fill(~mask, torch.finfo(dtype).min)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[KVCache]]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch, seq]")
        batch_size, seq_len = input_ids.shape
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].shape[2]
        kv_len = past_len + seq_len
        if kv_len > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {kv_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        if position_ids is None:
            position_ids = torch.arange(past_len, kv_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape != (batch_size, seq_len):
            raise ValueError(f"position_ids must be [{batch_size}, {seq_len}]")

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        attention_mask_4d = self._prepare_causal_mask(
            attention_mask,
            batch_size,
            seq_len,
            kv_len,
            past_len,
            hidden_states.dtype,
            hidden_states.device,
        )

        new_cache: Optional[List[KVCache]] = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = None
            if past_key_values is not None and idx < len(past_key_values):
                layer_past = past_key_values[idx]
            if self.gradient_checkpointing and self.training:
                hidden_states, cache = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    cos,
                    sin,
                    attention_mask_4d,
                    layer_past,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, cache = layer(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask=attention_mask_4d,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
            if new_cache is not None:
                new_cache.append(cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_cache


class CortexForCausalLM(nn.Module):
    """Causal language model with next-token-prediction head."""

    def __init__(self, config: CortexConfig) -> None:
        super().__init__()
        self.config = config
        self.model = CortexModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.model.embed_tokens.weight.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: bool = False,
        **_: object,
    ) -> dict:
        hidden_states, new_cache = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss, "past_key_values": new_cache}
