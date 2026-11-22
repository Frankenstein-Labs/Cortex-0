"""The CORTEX-0 causal language model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cortex.config import CortexConfig
from cortex.model.layers import Attention, RMSNorm, SwiGLU, scaled_init


class Block(nn.Module):
    def __init__(self, cfg: CortexConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.max_seq_len,
            cfg.rope_theta,
            cfg.dropout,
            cfg.bias,
        )
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(self.attn_norm(x), past_kv, use_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, present


class CortexLM(nn.Module):
    """Decoder-only Transformer with tied input/output embeddings."""

    def __init__(self, cfg: CortexConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        scaled_init(self, cfg.init_std, cfg.n_layers)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        loss_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.drop(self.tok_emb(input_ids))
        presents: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if past_kv is None else past_kv[i], use_cache)
            if presents is not None:
                presents.append(present)  # type: ignore[arg-type]
        x = self.norm(x)
        logits = self.lm_head(x)

        out: dict[str, Any] = {"logits": logits, "hidden": x}
        if presents is not None:
            out["past_kv"] = presents
        if targets is not None:
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            if loss_mask is not None:
                flat_mask = loss_mask.reshape(-1).bool()
                flat_logits = flat_logits[flat_mask]
                flat_targets = flat_targets[flat_mask]
            out["loss"] = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
            with torch.no_grad():
                valid = flat_targets != -100
                if valid.any():
                    pred = flat_logits.argmax(dim=-1)
                    out["token_accuracy"] = (pred[valid] == flat_targets[valid]).float().mean()
                else:
                    out["token_accuracy"] = logits.new_tensor(0.0)
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        from cortex.generation.sampler import sample_next_token

        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        generated = input_ids
        past_kv = None
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        for _ in range(max_new_tokens):
            if past_kv is None or not use_cache:
                model_input = generated[:, -self.cfg.max_seq_len :]
            else:
                model_input = generated[:, -1:]
            out = self.forward(model_input, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                past_kv = out["past_kv"]
            logits = out["logits"][:, -1, :]
            if repetition_penalty and repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, generated, repetition_penalty)
            next_id = sample_next_token(
                logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
            )
            generated = torch.cat((generated, next_id.view(-1, 1)), dim=1)
            if eos_token_id is not None and bool((next_id == eos_token_id).any()):
                break
            if generated.shape[1] >= self.cfg.max_seq_len:
                break
        return generated


def apply_repetition_penalty(logits: torch.Tensor, ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """Penalise logits of tokens that already appear in the sequence."""
    logits = logits.clone()
    for row in range(ids.shape[0]):
        seen = torch.unique(ids[row])
        selected = logits[row, seen]
        logits[row, seen] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits


def save_checkpoint(model: CortexLM, path: str | Path, extra: dict[str, Any] | None = None) -> None:
    from safetensors.torch import save_model

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model
    save_model(model_to_save, str(path))
    meta = {"model": model.cfg.to_dict(), "extra": extra or {}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_model(path: str | Path, device: str | torch.device = "cpu") -> CortexLM:
    """Load a model from a safetensors checkpoint plus its ``.json`` sidecar."""
    from safetensors.torch import load_model as load_safetensors

    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    cfg = CortexConfig.from_dict(meta["model"])
    model = CortexLM(cfg)
    load_safetensors(model, str(path), device=str(device))
    model.to(device)
    model.eval()
    return model
