"""Group Relative Policy Optimisation (experimental).

Implements the GRPO objective used by DeepSeek-R1-style training:

1. sample a group of G completions for the same prompt;
2. score each with a programmatic reward (e.g. verifier correctness);
3. normalise rewards inside the group to obtain advantages;
4. maximise the clipped importance-ratio objective with an optional KL term.

This module is a tested building block, not evidence of a trained reasoning
model.  See ``scripts/train_grpo.py`` for the experimental loop.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    if rewards.numel() % group_size != 0:
        raise ValueError("rewards must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / (std + eps)).view(-1)


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    clip_eps: float = 0.2,
    mask: Optional[torch.Tensor] = None,
    ref_logprobs: Optional[torch.Tensor] = None,
    kl_coef: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute the GRPO surrogate loss.

    ``logprobs`` and ``old_logprobs`` are token log-probabilities of shape
    ``[batch, seq]``; ``rewards`` has shape ``[batch]``.  ``mask`` marks valid
    completion tokens.  Returns the scalar loss and diagnostics.
    """
    if logprobs.shape != old_logprobs.shape:
        raise ValueError("logprobs and old_logprobs must have the same shape")
    if rewards.dim() != 1 or rewards.shape[0] != logprobs.shape[0]:
        raise ValueError("rewards must have shape [batch]")
    advantages = group_advantages(rewards.detach(), group_size).detach().unsqueeze(-1)
    ratio = torch.exp(logprobs - old_logprobs.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    if mask is None:
        mask = torch.ones_like(logprobs)
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    policy_loss = -(surrogate * mask).sum() / denom

    kl = torch.zeros((), device=logprobs.device)
    if ref_logprobs is not None and kl_coef:
        # k3 estimator: exp(ref - policy) - (ref - policy) - 1, always >= 0
        diff = ref_logprobs.detach() - logprobs
        kl = (torch.exp(diff) - diff - 1.0)
        kl = (kl * mask).sum() / denom
    loss = policy_loss + kl_coef * kl

    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float()
        clip_fraction = (clip_fraction * mask).sum() / denom
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "clip_fraction": clip_fraction,
        "mean_ratio": ((ratio * mask).sum() / denom).detach(),
        "mean_advantage": advantages.mean().detach(),
    }


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-token log-probabilities for completion tokens only.

    The returned tensor has the same shape as ``completion_mask``; positions
    outside the completion are zero and masked out by the loss.
    """
    logits = model(input_ids=input_ids)["logits"].float()
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask[:, 1:].float()
    return gathered * mask
