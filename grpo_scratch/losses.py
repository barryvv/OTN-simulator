"""GRPO loss: clipped surrogate + optional KL.

The clipped surrogate is inherited verbatim from PPO. The only thing GRPO
changes is how `advantages` is computed (no value model, group-relative
baseline) — see `grpo_scratch.advantages`.
"""

from __future__ import annotations

from typing import Optional

import torch


def grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    log_probs_ref: Optional[torch.Tensor] = None,
    epsilon: float = 0.2,
    beta: float = 0.0,
    completion_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO loss for one batch.

    Two modes, selected by the shape of the log-prob tensors:

    - **Sequence-level** (1D, [B]): each entry is the *summed* completion
      log-prob. The clipped surrogate is computed once per sequence. Simple,
      but the gradient magnitude scales with completion length (long answers
      dominate) — this is the length bias GRPO is known for.

    - **Token-level** (2D, [B, T] with `completion_mask`): each entry is a
      per-token log-prob. The importance ratio and clipped surrogate are
      computed per token, then averaged over completion tokens within each
      sequence before averaging over the batch. This is what TRL / DAPO do
      and removes the length bias — every completion contributes on the same
      per-token scale regardless of length, and the gradient stays in a sane
      range. **Prefer this mode for real training.**

    Args:
        log_probs_new: [B] or [B, T] log-prob under current policy.
        log_probs_old: same shape, under the sampling policy (detached).
        advantages: [B] group-relative advantages (detached).
        log_probs_ref: same shape as log_probs_new under the reference model
            (detached), or None.
        epsilon: clipping parameter for the importance-sampling ratio.
        beta: KL coefficient (0 to disable, following DAPO).
        completion_mask: [B, T] 1 over completion tokens, 0 elsewhere.
            Required for (and only used in) token-level mode.

    Returns:
        loss: scalar tensor.
        metrics: dict of training diagnostics.
    """
    token_level = log_probs_new.dim() == 2
    ratio = torch.exp(log_probs_new - log_probs_old)             # [B] or [B,T]
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    if token_level:
        if completion_mask is None:
            raise ValueError("token-level grpo_loss requires completion_mask")
        mask = completion_mask.to(ratio.dtype)                   # [B, T]
        adv = advantages.unsqueeze(1)                            # [B, 1]
        surrogate = torch.min(ratio * adv, clipped_ratio * adv)  # [B, T]
        tok_per_seq = mask.sum(dim=1).clamp(min=1.0)             # [B]
        # Per-sequence token-mean, then batch-mean (classic GRPO objective).
        per_seq = (surrogate * mask).sum(dim=1) / tok_per_seq    # [B]
        policy_loss = -per_seq.mean()

        clipped = ((ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)).to(ratio.dtype)
        fraction_clipped = (clipped * mask).sum() / mask.sum().clamp(min=1.0)
        mean_ratio = (ratio * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        policy_loss = -surrogate.mean()
        fraction_clipped = (
            (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
        ).float().mean()
        mean_ratio = ratio.mean()

    metrics = {
        "policy_loss": policy_loss.item(),
        "mean_ratio": mean_ratio.item(),
        "fraction_clipped": fraction_clipped.item(),
        "mean_advantage": advantages.mean().item(),
        "adv_abs_mean": advantages.abs().mean().item(),
    }

    if beta > 0.0 and log_probs_ref is not None:
        if token_level:
            mask = completion_mask.to(log_probs_new.dtype)
            # k3 estimator of KL(pi || pi_ref), per token, masked-averaged.
            diff = log_probs_ref - log_probs_new
            kl_tok = torch.exp(diff) - diff - 1.0
            kl = (kl_tok * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            kl = (log_probs_new - log_probs_ref).mean()
        loss = policy_loss + beta * kl
        metrics["kl"] = kl.item()
    else:
        loss = policy_loss
        metrics["kl"] = 0.0

    return loss, metrics
