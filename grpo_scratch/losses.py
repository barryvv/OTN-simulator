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
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO loss for one batch.

    Args:
        log_probs_new: [B] log-prob of completion under current policy.
        log_probs_old: [B] log-prob under the sampling policy (detached).
        advantages: [B] group-relative advantages (detached).
        log_probs_ref: [B] log-prob under reference model (detached), or None.
        epsilon: clipping parameter for the importance-sampling ratio.
        beta: KL coefficient (0 to disable, following DAPO).

    Returns:
        loss: scalar tensor.
        metrics: dict of training diagnostics.
    """
    ratio = torch.exp(log_probs_new - log_probs_old)              # [B]
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    surrogate_a = ratio * advantages                              # [B]
    surrogate_b = clipped_ratio * advantages                      # [B]
    surrogate = torch.min(surrogate_a, surrogate_b)               # [B]
    policy_loss = -surrogate.mean()

    fraction_clipped = (
        (ratio < 1.0 - epsilon) | (ratio > 1.0 + epsilon)
    ).float().mean()

    metrics = {
        "policy_loss": policy_loss.item(),
        "mean_ratio": ratio.mean().item(),
        "fraction_clipped": fraction_clipped.item(),
        "mean_advantage": advantages.mean().item(),
    }

    if beta > 0.0 and log_probs_ref is not None:
        # Approximation: KL(pi || pi_ref) ~= E[log pi - log pi_ref].
        kl = (log_probs_new - log_probs_ref).mean()
        loss = policy_loss + beta * kl
        metrics["kl"] = kl.item()
    else:
        loss = policy_loss
        metrics["kl"] = 0.0

    return loss, metrics
