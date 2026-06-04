"""Group-relative advantage estimation for GRPO.

This is the single line of math that distinguishes GRPO from PPO. PPO uses
a learned value head to compute advantages; GRPO uses the per-group mean
and standard deviation of the rewards as the baseline. No value model.
"""

from __future__ import annotations

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
    scale_by_std: bool = True,
    min_std: float = 1e-4,
) -> torch.Tensor:
    """Group-relative advantage baseline for GRPO.

    Args:
        rewards: [B] flat tensor where B = num_prompts * group_size.
                 Groups are contiguous; e.g. with G=4, rewards[0..3] are
                 4 completions for prompt 0, rewards[4..7] are 4 for
                 prompt 1, etc.
        group_size: int G
        eps: numerical stabilizer added to the std before division.
        scale_by_std: if True (default, original GRPO), divide the centered
            reward by the within-group std so advantages are standardized to
            unit variance. If False (Dr. GRPO, Liu et al. 2025), use only the
            mean baseline: advantage = reward - group_mean.

            WHY THIS MATTERS: when a group's completions score almost
            identically (std ~ 1e-3), dividing by that tiny std blows a
            rounding-noise reward gap up to ±1.0 advantages — the optimizer
            then chases pure noise. Removing std scaling makes the advantage
            magnitude proportional to the *real* reward spread, so degenerate
            groups contribute ~0 signal instead of amplified noise.
        min_std: when scale_by_std is True, groups whose std falls below this
            floor are treated as degenerate and their advantages are zeroed
            (rather than divided by ~eps and exploded). No effect when
            scale_by_std is False.

    Returns:
        advantages: [B].
    """
    B = rewards.size(0)
    if B % group_size != 0:
        raise ValueError(
            f"batch size {B} not divisible by group size {group_size}"
        )
    num_prompts = B // group_size

    grouped = rewards.view(num_prompts, group_size)

    mean = grouped.mean(dim=1, keepdim=True)                       # [N, 1]
    centered = grouped - mean                                       # [N, G]

    if not scale_by_std:
        # Dr. GRPO: mean baseline only. Degenerate groups self-zero because
        # `centered` is ~0 when all rewards in the group are equal.
        return centered.reshape(-1)

    std = grouped.std(dim=1, keepdim=True, unbiased=False)         # [N, 1]
    advantages = centered / (std + eps)                            # [N, G]
    # Kill noise amplification: a group with no real reward spread carries no
    # learnable signal, so zero it instead of dividing by ~eps.
    degenerate = (std < min_std)                                   # [N, 1]
    advantages = torch.where(degenerate, torch.zeros_like(advantages), advantages)
    return advantages.reshape(-1)                                  # [B]
