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
) -> torch.Tensor:
    """Standardize rewards within each group.

    Args:
        rewards: [B] flat tensor where B = num_prompts * group_size.
                 Groups are contiguous; e.g. with G=4, rewards[0..3] are
                 4 completions for prompt 0, rewards[4..7] are 4 for
                 prompt 1, etc.
        group_size: int G
        eps: numerical stabilizer added to the std before division.

    Returns:
        advantages: [B] standardized within group.
    """
    B = rewards.size(0)
    if B % group_size != 0:
        raise ValueError(
            f"batch size {B} not divisible by group size {group_size}"
        )
    num_prompts = B // group_size

    grouped = rewards.view(num_prompts, group_size)

    mean = grouped.mean(dim=1, keepdim=True)                       # [N, 1]
    std = grouped.std(dim=1, keepdim=True, unbiased=False) + eps   # match TRL

    advantages = (grouped - mean) / std                            # [N, G]
    return advantages.view(-1)                                     # [B]
