"""From-scratch GRPO implementation in raw PyTorch.

This module implements Group Relative Policy Optimization (GRPO) without
relying on `trl.GRPOTrainer`. It exists alongside `train_grpo_gemma3b.py`
(the TRL-based production trainer).

Modules
-------
- sampling.py        Group sampling: G completions per prompt
- log_probs.py       Per-token log-prob computation with masking
- advantages.py      Group-relative advantage standardization
- losses.py          Clipped surrogate loss + optional KL penalty
- rewards.py         OTN-specific reward functions
- trainer.py         Training loop with explicit phase-by-phase structure
"""

from grpo_scratch.advantages import group_relative_advantages
from grpo_scratch.log_probs import compute_log_probs
from grpo_scratch.losses import grpo_loss
from grpo_scratch.rewards import otn_alarm_reward, otn_format_reward
from grpo_scratch.sampling import sample_group
from grpo_scratch.trainer import FromScratchGRPOTrainer, GRPOScratchConfig

__all__ = [
    "FromScratchGRPOTrainer",
    "GRPOScratchConfig",
    "compute_log_probs",
    "group_relative_advantages",
    "grpo_loss",
    "otn_alarm_reward",
    "otn_format_reward",
    "sample_group",
]
