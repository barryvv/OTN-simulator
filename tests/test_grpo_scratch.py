"""Unit tests for from-scratch GRPO components.

These tests prove the core math is correct without requiring a real
language model: a tiny stub LM is enough to exercise the log-prob
shift, and the advantage / loss functions are pure tensor operations.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

# Allow running `pytest` from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grpo_scratch.advantages import group_relative_advantages
from grpo_scratch.log_probs import compute_log_probs
from grpo_scratch.losses import grpo_loss


# ─────────────────────────────────────────────────────────────
# Advantages
# ─────────────────────────────────────────────────────────────

def test_group_relative_advantages_mean_zero_std_one():
    """After group-relative standardization, each group should have
    mean ~= 0 and std ~= 1 (using the biased std convention)."""
    G = 4
    rewards = torch.tensor(
        [1.0, 2.0, 3.0, 4.0,
         10.0, 20.0, 30.0, 40.0]
    )
    adv = group_relative_advantages(rewards, group_size=G)
    grouped = adv.view(2, G)

    assert torch.allclose(grouped.mean(dim=1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(
        grouped.std(dim=1, unbiased=False), torch.ones(2), atol=1e-3,
    )


def test_group_relative_advantages_constant_group_does_not_blow_up():
    """A group with zero variance should produce finite (near-zero)
    advantages thanks to the eps stabilizer."""
    rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
    adv = group_relative_advantages(rewards, group_size=4)
    assert torch.isfinite(adv).all()
    assert adv.abs().max().item() < 1e-3


def test_group_relative_advantages_rejects_bad_batch():
    with pytest.raises(ValueError):
        group_relative_advantages(torch.zeros(5), group_size=4)


# ─────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────

def test_grpo_loss_clipping_activates_for_large_ratios():
    """When the ratio exceeds 1+epsilon, the clipped surrogate should
    activate and limit the loss contribution."""
    log_probs_new = torch.tensor([0.0, 1.5])   # sample 1 has a big ratio
    log_probs_old = torch.tensor([0.0, 0.0])
    advantages = torch.tensor([1.0, 1.0])

    loss, metrics = grpo_loss(
        log_probs_new=log_probs_new,
        log_probs_old=log_probs_old,
        advantages=advantages,
        epsilon=0.2,
        beta=0.0,
    )

    # Sample 0: ratio = 1.0 (no clip), surrogate = 1.0 * 1 = 1.0
    # Sample 1: ratio = exp(1.5) ~= 4.48; clipped to 1.2; surrogate = min(4.48, 1.2) = 1.2
    # Mean surrogate = (1.0 + 1.2) / 2 = 1.1  =>  loss = -1.1
    assert abs(loss.item() - (-1.1)) < 1e-3
    assert metrics["fraction_clipped"] == 0.5


def test_grpo_loss_kl_only_applied_when_beta_positive():
    log_probs_new = torch.tensor([0.0, 0.0])
    log_probs_old = torch.tensor([0.0, 0.0])
    log_probs_ref = torch.tensor([1.0, 1.0])  # would give KL = -1 if applied
    advantages = torch.tensor([1.0, 1.0])

    loss_off, metrics_off = grpo_loss(
        log_probs_new, log_probs_old, advantages,
        log_probs_ref=log_probs_ref, epsilon=0.2, beta=0.0,
    )
    loss_on, metrics_on = grpo_loss(
        log_probs_new, log_probs_old, advantages,
        log_probs_ref=log_probs_ref, epsilon=0.2, beta=0.5,
    )

    assert metrics_off["kl"] == 0.0
    # KL = mean(0 - 1) = -1, so loss_on = loss_off + 0.5 * (-1) = loss_off - 0.5
    assert abs((loss_on - loss_off).item() - (-0.5)) < 1e-5


# ─────────────────────────────────────────────────────────────
# Log-probs
# ─────────────────────────────────────────────────────────────

class _ConstLogitsLM(nn.Module):
    """Tiny stand-in for a HF causal LM that returns fixed logits."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        # Register the logits as a buffer so we get .to(device) for free.
        self.register_buffer("_logits", logits)

    def forward(self, input_ids, attention_mask=None):
        # Mimic the HF output's .logits attribute.
        class _Out:
            pass
        out = _Out()
        # Expand to match the batch dimension of input_ids.
        B = input_ids.size(0)
        out.logits = self._logits.unsqueeze(0).expand(B, -1, -1)
        return out


def test_compute_log_probs_shift_and_mask():
    """Verify the off-by-one shift and the completion mask: only
    log-probs of *completion* tokens should be summed.

    Setup:
      input_ids = [p, p, c, c] = [9, 9, 0, 1]
      completion_mask = [0, 0, 1, 1]
        - positions 2, 3 are completion tokens
        - positions 0, 1 are prompt
      Inside compute_log_probs the mask is shifted: completion_mask[:, 1:]
      = [0, 1, 1]. That keeps shifted indices 1 and 2 (predicting the two
      completion tokens, ids 0 and 1) and drops shifted index 0
      (predicting the prompt-continuation token at original position 1).
    """
    import torch.nn.functional as F

    V = 5
    T = 4
    # Constant logits where id 0 is the "winner" (log-prob ~= 0) and id 1
    # is a "loser" (log-prob ~= -10).
    logits = torch.zeros(T, V)
    logits[:, 0] = 10.0
    model = _ConstLogitsLM(logits)

    # Use a distinct prompt token (id 9 % V = 4) so that if the prompt
    # log-prob accidentally leaked through, the test would catch it.
    input_ids = torch.tensor([[4, 4, 0, 1]])
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.tensor([[0, 0, 1, 1]])

    log_probs_sum, masked_token_log_probs = compute_log_probs(
        model, input_ids, attention_mask, completion_mask,
    )

    # Shape sanity check.
    assert masked_token_log_probs.shape == (1, T - 1)

    # Shifted index 0 (predicting the prompt-continuation token) must be
    # masked to zero even though the model assigns it a low log-prob.
    assert masked_token_log_probs[0, 0].item() == 0.0

    # Expected sum is log P(id=0) + log P(id=1) under softmax([10,0,0,0,0]).
    lp = F.log_softmax(logits[0], dim=-1)
    expected = lp[0].item() + lp[1].item()
    assert abs(log_probs_sum.item() - expected) < 1e-4


if __name__ == "__main__":
    test_group_relative_advantages_mean_zero_std_one()
    test_group_relative_advantages_constant_group_does_not_blow_up()
    test_grpo_loss_clipping_activates_for_large_ratios()
    test_grpo_loss_kl_only_applied_when_beta_positive()
    test_compute_log_probs_shift_and_mask()
    print("[ok] all tests passed.")
