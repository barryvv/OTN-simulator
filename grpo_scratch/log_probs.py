"""Log-prob computation for policy and reference models.

This is the most error-prone piece of any from-scratch RL implementation
because of the off-by-one shift between logits and targets. The convention:

  logits[:, t, :]  predicts the token at position  t + 1.

So to score the token at position `t+1`, we look at `logits[:, t, :]` and
gather the entry at index `input_ids[:, t+1]`. That is achieved by:

  logits  = outputs.logits[:, :-1, :]   # drop the last logit position
  targets = input_ids[:, 1:]             # shift inputs left by one
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-probs for completions, summed across each sequence.

    Args:
        model: causal LM (policy or reference)
        input_ids: [B, T] full prompt + completion token IDs
        attention_mask: [B, T] 1 for real tokens, 0 for padding
        completion_mask: [B, T] 1 for completion tokens only, 0 elsewhere
                         (i.e., 0 over the prompt and the padding)

    Returns:
        log_probs_sum: [B] sum of token log-probs over completion tokens
        masked_token_log_probs: [B, T-1] per-token log-probs masked to
            completion positions (zero elsewhere). Useful for KL terms
            that need per-token quantities.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]            # [B, T-1, V]
    targets = input_ids[:, 1:]                     # [B, T-1]
    mask = completion_mask[:, 1:].to(logits.dtype)  # [B, T-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)                                  # [B, T-1]

    masked = token_log_probs * mask                # [B, T-1]
    log_probs_sum = masked.sum(dim=-1)             # [B]
    return log_probs_sum, masked
