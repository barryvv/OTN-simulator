"""Group sampling for GRPO.

Generate G completions for a single prompt using the policy's `.generate()`
method, then build the attention and completion masks that the rest of
the pipeline needs.
"""

from __future__ import annotations

import torch


def sample_group(
    model,
    tokenizer,
    prompt_text: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device | str,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample G completions for one prompt.

    Args:
        model: causal LM to sample from (in eval mode).
        tokenizer: HuggingFace tokenizer matching the model.
        prompt_text: single prompt string already formatted with chat template.
        group_size: number of completions to draw for this prompt.
        max_new_tokens: cap on generation length.
        temperature: sampling temperature; higher => more diverse completions.
        device: where to run generation.

    Returns:
        completions_text: list of G generated text strings (completion-only).
        input_ids: [G, T] full prompt+completion token IDs.
        attention_mask: [G, T] 1 for real tokens, 0 for padding.
        completion_mask: [G, T] 1 over completion tokens only.
    """
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
    )
    prompt_ids = enc.input_ids.to(device)               # [1, P]
    prompt_attn = enc.attention_mask.to(device)         # [1, P]
    prompt_len = prompt_ids.size(1)

    outputs = model.generate(
        prompt_ids.expand(group_size, -1),
        attention_mask=prompt_attn.expand(group_size, -1),
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )                                                   # [G, P+C]

    full = outputs                                      # [G, T]
    attention_mask = (full != tokenizer.pad_token_id).long()
    completion_mask = torch.zeros_like(full)
    completion_mask[:, prompt_len:] = attention_mask[:, prompt_len:]

    completions_text = [
        tokenizer.decode(full[i, prompt_len:], skip_special_tokens=True)
        for i in range(group_size)
    ]

    return completions_text, full, attention_mask, completion_mask
