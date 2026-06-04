"""From-scratch GRPO trainer for OTN RCA.

The training step is split into six explicit phases so that the algorithm
is readable end to end:

    Phase 1 SAMPLE         draw G completions for each prompt   (no grad)
    Phase 2 REWARDS/ADV    score completions, standardize per group
    Phase 3 OLD LOG-PROBS  snapshot the sampling policy        (no grad)
    Phase 4 REF LOG-PROBS  optional reference model            (no grad)
    Phase 5 NEW LOG-PROBS  current policy + clipped surrogate  (with grad)
    Phase 6 BACKWARD       gradient clipping + optimizer step
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import torch

from grpo_scratch.advantages import group_relative_advantages
from grpo_scratch.log_probs import compute_log_probs
from grpo_scratch.losses import grpo_loss
from grpo_scratch.sampling import sample_group


@dataclass
class GRPOScratchConfig:
    group_size: int = 4
    max_new_tokens: int = 1024
    temperature: float = 0.7
    learning_rate: float = 5e-6
    # LR schedule. "constant" reproduces the old flat-LR behavior; "cosine"
    # / "linear" warm up over `warmup_steps` then decay to 0 across the run.
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    epsilon: float = 0.2
    beta: float = 0.0
    max_grad_norm: float = 1.0
    # Advantage normalization. True = original GRPO (divide by within-group
    # std). False = Dr. GRPO (mean baseline only) — far more robust when the
    # reward landscape is flat, because it stops tiny std values from
    # amplifying rounding noise into unit-magnitude advantages.
    scale_rewards: bool = False
    log_every: int = 1
    save_every: int = 50
    output_dir: str = "adapters/grpo_scratch"
    reward_weights: Sequence[float] = field(default_factory=lambda: (0.8, 0.2))
    # Memory knobs (mostly relevant for MPS / single-GPU runs).
    gradient_checkpointing: bool = False
    empty_cache_between_phases: bool = False


class FromScratchGRPOTrainer:
    """Raw-PyTorch GRPO trainer that does NOT use trl.GRPOTrainer.

    The policy update loop, log-prob computation, group-relative advantage
    estimation, clipped surrogate loss, and optimizer step are all written
    here in raw PyTorch. HuggingFace Transformers is used only for model
    loading and generation, and PEFT only for LoRA adapter mechanics.
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        reward_funcs: Sequence[Callable],
        config: GRPOScratchConfig,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.reward_funcs = list(reward_funcs)
        self.config = config

        if len(self.config.reward_weights) != len(self.reward_funcs):
            raise ValueError(
                f"reward_weights ({len(self.config.reward_weights)}) "
                f"must match number of reward functions ({len(self.reward_funcs)})"
            )

        self.device = next(self.policy.parameters()).device
        self.optimizer = torch.optim.AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=self.config.learning_rate,
        )
        # Scheduler is built lazily in train() once the step count is known.
        self.scheduler = None

        if self.ref is not None:
            self.ref.eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)

        if self.config.gradient_checkpointing:
            # PEFT + gradient checkpointing requires this to make sure the
            # gradient reaches the LoRA adapters through the frozen base.
            if hasattr(self.policy, "enable_input_require_grads"):
                self.policy.enable_input_require_grads()
            self.policy.gradient_checkpointing_enable()
            # Caching is incompatible with checkpointing.
            if hasattr(self.policy, "config"):
                self.policy.config.use_cache = False

    # ─────────────────────────────────────────────────────────
    # Reward aggregation
    # ─────────────────────────────────────────────────────────
    def compute_rewards(
        self,
        completions: list[str],
        ground_truth_kwargs: dict,
    ) -> torch.Tensor:
        """Call each reward function and combine with the configured weights."""
        per_func = []
        for func in self.reward_funcs:
            r = func(completions, **ground_truth_kwargs)
            per_func.append(
                torch.tensor(r, dtype=torch.float32, device=self.device)
            )
        stacked = torch.stack(per_func, dim=0)          # [num_funcs, B]
        weights = torch.tensor(
            self.config.reward_weights, dtype=torch.float32, device=self.device,
        )
        return (stacked * weights.unsqueeze(1)).sum(dim=0)    # [B]

    # ─────────────────────────────────────────────────────────
    # Per-step training
    # ─────────────────────────────────────────────────────────
    def train_step(self, batch: dict) -> tuple[float, dict]:
        """Run one GRPO update on a batch of N prompts.

        batch: dict with keys
            "prompt": list of N prompt strings
            "ground_truth_*": ground-truth fields aligned with prompts
        """
        N = len(batch["prompt"])
        G = self.config.group_size

        # ─── Phase 1: SAMPLE ───────────────────────────────────
        all_input_ids: list[torch.Tensor] = []
        all_attn: list[torch.Tensor] = []
        all_comp: list[torch.Tensor] = []
        all_completions_text: list[str] = []

        self.policy.eval()
        # Sampling needs a KV cache; checkpointing disables it, so flip
        # use_cache on/off around generation.
        had_cache = getattr(self.policy.config, "use_cache", True)
        if self.config.gradient_checkpointing:
            self.policy.config.use_cache = True
        with torch.no_grad():
            for prompt in batch["prompt"]:
                text, ids, attn, comp = sample_group(
                    self.policy,
                    self.tokenizer,
                    prompt,
                    G,
                    self.config.max_new_tokens,
                    self.config.temperature,
                    self.device,
                )
                all_completions_text.extend(text)
                all_input_ids.append(ids)
                all_attn.append(attn)
                all_comp.append(comp)
        if self.config.gradient_checkpointing:
            self.policy.config.use_cache = had_cache

        # Pad across prompts to the same length so we can stack.
        input_ids, attention_mask, completion_mask = _pad_and_stack(
            all_input_ids, all_attn, all_comp, self.tokenizer.pad_token_id,
        )
        self._maybe_empty_cache()

        # ─── Phase 2: REWARDS + ADVANTAGES ─────────────────────
        gt_kwargs = self._expand_ground_truth(batch, G)
        rewards = self.compute_rewards(all_completions_text, gt_kwargs)
        advantages = group_relative_advantages(
            rewards, G, scale_by_std=self.config.scale_rewards,
        ).detach()

        # Completion mask aligned with the per-token log-probs (which are
        # shifted left by one, so we drop the first mask column to match).
        token_mask = completion_mask[:, 1:]

        # ─── Phase 3: OLD LOG-PROBS ────────────────────────────
        # The "old" policy is the policy at sampling time. Since we just
        # sampled and haven't updated, the current policy IS pi_old. We
        # snapshot its per-token log-probs and detach.
        with torch.no_grad():
            _, tok_old = compute_log_probs(
                self.policy, input_ids, attention_mask, completion_mask,
            )
        self._maybe_empty_cache()

        # ─── Phase 4: REF LOG-PROBS (optional) ─────────────────
        tok_ref = None
        if self.config.beta > 0.0 and self.ref is not None:
            with torch.no_grad():
                _, tok_ref = compute_log_probs(
                    self.ref, input_ids, attention_mask, completion_mask,
                )
            self._maybe_empty_cache()

        # ─── Phase 5: NEW LOG-PROBS + LOSS ─────────────────────
        # Switch to train() so that gradient checkpointing actually
        # activates (HF transformers guards it with `self.training`). This
        # is only safe when lora_dropout=0, which the entrypoint enforces;
        # otherwise old vs new log-probs would diverge purely from dropout
        # noise. With dropout off, train/eval are mathematically identical
        # and the importance ratio still reflects only real policy change.
        self.policy.train()
        _, tok_new = compute_log_probs(
            self.policy, input_ids, attention_mask, completion_mask,
        )
        self.policy.eval()

        # Token-level clipped surrogate (per-token mean) removes the length
        # bias that summed-sequence log-probs introduce.
        loss, metrics = grpo_loss(
            log_probs_new=tok_new,
            log_probs_old=tok_old.detach(),
            advantages=advantages,
            log_probs_ref=tok_ref.detach() if tok_ref is not None else None,
            epsilon=self.config.epsilon,
            beta=self.config.beta,
            completion_mask=token_mask,
        )

        # ─── Phase 6: BACKWARD + STEP ──────────────────────────
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in self.policy.parameters() if p.requires_grad),
            max_norm=self.config.max_grad_norm,
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._maybe_empty_cache()

        metrics["reward_mean"] = rewards.mean().item()
        metrics["reward_std"] = rewards.std().item() if rewards.numel() > 1 else 0.0
        metrics["grad_norm"] = float(grad_norm)
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return loss.item(), metrics

    def _maybe_empty_cache(self) -> None:
        """Release cached buffers between phases when configured to do so.
        Useful on MPS where the allocator does not free aggressively."""
        if not self.config.empty_cache_between_phases:
            return
        dev = str(self.device)
        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif dev.startswith("mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _expand_ground_truth(self, batch: dict, group_size: int) -> dict:
        """Replicate each ground-truth field G times so it aligns with
        the flattened [N*G] completions list."""
        expanded: dict = {}
        for k, v in batch.items():
            if k.startswith("ground_truth"):
                expanded[k] = [item for item in v for _ in range(group_size)]
        return expanded

    # ─────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────
    def train(self, dataloader: Iterable[dict], num_steps: int) -> None:
        self._build_scheduler(num_steps)
        step = 0
        epoch = 0
        # Cycle the dataloader so num_steps can exceed the dataset size:
        # with 147 prompts and batch_size=1, a single pass is only 147
        # steps, so anything larger needs multiple epochs.
        while step < num_steps:
            epoch += 1
            for batch in dataloader:
                if step >= num_steps:
                    break
                loss, metrics = self.train_step(batch)
                self._log_and_save(step, loss, metrics)
                step += 1
        self.save(Path(self.config.output_dir) / "final")

    def _log_and_save(self, step: int, loss: float, metrics: dict) -> None:
        if step % self.config.log_every == 0:
            print(
                f"step {step}: loss={loss:.4f}  "
                f"reward_mean={metrics['reward_mean']:.4f}  "
                f"reward_std={metrics['reward_std']:.4f}  "
                f"adv_abs={metrics['adv_abs_mean']:.4f}  "
                f"frac_clipped={metrics['fraction_clipped']:.3f}  "
                f"mean_ratio={metrics['mean_ratio']:.3f}  "
                f"kl={metrics['kl']:.4f}  "
                f"grad_norm={metrics['grad_norm']:.3f}  "
                f"lr={metrics['lr']:.2e}",
                flush=True,
            )
        if step > 0 and step % self.config.save_every == 0:
            self.save(Path(self.config.output_dir) / f"step-{step}")

    def _build_scheduler(self, num_steps: int) -> None:
        """Construct the LR scheduler now that the total step count is known.

        "constant" leaves the optimizer LR flat (old behavior). "cosine" and
        "linear" warm up linearly over `warmup_steps` then decay to 0 across
        the remaining steps. transformers.get_scheduler is used so the warmup
        + decay curve matches the TRL trainer for apples-to-apples runs."""
        sched_type = self.config.lr_scheduler_type
        if sched_type == "constant" and self.config.warmup_steps == 0:
            self.scheduler = None
            return
        from transformers import get_scheduler

        self.scheduler = get_scheduler(
            name=sched_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_steps,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[saved] {path}", flush=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _pad_and_stack(
    input_ids_list: list[torch.Tensor],
    attn_list: list[torch.Tensor],
    comp_list: list[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad each [G_i, T_i] tensor to the global max T, then concatenate."""
    max_len = max(t.size(1) for t in input_ids_list)

    def _pad(t: torch.Tensor, fill: int) -> torch.Tensor:
        pad_amount = max_len - t.size(1)
        if pad_amount == 0:
            return t
        pad = torch.full(
            (t.size(0), pad_amount), fill, dtype=t.dtype, device=t.device,
        )
        return torch.cat([t, pad], dim=1)

    ids = torch.cat([_pad(t, pad_token_id) for t in input_ids_list], dim=0)
    attn = torch.cat([_pad(t, 0) for t in attn_list], dim=0)
    comp = torch.cat([_pad(t, 0) for t in comp_list], dim=0)
    return ids, attn, comp
