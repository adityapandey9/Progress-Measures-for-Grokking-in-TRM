"""TRM ACT shell with Nanda-faithful one-layer encoder (no z_H/z_L recursion).

Bypasses the recursive z_H/z_L loop and runs a single causal-attention + ReLU MLP
block like Nanda et al. (arXiv:2301.05217), while keeping the ACT training wrapper
used by ``train_grokking.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig, nanda_config_from_modadd
from pluto.trm.base_mechanistic_interpretability.models.nanda_one_layer import NandaFaithfulTransformer
from pluto.trm.models.layers import CastedLinear
from pluto.trm.models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1Config,
)


@dataclass
class TrmNandaBypassInnerCarry:
    """Stateless carry (no z_H/z_L latents)."""

    pass


class TrmNandaBypassInner(nn.Module):
    """Nanda-faithful encoder inside the TRM ACT interface."""

    def __init__(self, modadd_cfg: ModAddGrokkingConfig, act_cfg: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.modadd_cfg = modadd_cfg
        self.config = act_cfg
        self.nanda = NandaFaithfulTransformer(nanda_config_from_modadd(modadd_cfg))
        self.q_head = CastedLinear(act_cfg.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore[union-attr]

    @property
    def embed_tokens(self):
        return self.nanda.embed_tokens

    @property
    def lm_head(self):
        return self.nanda.lm_head

    def empty_carry(self, batch_size: int, device: torch.device | str) -> TrmNandaBypassInnerCarry:
        del batch_size, device
        return TrmNandaBypassInnerCarry()

    def reset_carry(
        self, reset_flag: torch.Tensor, carry: TrmNandaBypassInnerCarry
    ) -> TrmNandaBypassInnerCarry:
        del reset_flag
        return carry

    def forward(
        self, carry: TrmNandaBypassInnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TrmNandaBypassInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        del carry
        hidden = self.nanda.encode(batch["inputs"])
        logits = hidden @ self.nanda.W_U
        q_logits = self.q_head(hidden[:, 0]).to(torch.float32)
        return TrmNandaBypassInnerCarry(), logits, (q_logits[..., 0], q_logits[..., 1])


class TrmNandaBypassACTV1(nn.Module):
    """ACTV1-compatible wrapper over ``TrmNandaBypassInner``."""

    def __init__(self, modadd_cfg: ModAddGrokkingConfig, *, batch_size: int) -> None:
        super().__init__()
        self.modadd_cfg = modadd_cfg
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**modadd_cfg.to_model_dict(batch_size=batch_size))
        self.inner = TrmNandaBypassInner(modadd_cfg, self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_ACTV1Carry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )
        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)
                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )
        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
