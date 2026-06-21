"""Probabilistic TRM (PTRM) inference for BMI mod-add (arXiv:2605.19943).

At test time, run K parallel latent rollouts with Gaussian noise injected before
each deep-recursion (inner) step, then select the rollout with the highest mean
Q-halt logit (best-Q@K). No retraining required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from pluto.trm.models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)


@dataclass(frozen=True)
class PTRMConfig:
    """PTRM hyperparameters (width × depth × noise)."""

    num_rollouts: int = 64
    supervision_steps: int = 16
    noise_sigma: float = 0.2
    select_by: str = "q_halt"  # q_halt | fve_adaptive (requires scorer callback)


@dataclass
class PTRMRolloutResult:
    logits: torch.Tensor
    q_halt_logits: torch.Tensor
    q_score: float
    fve_adaptive: float
    rollout_index: int


def _init_inner_carry(inner: TinyRecursiveReasoningModel_ACTV1_Inner, batch_size: int, device: torch.device):
    carry = inner.empty_carry(batch_size, device)
    reset = torch.ones(batch_size, dtype=torch.bool, device=device)
    return inner.reset_carry(reset, carry)


def _inject_noise(
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
    if sigma <= 0:
        return carry
    eps_h = torch.randn(carry.z_H.shape, device=carry.z_H.device, dtype=carry.z_H.dtype, generator=generator)
    eps_l = torch.randn(carry.z_L.shape, device=carry.z_L.device, dtype=carry.z_L.dtype, generator=generator)
    return TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=carry.z_H + sigma * eps_h,
        z_L=carry.z_L + sigma * eps_l,
    )


@torch.no_grad()
def ptrm_single_rollout(
    inner: TinyRecursiveReasoningModel_ACTV1_Inner,
    batch: Dict[str, torch.Tensor],
    *,
    supervision_steps: int,
    noise_sigma: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One stochastic rollout: D inner steps with noise, persistent latents."""
    device = batch["inputs"].device
    bs = batch["inputs"].shape[0]
    carry = _init_inner_carry(inner, bs, device)
    logits_final: Optional[torch.Tensor] = None
    q_halt_final: Optional[torch.Tensor] = None
    for _ in range(max(1, supervision_steps)):
        carry = _inject_noise(carry, noise_sigma, generator)
        carry, logits, (q_halt, _q_cont) = inner(carry, batch)
        logits_final = logits
        q_halt_final = q_halt
    assert logits_final is not None and q_halt_final is not None
    return logits_final, q_halt_final


def _q_score(q_halt: torch.Tensor) -> float:
    return float(torch.sigmoid(q_halt.float()).mean().item())


@torch.no_grad()
def ptrm_best_rollout(
    inner: TinyRecursiveReasoningModel_ACTV1_Inner,
    batch: Dict[str, torch.Tensor],
    cfg: PTRMConfig,
    *,
    fve_scorer=None,
    base_seed: int = 0,
) -> PTRMRolloutResult:
    """Run K rollouts; return best by mean Q-halt (or adaptive FVE if scorer provided)."""
    best: Optional[PTRMRolloutResult] = None
    for k in range(cfg.num_rollouts):
        gen = torch.Generator(device=batch["inputs"].device)
        gen.manual_seed(base_seed + k * 9973)
        logits, q_halt = ptrm_single_rollout(
            inner,
            batch,
            supervision_steps=cfg.supervision_steps,
            noise_sigma=cfg.noise_sigma,
            generator=gen,
        )
        q_s = _q_score(q_halt)
        fve = float(fve_scorer(logits)) if fve_scorer is not None else 0.0
        if cfg.select_by == "fve_adaptive" and fve_scorer is not None:
            score = fve
        else:
            score = q_s
        candidate = PTRMRolloutResult(
            logits=logits,
            q_halt_logits=q_halt,
            q_score=q_s,
            fve_adaptive=fve,
            rollout_index=k,
        )
        if best is None or score > (best.fve_adaptive if cfg.select_by == "fve_adaptive" else best.q_score):
            best = candidate
    assert best is not None
    return best


@torch.no_grad()
def ptrm_all_pairs_logits(
    model,
    cfg,
    device: torch.device,
    ptrm_cfg: PTRMConfig,
    *,
    base_seed: int = 0,
    fve_scorer=None,
) -> PTRMRolloutResult:
    """PTRM inference on full mod-add batch; returns logits at '=' [p*p, p]."""
    from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch
    from pluto.trm.models.losses import ACTLossHead

    if not isinstance(model, ACTLossHead):
        raise TypeError("PTRM requires ACTLossHead-wrapped TRM")
    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    inner = model.model.inner
    result = ptrm_best_rollout(inner, batch, ptrm_cfg, fve_scorer=fve_scorer, base_seed=base_seed)
    eq_logits = result.logits[:, 2, : cfg.p]
    return PTRMRolloutResult(
        logits=eq_logits,
        q_halt_logits=result.q_halt_logits,
        q_score=result.q_score,
        fve_adaptive=result.fve_adaptive,
        rollout_index=result.rollout_index,
    )
