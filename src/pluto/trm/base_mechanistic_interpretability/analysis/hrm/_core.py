"""Shared HRM classifiers and ACT rollout helpers."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import rollout_act_steps
from pluto.trm.models.losses import ACTLossHead


def fixed_point_violation_rate(steps: List[Dict[str, torch.Tensor]]) -> float:
    exact = torch.stack([s["exact"] for s in steps], dim=1).float()
    violations = []
    for b in range(exact.shape[0]):
        solved_at = torch.where(exact[b] > 0.5)[0]
        if len(solved_at) == 0:
            continue
        first = int(solved_at[0].item())
        later = exact[b, first + 1 :]
        if later.numel() == 0:
            continue
        violations.append((later < 0.5).any().float().item())
    return float(sum(violations) / max(1, len(violations)))


def classify_reasoning_mode(step_ce: torch.Tensor, step_exact: torch.Tensor) -> str:
    ce = step_ce.cpu().numpy()
    ex = step_exact.cpu().numpy()
    if ex[0] > 0.5:
        return "trivial_success"
    if ex[-1] < 0.5 and ce.max() - ce.min() < 0.05:
        return "trivial_failure"
    if ex[-1] > 0.5:
        plateau = int((ce > ce.min() * 1.05).sum())
        if plateau >= max(1, len(ce) // 2):
            return "nontrivial_success"
    if ex[-1] < 0.5 and ce.std() < 0.02:
        return "nontrivial_failure"
    return "nontrivial_failure"


def latent_pca_coords(steps: List[Dict[str, torch.Tensor]], sample_idx: int = 0) -> List[List[float]]:
    zs = torch.stack([s["z_H"][sample_idx, 2].float() for s in steps], dim=0)
    zs = zs - zs.mean(0, keepdim=True)
    if zs.shape[0] < 2:
        return [[0.0, 0.0]] * len(steps)
    _, _, vh = torch.linalg.svd(zs, full_matrices=False)
    proj = zs @ vh[:2].T
    return proj.cpu().tolist()


def grokking_stats(steps: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    ce = torch.stack([s["ce"] for s in steps], dim=1).detach()
    ex = torch.stack([s["exact"] for s in steps], dim=1).float().detach()
    mean_ce = ce.mean(0).tolist()
    mean_ex = ex.mean(0).tolist()
    return {
        "mean_field_ce_by_act_step": mean_ce,
        "mean_field_exact_by_act_step": mean_ex,
        "grokking_plateau_act_steps": int(sum(1 for x in mean_ce if x > min(mean_ce) * 1.05)),
        "critical_act_grokking_step": critical_act_grokking_step(mean_ce, mean_ex),
    }


def critical_act_grokking_step(mean_ce: List[float], mean_ex: List[float]) -> int:
    """Ren §4: first ACT step where batch mean exact jumps to ≥0.95 after a flat CE plateau."""
    if not mean_ex:
        return -1
    plateau_end = 0
    for i, acc in enumerate(mean_ex):
        if acc >= 0.95:
            return i
        if i > 0 and mean_ce[i] <= mean_ce[0] * 1.1:
            plateau_end = i
    return plateau_end


def depth_sensitivity(act_depth_rows: List[Dict[str, Any]]) -> float:
    """Accuracy drop from shallowest to deepest ACT cap (Ren fixed-point stress test)."""
    if len(act_depth_rows) < 2:
        return 0.0
    shallow = float(act_depth_rows[0].get("exact_accuracy", 0.0))
    deep = float(act_depth_rows[-1].get("exact_accuracy", 0.0))
    return max(0.0, shallow - deep)


def ren_mechanism_verdict(
    *,
    model_type: str,
    nanda_causes: List[str],
    fixed_point_violation_rate: float,
    depth_acc_drop: float,
    critical_act_step: int,
    halt_max_steps: int,
) -> Dict[str, Any]:
    """Map probes to spec hypotheses H1–H6 (arXiv:2601.10679 + Nanda cleanup)."""
    hypotheses: List[str] = []
    ren_applies = model_type == "trm_full"
    ren_null_minimal = model_type == "trm_minimal" and fixed_point_violation_rate < 0.05 and depth_acc_drop < 0.05

    if "incomplete_grokking_by_20k" in nanda_causes:
        hypotheses.append("H1_undertraining")
    if "cleanup_phase_fve_collapse" in nanda_causes or "post_grokking_fve_decay" in nanda_causes:
        hypotheses.append("H2_nanda_cleanup")
    if fixed_point_violation_rate >= 0.1:
        hypotheses.append("H3_fixed_point_violation")
    if critical_act_step >= 2 and halt_max_steps > 1:
        hypotheses.append("H4_act_step_guessing")
    if depth_acc_drop >= 0.15 and halt_max_steps > 1:
        hypotheses.append("H4_depth_guessing")
    if "stable_circuit" in nanda_causes and not hypotheses:
        hypotheses.append("H_stable_circuit")

    if ren_null_minimal:
        primary = "nanda_cleanup" if "H2_nanda_cleanup" in hypotheses else "stable_circuit"
        ren_applies = False
    elif ren_applies and (fixed_point_violation_rate >= 0.1 or depth_acc_drop >= 0.15):
        primary = "ren_guessing_fixed_point"
    elif "H2_nanda_cleanup" in hypotheses:
        primary = "nanda_cleanup"
    elif "H1_undertraining" in hypotheses:
        primary = "undertraining"
    else:
        primary = "stable_circuit"

    return {
        "primary_mechanism": primary,
        "hypotheses": hypotheses,
        "ren_applies": ren_applies,
        "ren_null_on_minimal": ren_null_minimal,
        "fixed_point_violation_rate": fixed_point_violation_rate,
        "depth_accuracy_drop": depth_acc_drop,
        "critical_act_grokking_step": critical_act_step,
    }


def run_hrm_probes(
    model: ACTLossHead,
    batch: Dict[str, torch.Tensor],
    *,
    max_probe_samples: int = 32,
) -> Dict[str, Any]:
    n = min(max_probe_samples, batch["inputs"].shape[0])
    sub = {k: v[:n] for k, v in batch.items()}
    steps = rollout_act_steps(model, sub)
    ex = torch.stack([s["exact"] for s in steps], dim=1).float()
    modes = [
        classify_reasoning_mode(torch.stack([s["ce"][b] for s in steps]), ex[b]) for b in range(n)
    ]
    mode_counts = {m: modes.count(m) for m in set(modes)}

    fp_rates = []
    for b in range(min(32, n)):
        one = {k: v[b : b + 1] for k, v in sub.items()}
        probe = rollout_act_steps(model, one)
        fp_rates.append(fixed_point_violation_rate(probe))

    stats = grokking_stats(steps)
    return {
        "paper": "2601.10679",
        "reasoning_mode_counts": mode_counts,
        "fixed_point_violation_rate": float(sum(fp_rates) / max(1, len(fp_rates))),
        "latent_pca_sample0": latent_pca_coords(steps, sample_idx=0),
        "n_act_steps_collected": len(steps),
        **stats,
    }
