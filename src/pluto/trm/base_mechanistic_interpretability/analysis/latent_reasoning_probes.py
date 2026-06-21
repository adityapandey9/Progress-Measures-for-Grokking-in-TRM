#!/usr/bin/env python3
"""Reasoning vs guessing probes (arXiv:2601.10679) adapted to TRM ACT latent steps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import (
    ensure_dir,
    load_analysis_bundle,
    rollout_act_steps,
    save_json,
)
from pluto.trm.models.losses import ACTLossHead
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch


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


def classify_mode(step_ce: torch.Tensor, step_exact: torch.Tensor) -> str:
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
    zs = torch.stack([s["z_H"][sample_idx, 2].float() for s in steps], dim=0)  # [T, D]
    zs = zs - zs.mean(0, keepdim=True)
    if zs.shape[0] < 2:
        return [[0.0, 0.0]] * len(steps)
    _, _, vh = torch.linalg.svd(zs, full_matrices=False)
    proj = zs @ vh[:2].T
    return proj.cpu().tolist()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg, _, _ = load_analysis_bundle(args.checkpoint, args.model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("Latent reasoning probes require TRM checkpoints")

    test_batch = {k: v.to(device) for k, v in all_pairs_batch(cfg, test_only=True).items()}
    # Subsample for tractable per-sample analysis
    n = min(args.batch_size, test_batch["inputs"].shape[0])
    batch = {k: v[:n] for k, v in test_batch.items()}

    steps = rollout_act_steps(model, batch)
    ce_mean = torch.stack([s["ce"] for s in steps], dim=1).mean(0).tolist()
    ex = torch.stack([s["exact"] for s in steps], dim=1).float()
    modes = [classify_mode(torch.stack([s["ce"][b] for s in steps]), ex[b]) for b in range(n)]
    mode_counts = {m: modes.count(m) for m in set(modes)}

    fp_rates = []
    for b in range(min(32, n)):
        one = {k: v[b : b + 1] for k, v in batch.items()}
        probe = rollout_act_steps(model, one, max_steps=cfg.halt_max_steps)
        fp_rates.append(fixed_point_violation_rate(probe))

    pca_traj = latent_pca_coords(steps, sample_idx=0)

    results: Dict[str, Any] = {
        "paper": "2601.10679",
        "variant": "base_mechanistic_interpretability",
        "task": "modular_addition",
        "mean_field_ce_by_act_step": ce_mean,
        "reasoning_mode_counts": mode_counts,
        "fixed_point_violation_rate": float(sum(fp_rates) / max(1, len(fp_rates))),
        "grokking_plateau_act_steps": int(sum(1 for x in ce_mean if x > min(ce_mean) * 1.05)),
        "latent_pca_sample0": pca_traj,
        "n_act_steps_collected": len(steps),
    }

    out = ensure_dir(Path(args.output_dir))
    save_json(out / "latent_reasoning_probes.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/reasoning")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--frac-train", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-type", default="trm_full", choices=["trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run(args)
    print(f"fp_violation={r['fixed_point_violation_rate']:.4f} modes={r['reasoning_mode_counts']}")


if __name__ == "__main__":
    main()
