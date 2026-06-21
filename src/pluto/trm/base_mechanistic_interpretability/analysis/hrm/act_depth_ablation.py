#!/usr/bin/env python3
"""ACT-depth ablations for full TRM latent recursion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, rollout_act_steps, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig, mod_add_dataset_config
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch
from pluto.trm.models.losses import ACTLossHead


def summarize_act_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "best_exact_max_steps": int(max(rows, key=lambda r: r["exact_accuracy"])["max_steps"]) if rows else 0,
        "best_ce_max_steps": int(min(rows, key=lambda r: r["mean_ce"])["max_steps"]) if rows else 0,
        "n_depths": len(rows),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg = load_model_for_analysis(args.checkpoint, args.model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("ACT-depth ablation requires TRM checkpoint")
    ds_cfg = cfg if isinstance(cfg, ModAddGrokkingConfig) else mod_add_dataset_config(cfg)
    batch = {k: v.to(device)[: args.batch_size] for k, v in all_pairs_batch(ds_cfg, test_only=True).items()}

    rows: List[Dict[str, Any]] = []
    for max_steps in args.depths:
        steps = rollout_act_steps(model, batch, max_steps=max_steps)
        final = steps[-1]
        rows.append(
            {
                "max_steps": int(max_steps),
                "mean_ce": float(final["ce"].mean().item()),
                "exact_accuracy": float(final["exact"].float().mean().item()),
                "n_steps_collected": len(steps),
            }
        )

    results = {"checkpoint": args.checkpoint, "rows": rows, "summary": summarize_act_rows(rows)}
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "act_depth_ablation.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/hrm")
    p.add_argument("--model-type", default="trm_full", choices=["trm_minimal", "trm_full"])
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run(args)
    print(r["summary"])


if __name__ == "__main__":
    main()
