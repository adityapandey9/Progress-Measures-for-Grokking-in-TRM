#!/usr/bin/env python3
"""All-frequency retain/ablate grid for Nanda-style causal evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import (
    ensure_dir,
    eval_all_pairs_logits_from_checkpoint,
    save_json,
)
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    calculate_excluded_loss,
    calculate_trig_loss,
    fourier_basis,
    logits_grid,
)


def summarize_frequency_rows(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    return {
        "n_frequencies": len(rows),
        "top_ablate_frequencies": [
            int(r["frequency"]) for r in sorted(rows, key=lambda x: x["ablate_loss_test"], reverse=True)
        ],
        "top_retain_frequencies": [
            int(r["frequency"]) for r in sorted(rows, key=lambda x: x["retain_loss_test"])
        ],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logits, cfg, _model = eval_all_pairs_logits_from_checkpoint(args.checkpoint, args.model_type, device)
    ds = ModAddFullDataset(cfg)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)
    basis = fourier_basis(cfg.p, device)

    rows: List[Dict[str, float]] = []
    for k in range(1, cfg.p // 2 + 1):
        retain_train = calculate_trig_loss(grid, [k], labels, train_m, test_m, basis, mode="train")
        retain_test = calculate_trig_loss(grid, [k], labels, train_m, test_m, basis, mode="test")
        ablate_train, _ = calculate_excluded_loss(grid, [k], labels, train_m, test_m, basis, mode="train")
        ablate_test, _ = calculate_excluded_loss(grid, [k], labels, train_m, test_m, basis, mode="test")
        rows.append(
            {
                "frequency": float(k),
                "retain_loss_train": retain_train,
                "retain_loss_test": retain_test,
                "ablate_loss_train": ablate_train,
                "ablate_loss_test": ablate_test,
            }
        )

    results = {"checkpoint": args.checkpoint, "model_type": args.model_type, "rows": rows}
    results["summary"] = summarize_frequency_rows(rows)
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "all_frequency_ablations.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/all_frequency_ablations")
    p.add_argument("--model-type", default="trm_full", choices=["nanda", "trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run(args)
    print(f"all-frequency ablations: {len(r['rows'])} frequencies")


if __name__ == "__main__":
    main()
