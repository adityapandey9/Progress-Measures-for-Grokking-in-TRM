#!/usr/bin/env python3
"""Recompute trig-FVE (plain vs bias-corrected) on existing checkpoints.

Cheap verification (no retraining): tests whether the plain faithful FVE
under-reports because it cannot represent the constant logit offset that Nanda
restores via bias correction. Run on existing Nanda/TRM final checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis
from pluto.trm.base_mechanistic_interpretability.config import mod_add_dataset_config
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset, all_pairs_batch
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    embedding_fourier_norms,
    fit_trig_logits_fve_bias_corrected,
    fit_trig_logits_fve_faithful,
    identify_key_frequencies,
    identify_key_frequencies_by_excluded,
    logits_grid,
)


@torch.no_grad()
def fve_for_checkpoint(ckpt: str, model_type: str, device: torch.device) -> dict:
    model, cfg = load_model_for_analysis(ckpt, model_type, device)
    ds_cfg = mod_add_dataset_config(cfg)
    batch = {k: v.to(device) for k, v in all_pairs_batch(ds_cfg).items()}
    if model_type == "nanda":
        logits = model(batch["inputs"])[:, 2, : cfg.p]
    else:
        out = model.model(model.model.initial_carry(batch), batch) if hasattr(model, "model") else None
        # fall back to generic forward for non-nanda is out of scope here
        raise SystemExit("This quick check targets nanda checkpoints")
    ds = ModAddFullDataset(ds_cfg)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)
    key_excl = identify_key_frequencies_by_excluded(grid, labels, train_m, test_m, cfg.p, top_k=5)
    plain = fit_trig_logits_fve_faithful(grid, key_excl, cfg.p)["fve_mean"]
    bias = fit_trig_logits_fve_bias_corrected(grid, key_excl, cfg.p)["fve_mean"]
    return {"checkpoint": ckpt, "key_freqs": key_excl, "fve_plain": plain, "fve_bias_corrected": bias}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="bmi_hybrid_50k")
    ap.add_argument("--model-dir", default="nanda_a_mlp")
    ap.add_argument("--model-type", default="nanda")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.results_root) / args.model_dir
    rows: List[dict] = []
    for seed_dir in sorted(root.glob("seed_*")):
        ckpt = seed_dir / "checkpoint_final.pt"
        if not ckpt.exists():
            continue
        r = fve_for_checkpoint(str(ckpt), args.model_type, device)
        r["seed"] = seed_dir.name
        rows.append(r)
        print(f"{seed_dir.name}: plain={r['fve_plain']:.3f}  bias_corrected={r['fve_bias_corrected']:.3f}  freqs={r['key_freqs']}")

    if rows:
        n = len(rows)
        p80 = sum(r["fve_bias_corrected"] >= 0.80 for r in rows)
        p90 = sum(r["fve_bias_corrected"] >= 0.90 for r in rows)
        mean_plain = sum(r["fve_plain"] for r in rows) / n
        mean_bias = sum(r["fve_bias_corrected"] for r in rows) / n
        print(f"\nmean plain={mean_plain:.3f}  mean bias_corrected={mean_bias:.3f}")
        print(f"bias_corrected >=0.80: {p80}/{n}  >=0.90: {p90}/{n}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
