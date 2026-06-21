#!/usr/bin/env python3
"""EP1b: evaluate adaptive FVE at best-FVE training checkpoint (early-stop protocol)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.checkpoint_selection import select_checkpoints
from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.corrected_fve_summary import summarize_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--model-dir", default="trm_minimal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.results_root) / args.model_dir
    rows: List[Dict[str, Any]] = []

    for seed_dir in sorted(root.glob("seed_*")):
        hist_path = seed_dir / "training_history.json"
        if not hist_path.exists():
            continue
        history = json.loads(hist_path.read_text())
        sel = select_checkpoints(history)
        best = sel.get("best_fve_checkpoint") or {}
        step = int(best.get("step", -1))
        ckpt = seed_dir / f"checkpoint_step{step}.pt"
        if not ckpt.exists():
            ckpt = seed_dir / "checkpoint_final.pt"
        # Symlink-style eval via temp final name is messy; evaluate step ckpt directly.
        from pluto.trm.base_mechanistic_interpretability.analysis.corrected_fve_summary import summarize_model as _unused
        from pluto.trm.base_mechanistic_interpretability.analysis.common import eval_all_pairs_logits_from_checkpoint
        from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset
        from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
            fit_trig_logits_fve_bias_corrected,
            identify_key_frequencies_adaptive,
            logits_grid,
        )

        logits, cfg, _ = eval_all_pairs_logits_from_checkpoint(str(ckpt), "trm_minimal", device, prefer_ptrm=False)
        ds = ModAddFullDataset(cfg)
        lab = ds.labels[:, 2].to(device)
        tr = ds.train_mask.to(device)
        te = ds.test_mask.to(device)
        grid = logits_grid(logits, cfg.p)
        freqs = identify_key_frequencies_adaptive(grid, lab, tr, te, cfg.p)
        fve = float(fit_trig_logits_fve_bias_corrected(grid, freqs, cfg.p)["fve_mean"])
        final_ck = seed_dir / "checkpoint_final.pt"
        flogits, _, _ = eval_all_pairs_logits_from_checkpoint(str(final_ck), "trm_minimal", device, prefer_ptrm=False)
        fgrid = logits_grid(flogits, cfg.p)
        ffreqs = identify_key_frequencies_adaptive(fgrid, lab, tr, te, cfg.p)
        final_fve = float(fit_trig_logits_fve_bias_corrected(fgrid, ffreqs, cfg.p)["fve_mean"])
        rows.append(
            {
                "seed": seed_dir.name,
                "best_fve_step": step,
                "best_fve_checkpoint": str(ckpt.name),
                "best_fve_adaptive": round(fve, 4),
                "final_fve_adaptive": round(final_fve, 4),
            }
        )
        print(f"{seed_dir.name}: final={final_fve:.3f} best@{step}={fve:.3f}")

    n_best = sum(1 for r in rows if r["best_fve_adaptive"] >= 0.95)
    n_final = sum(1 for r in rows if r["final_fve_adaptive"] >= 0.95)
    out = {"seeds": rows, "n_clean_best_fve": n_best, "n_clean_final": n_final}
    save_json(Path(args.out), out)
    print(f"best-FVE >=0.95: {n_best}/{len(rows)}  final >=0.95: {n_final}/{len(rows)}")


if __name__ == "__main__":
    main()
