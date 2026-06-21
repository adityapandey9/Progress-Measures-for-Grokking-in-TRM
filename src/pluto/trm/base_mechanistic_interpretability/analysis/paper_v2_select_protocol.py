#!/usr/bin/env python3
"""Select best EP1 weight-decay protocol by final-checkpoint adaptive FVE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import save_json
from pluto.trm.base_mechanistic_interpretability.analysis.corrected_fve_summary import (
    summarize_model,
)


def _count_clean(rows: List[Dict[str, Any]], threshold: float = 0.95) -> int:
    return sum(1 for r in rows if r.get("fve_adaptive", 0) >= threshold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True, help="EP1 root with wd_*/trm_minimal/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.results_root)
    candidates: List[Dict[str, Any]] = []

    for wd_dir in sorted(root.glob("wd_*")):
        trm_root = wd_dir / "trm_minimal"
        if not trm_root.exists():
            continue
        wd_str = wd_dir.name.replace("wd_", "")
        try:
            wd = float(wd_str)
        except ValueError:
            continue
        rows = summarize_model(wd_dir, "trm_minimal", "trm_minimal", device)
        n_clean = _count_clean(rows, args.threshold)
        mean_fve = sum(r["fve_adaptive"] for r in rows) / max(len(rows), 1)
        candidates.append(
            {
                "weight_decay": wd,
                "n_seeds": len(rows),
                "n_clean_final": n_clean,
                "mean_fve_adaptive": round(mean_fve, 4),
                "seeds": rows,
            }
        )
        print(f"wd={wd}: {n_clean}/{len(rows)} seeds >= {args.threshold}")

    if not candidates:
        raise SystemExit(f"No EP1 candidates under {root}")

    # Prefer most clean seeds, then highest mean FVE.
    best = max(candidates, key=lambda c: (c["n_clean_final"], c["mean_fve_adaptive"]))
    out = {
        "best_weight_decay": best["weight_decay"],
        "best_n_clean": best["n_clean_final"],
        "threshold": args.threshold,
        "candidates": candidates,
    }
    save_json(Path(args.out), out)
    print(f"selected wd={best['weight_decay']} ({best['n_clean_final']} clean seeds)")


if __name__ == "__main__":
    main()
