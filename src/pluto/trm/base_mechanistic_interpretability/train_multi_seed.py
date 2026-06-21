#!/usr/bin/env python3
"""Train multiple seeds for robustness (shorter smoke-friendly variant)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig
from pluto.trm.base_mechanistic_interpretability.train_grokking import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--runs-root", default="bmi_grokking_runs")
    args = p.parse_args()
    root = Path(args.runs_root)
    py = sys.executable
    analysis_root = Path(__file__).resolve().parent

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        out = root / f"seed_{seed}"
        cfg = ModAddGrokkingConfig(seed=seed, max_steps=args.max_steps, save_every=2000)
        print(f"==> seed={seed} -> {out}")
        train(cfg, out)
        ckpt = out / "checkpoint_final.pt"
        hist = out / "training_history.json"
        subprocess.check_call(
            [
                py,
                str(analysis_root / "run_all.py"),
                "--checkpoint",
                str(ckpt),
                "--run-dir",
                str(out),
                "--output-dir",
                str(out / "analysis"),
                "--training-history",
                str(hist),
                "--skip-trajectory",
            ]
        )


if __name__ == "__main__":
    main()
