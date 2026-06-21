#!/usr/bin/env python3
"""Aggregate multi-seed grokking runs for statistics (Nanda robustness)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json


def run(args: argparse.Namespace) -> Dict[str, Any]:
    base = Path(args.runs_root)
    seeds: List[Dict[str, Any]] = []
    for seed_dir in sorted(base.glob("seed_*")):
        hist_path = seed_dir / "training_history.json"
        pm_path = seed_dir / "analysis" / "progress" / "progress_measures_grokking.json"
        if not hist_path.exists():
            continue
        hist = json.loads(hist_path.read_text())
        final = hist[-1] if hist else {}
        row: Dict[str, Any] = {
            "seed_dir": str(seed_dir),
            "final_test_acc": final.get("test_acc", 0.0),
            "final_train_acc": final.get("train_acc", 0.0),
            "grokking_step": next((h["step"] for h in hist if h.get("test_acc", 0) > 0.9), None),
        }
        if pm_path.exists():
            pm = json.loads(pm_path.read_text())
            row["key_frequencies"] = pm.get("key_frequencies", [])
            row["trig_loss_test"] = pm.get("trig_loss_test")
            row["excluded_loss_test"] = pm.get("excluded_loss_test")
        seeds.append(row)

    test_accs = [s["final_test_acc"] for s in seeds]
    agg = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "final_test_acc_mean": statistics.mean(test_accs) if test_accs else 0.0,
        "final_test_acc_std": statistics.pstdev(test_accs) if len(test_accs) > 1 else 0.0,
    }
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "multi_seed_aggregate.json", agg)
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default="bmi_grokking_runs")
    p.add_argument("--output-dir", default="bmi_analysis/multi_seed")
    args = p.parse_args()
    r = run(args)
    print(f"n={r['n_seeds']} test_acc_mean={r['final_test_acc_mean']:.4f}")


if __name__ == "__main__":
    main()
