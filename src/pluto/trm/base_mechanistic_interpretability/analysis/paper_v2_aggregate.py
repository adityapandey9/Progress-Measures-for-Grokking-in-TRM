#!/usr/bin/env python3
"""Aggregate paper v2 Arc A metrics: Nanda calibration + best-wd TRM minimal + sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.hybrid_aggregate import (
    _aggregate_metric,
    collect_seed_run,
    diagnose_seed_variance,
    mean_std,
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _collect_model(root: Path, model_name: str) -> Dict[str, Any]:
    seed_runs: List[Dict[str, Any]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        if seed_dir.is_dir():
            seed_runs.append(collect_seed_run(seed_dir))
    if not seed_runs:
        return {}
    out: Dict[str, Any] = {"seeds": seed_runs, "model_root": str(root)}
    for metric in [
        "test_accuracy",
        "logit_trig_fve",
        "logit_trig_fve_faithful",
        "logit_trig_fve_adaptive",
        "trig_loss_test",
        "excluded_loss_test",
    ]:
        out[f"final_{metric}"] = _aggregate_metric(seed_runs, metric, "final_metrics")
        out[f"grokking_{metric}"] = mean_std(
            s.get("history_metrics", {}).get("grokking", {}).get(metric, 0.0) for s in seed_runs
        )
        out[f"best_fve_{metric}"] = mean_std(
            s.get("history_metrics", {}).get("best_fve", {}).get(metric, 0.0) for s in seed_runs
        )
        out[metric] = out[f"final_{metric}"]
    out["seed_diagnoses"] = [s["seed_diagnosis"] for s in seed_runs]
    best_seed = max(
        seed_runs,
        key=lambda s: s.get("history_metrics", {}).get("best_fve", {}).get("logit_trig_fve_faithful", 0.0),
    )
    out["best_seed_run"] = {
        "seed": best_seed.get("seed"),
        "run_dir": best_seed.get("run_dir"),
        "best_fve": best_seed.get("history_metrics", {}).get("best_fve", {}),
        "grokking": best_seed.get("history_metrics", {}).get("grokking", {}),
        "final": best_seed.get("history_metrics", {}).get("final", {}),
    }
    # Stable seed: highest final adaptive FVE (for mainline RE).
    stable = max(
        seed_runs,
        key=lambda s: s.get("final_metrics", {}).get("logit_trig_fve_adaptive", 0.0),
    )
    out["mainline_seed_run"] = {
        "seed": stable.get("seed"),
        "run_dir": stable.get("run_dir"),
        "final_metrics": stable.get("final_metrics", {}),
    }
    return out


def _sweep_summary(sweep_root: Path, param_key: str, *, prefix: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = f"{prefix}*/trm_minimal/seed_*" if prefix else "*/trm_minimal/seed_*"
    for run_dir in sorted(sweep_root.glob(pattern)):
        tag = run_dir.parent.parent.name
        if prefix and not tag.startswith(prefix):
            continue
        param_val = tag.split("_", 1)[-1]
        hist_path = run_dir / "training_history.json"
        if not hist_path.exists():
            continue
        history = json.loads(hist_path.read_text())
        diag = diagnose_seed_variance(history, model_name=tag)
        final = collect_seed_run(run_dir)
        rows.append(
            {
                param_key: param_val,
                "tag": tag,
                "run_dir": str(run_dir),
                "grokking_step": diag.get("grokking_step", -1),
                "final_fve": diag.get("final_fve", 0.0),
                "final_accuracy": diag.get("final_accuracy", 0.0),
                "peak_fve": diag.get("peak_fve", 0.0),
                "final_metrics": final.get("final_metrics", {}),
            }
        )
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.results_root)
    best_wd = float(args.best_wd)
    models: Dict[str, Any] = {}

    nanda_root = root / "nanda_a_mlp"
    if nanda_root.exists():
        models["nanda_a_mlp"] = _collect_model(nanda_root, "nanda_a_mlp")

    trm_root = root / "ep1" / f"wd_{best_wd}" / "trm_minimal"
    if trm_root.exists():
        models["trm_minimal"] = _collect_model(trm_root, "trm_minimal")

    protocol = _read_json(root / "ep1" / "protocol_selection.json")
    ep1_candidates = protocol.get("candidates", [])

    corrected = _read_json(root / "aggregate" / "corrected_fve_summary.json")
    if corrected:
        for model_key, block in corrected.items():
            if model_key in models:
                models[model_key]["corrected_fve"] = block

    results: Dict[str, Any] = {
        "results_root": str(root),
        "best_weight_decay": best_wd,
        "protocol_selection": protocol,
        "models": models,
        "ep1_weight_decay_sweep": ep1_candidates,
        "ep3_data_fraction_sweep": _sweep_summary(root / "ep3", "frac_train", prefix="frac_"),
        "ep3_weight_decay_sweep": _sweep_summary(root / "ep3", "weight_decay", prefix="wd_"),
    }

    if corrected:
        results["corrected_fve"] = corrected

    ptrm_path = root / "aggregate" / "ptrm_fve_summary.json"
    if ptrm_path.exists():
        results["ptrm_fve"] = json.loads(ptrm_path.read_text())

    out_dir = ensure_dir(Path(args.output_dir))
    save_json(out_dir / "paper_v2_metrics.json", results)
    save_json(out_dir / "hybrid_metrics.json", results)  # alias for plot helpers
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--best-wd", required=True, type=float)
    args = ap.parse_args()
    r = run(args)
    print("models", sorted(r["models"].keys()), "best_wd", r["best_weight_decay"])


if __name__ == "__main__":
    main()
