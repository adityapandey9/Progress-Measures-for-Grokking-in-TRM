#!/usr/bin/env python3
"""Aggregate hybrid rigor metrics across models and seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pluto.trm.base_mechanistic_interpretability.analysis.checkpoint_selection import (
    _fve,
    select_checkpoints,
)
from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json


def mean_std(xs: Iterable[float]) -> Dict[str, float | int]:
    vals = list(xs)
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return {"mean": mean, "std": math.sqrt(var), "n": len(vals)}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _metric_from_history_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    fve_val = row.get("logit_trig_fve_faithful", row.get("logit_trig_fve", 0.0))
    if isinstance(fve_val, dict):
        fve_faithful = float(fve_val.get("fve_mean", fve_val.get("fve_faithful", 0.0)))
        fve = float(row.get("logit_trig_fve", {}).get("fve_mean", 0.0)) if isinstance(row.get("logit_trig_fve"), dict) else fve_faithful
    else:
        fve_faithful = float(fve_val or 0.0)
        fve = float(row.get("logit_trig_fve", fve_faithful) or 0.0)
    fve_adaptive = row.get("logit_trig_fve_adaptive", 0.0)
    if isinstance(fve_adaptive, dict):
        fve_adaptive = float(fve_adaptive.get("fve_mean", 0.0))
    else:
        fve_adaptive = float(fve_adaptive or 0.0)
    return {
        "step": int(row.get("step", -1)),
        "test_accuracy": float(row.get("test_acc", row.get("test_accuracy", 0.0))),
        "test_loss": float(row.get("test_loss", 0.0)),
        "trig_loss_test": float(row.get("trig_loss_test", 0.0)),
        "excluded_loss_test": float(row.get("excluded_loss_test", 0.0)),
        "logit_trig_fve": fve,
        "logit_trig_fve_faithful": fve_faithful,
        "logit_trig_fve_adaptive": fve_adaptive,
        "n_key_frequencies_adaptive": int(row.get("n_key_frequencies_adaptive", 0)),
        "embedding_gini": float(row.get("embedding_gini", 0.0)),
        "unembed_gini": float(row.get("unembed_gini", 0.0)),
        "key_frequencies": row.get("key_frequencies", []),
    }


def _metric_from_progress(pm: Dict[str, Any]) -> Dict[str, Any]:
    fve = pm.get("logit_trig_fve", {})
    faithful = pm.get("logit_trig_fve_faithful", {})
    adaptive = pm.get("logit_trig_fve_adaptive", {})
    return {
        "test_accuracy": float(pm.get("test_accuracy", 0.0)),
        "trig_loss_test": float(pm.get("trig_loss_test", 0.0)),
        "excluded_loss_test": float(pm.get("excluded_loss_test", 0.0)),
        "logit_trig_fve": float(fve.get("fve_mean", 0.0)) if isinstance(fve, dict) else float(fve or 0.0),
        "logit_trig_fve_faithful": float(faithful.get("fve_mean", 0.0))
        if isinstance(faithful, dict)
        else float(faithful or 0.0),
        "logit_trig_fve_adaptive": float(adaptive.get("fve_mean", 0.0))
        if isinstance(adaptive, dict)
        else float(adaptive or 0.0),
        "key_frequencies": pm.get("key_frequencies", []),
    }


def diagnose_seed_variance(history: List[Dict[str, Any]], *, model_name: str) -> Dict[str, Any]:
    """Explain FVE/accuracy divergence across training (post-grokking cleanup)."""
    rows = sorted([r for r in history if str(r.get("step", "")).isdigit()], key=lambda r: int(r["step"]))
    if not rows:
        return {"model": model_name, "diagnosis": "no_history"}
    selected = select_checkpoints(rows)
    final = selected.get("final_checkpoint") or {}
    grok = selected.get("grokking_checkpoint") or {}
    best = selected.get("best_fve_checkpoint") or {}
    final_fve = _fve(final)
    grok_fve = _fve(grok) if grok else 0.0
    best_fve = _fve(best) if best else 0.0
    grok_step = int(grok.get("step", -1)) if grok else -1
    final_step = int(final.get("step", -1))
    post_grok = [r for r in rows if grok_step > 0 and int(r["step"]) >= grok_step]
    fve_after_grok = [_fve(r) for r in post_grok if _fve(r) > 0]
    peak_fve = max((_fve(r) for r in rows), default=0.0)
    peak_step = int(max(rows, key=_fve)["step"]) if rows else -1
    cleanup_drop = peak_fve - final_fve if peak_fve > 0 else 0.0
    causes: List[str] = []
    if grok and final_fve < grok_fve * 0.85:
        causes.append("post_grokking_fve_decay")
    if float(final.get("test_acc", 0)) < 0.99:
        causes.append("incomplete_grokking_by_20k")
    if cleanup_drop > 0.25:
        causes.append("cleanup_phase_fve_collapse")
    if not causes:
        causes.append("stable_circuit")
    return {
        "model": model_name,
        "grokking_step": grok_step,
        "peak_fve_step": peak_step,
        "peak_fve": peak_fve,
        "grokking_fve": grok_fve,
        "best_fve_checkpoint_fve": best_fve,
        "final_fve": final_fve,
        "final_accuracy": float(final.get("test_acc", 0)),
        "fve_drop_after_peak": cleanup_drop,
        "mean_fve_post_grokking": sum(fve_after_grok) / len(fve_after_grok) if fve_after_grok else 0.0,
        "likely_causes": causes,
    }


def collect_seed_run(run_dir: Path) -> Dict[str, Any]:
    history_path = run_dir / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    selected = select_checkpoints(history)
    progress = _read_json(run_dir / "analysis/progress/progress_measures_grokking.json")
    hrm = _read_json(run_dir / "analysis/hrm/hrm_reasoning_guessing.json")
    ren = _read_json(run_dir / "analysis/ren/ren_checkpoint_suite.json")
    seed_name = run_dir.name
    return {
        "run_dir": str(run_dir),
        "seed": seed_name,
        "selected_checkpoints": {
            k: _metric_from_history_row(v) for k, v in selected.items()
        },
        "final_metrics": _metric_from_progress(progress)
        if progress
        else _metric_from_history_row(selected.get("final_checkpoint")),
        "history_metrics": {
            "final": _metric_from_history_row(selected.get("final_checkpoint")),
            "grokking": _metric_from_history_row(selected.get("grokking_checkpoint")),
            "best_fve": _metric_from_history_row(selected.get("best_fve_checkpoint")),
        },
        "seed_diagnosis": diagnose_seed_variance(history, model_name=seed_name),
        "hrm": hrm,
        "ren_diagnosis": ren.get("mechanism_verdict", {}) if ren else {},
        "ren_suite": ren,
    }


def _aggregate_metric(seed_runs: List[Dict[str, Any]], key: str, source: str = "final_metrics") -> Dict[str, float | int]:
    return mean_std(s.get(source, {}).get(key, 0.0) for s in seed_runs)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.results_root)
    models: Dict[str, Any] = {}
    for model_dir in sorted(root.glob("*")):
        if not model_dir.is_dir() or model_dir.name in ("aggregate", "paper"):
            continue
        seed_runs = []
        for seed_dir in sorted(model_dir.glob("seed_*")):
            if seed_dir.is_dir():
                seed_runs.append(collect_seed_run(seed_dir))
        if not seed_runs and (model_dir / "training_history.json").exists():
            seed_runs.append(collect_seed_run(model_dir))
        if not seed_runs:
            continue
        models[model_dir.name] = {"seeds": seed_runs}
        for metric in [
            "test_accuracy",
            "logit_trig_fve",
            "logit_trig_fve_faithful",
            "logit_trig_fve_adaptive",
            "trig_loss_test",
            "excluded_loss_test",
        ]:
            models[model_dir.name][f"final_{metric}"] = _aggregate_metric(seed_runs, metric, "final_metrics")
            models[model_dir.name][f"grokking_{metric}"] = mean_std(
                s.get("history_metrics", {}).get("grokking", {}).get(metric, 0.0) for s in seed_runs
            )
            models[model_dir.name][f"best_fve_{metric}"] = mean_std(
                s.get("history_metrics", {}).get("best_fve", {}).get(metric, 0.0) for s in seed_runs
            )
            models[model_dir.name][metric] = models[model_dir.name][f"final_{metric}"]
        models[model_dir.name]["seed_diagnoses"] = [s["seed_diagnosis"] for s in seed_runs]
        best_seed = max(
            seed_runs,
            key=lambda s: s.get("history_metrics", {}).get("best_fve", {}).get("logit_trig_fve_faithful", 0.0),
        )
        models[model_dir.name]["best_seed_run"] = {
            "seed": best_seed.get("seed"),
            "best_fve": best_seed.get("history_metrics", {}).get("best_fve", {}),
            "grokking": best_seed.get("history_metrics", {}).get("grokking", {}),
        }

    results = {"results_root": str(root), "models": models}
    corr_path = root / "corrected_fve_summary.json"
    if corr_path.exists():
        corrected = json.loads(corr_path.read_text())
        results["corrected_fve"] = corrected
        for model_key, block in corrected.items():
            if model_key not in models:
                continue
            models[model_key]["corrected_fve"] = block
            models[model_key]["final_logit_trig_fve_adaptive"] = {
                "mean": block.get("mean_fve_adaptive", 0.0),
                "std": 0.0,
                "n": block.get("n_seeds", 0),
            }
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "hybrid_metrics.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", required=True)
    p.add_argument("--output-dir", default="bmi_hybrid/aggregate")
    args = p.parse_args()
    r = run(args)
    print("models", sorted(r["models"].keys()))


if __name__ == "__main__":
    main()
