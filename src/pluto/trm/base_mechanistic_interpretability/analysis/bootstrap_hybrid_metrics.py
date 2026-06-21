#!/usr/bin/env python3
"""Bootstrap hybrid_metrics.json from existing path_a / fidelity_a single-seed runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

from pluto.trm.base_mechanistic_interpretability.analysis.checkpoint_selection import select_checkpoints
from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.hybrid_aggregate import _metric_from_progress, mean_std


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def collect_run(run_dir: Path) -> Dict[str, Any]:
    history = _read_json(run_dir / "training_history.json")
    if isinstance(history, list):
        selected = select_checkpoints(history)
    else:
        selected = {"final_checkpoint": None, "grokking_checkpoint": None, "best_fve_checkpoint": None}
    progress = _read_json(run_dir / "analysis/progress/progress_measures_grokking.json")
    hrm = _read_json(run_dir / "analysis/hrm/hrm_reasoning_guessing.json")
    return {
        "run_dir": str(run_dir),
        "selected_checkpoints": selected,
        "final_metrics": _metric_from_progress(progress) if progress else {},
        "hrm": hrm,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "nanda_a_mlp": Path(args.nanda),
        "trm_minimal": Path(args.trm_minimal),
        "trm_full_b": Path(args.trm_full_b),
        "trm_full_a": Path(args.trm_full_a),
    }
    models: Dict[str, Any] = {}
    for name, run_dir in mapping.items():
        if not run_dir.exists():
            print(f"skip missing {name}: {run_dir}")
            continue
        seed_runs = [collect_run(run_dir)]
        models[name] = {"seeds": seed_runs}
        for metric in [
            "test_accuracy",
            "logit_trig_fve",
            "logit_trig_fve_faithful",
            "trig_loss_test",
            "excluded_loss_test",
        ]:
            models[name][metric] = mean_std(s.get("final_metrics", {}).get(metric, 0.0) for s in seed_runs)

    results = {"results_root": "bootstrap", "models": models, "source": "path_a_fidelity_a_bootstrap"}
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "hybrid_metrics.json", results)

    fig_dst = out / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    for src_name, run_dir in mapping.items():
        fig_src = run_dir / "analysis/figures"
        if fig_src.exists():
            for pdf in fig_src.glob("*.pdf"):
                shutil.copy2(pdf, fig_dst / f"{src_name}_{pdf.name}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=".bmi-remote-results/hybrid_bootstrap/aggregate")
    p.add_argument(
        "--nanda",
        default=".bmi-remote-results/fidelity_a/nanda_fidelity_a_mlp",
    )
    p.add_argument(
        "--trm-minimal",
        default=".bmi-remote-results/path_a_20k/trm_minimal",
    )
    p.add_argument(
        "--trm-full-b",
        default=".bmi-remote-results/path_a_20k/trm_full",
    )
    p.add_argument(
        "--trm-full-a",
        default=".bmi-remote-results/fidelity_a/trm_full_fidelity_a_grokking_12k",
    )
    args = p.parse_args()
    r = run(args)
    print("bootstrap models", sorted(r["models"].keys()))


if __name__ == "__main__":
    main()
