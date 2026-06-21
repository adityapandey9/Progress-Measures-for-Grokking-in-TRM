#!/usr/bin/env python3
"""Three-way comparison: Nanda baseline vs minimal TRM vs full TRM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _history_metrics(history_path: Path) -> Dict[str, List[float]]:
    if not history_path.exists():
        return {"steps": [], "test_acc": [], "trig_fve": [], "trig_fve_faithful": []}
    hist = json.loads(history_path.read_text())
    steps = [h["step"] for h in hist]
    test_acc = [h.get("test_acc", 0.0) for h in hist]
    trig_fve = [h.get("logit_trig_fve", h.get("trig_fve", 0.0)) for h in hist]
    trig_fve_f = [h.get("logit_trig_fve_faithful", h.get("trig_fve_faithful", 0.0)) for h in hist]
    return {"steps": steps, "test_acc": test_acc, "trig_fve": trig_fve, "trig_fve_faithful": trig_fve_f}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    models = {
        "nanda": {"dir": Path(args.nanda_dir), "label": "Nanda 1-layer"},
        "trm_minimal": {"dir": Path(args.trm_minimal_dir), "label": "TRM minimal"},
        "trm_full": {"dir": Path(args.trm_full_dir), "label": "TRM full"},
    }

    summary: Dict[str, Any] = {"models": {}}
    fig_dir = ensure_dir(Path(args.output_dir) / "figures")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = {"nanda": "#1f77b4", "trm_minimal": "#ff7f0e", "trm_full": "#2ca02c"}

    for key, meta in models.items():
        run_dir = meta["dir"]
        prog = _load_json(run_dir / "analysis/progress/progress_measures_grokking.json")
        hist = _history_metrics(run_dir / "training_history.json")
        hrm = _load_json(run_dir / "analysis/hrm/hrm_reasoning_guessing.json")

        entry: Dict[str, Any] = {"label": meta["label"], "run_dir": str(run_dir)}
        if prog:
            entry["final"] = {
                "test_accuracy": prog.get("test_accuracy"),
                "trig_loss_test": prog.get("trig_loss_test"),
                "excluded_loss_test": prog.get("excluded_loss_test"),
                "logit_trig_fve": prog.get("logit_trig_fve"),
                "logit_trig_fve_faithful": prog.get("logit_trig_fve_faithful"),
                "key_frequencies": prog.get("key_frequencies"),
            }
        if hrm:
            entry["hrm"] = {
                "reasoning_mode_counts": hrm.get("reasoning_mode_counts"),
                "fixed_point_violation_rate": hrm.get("fixed_point_violation_rate"),
            }
        summary["models"][key] = entry

        if hist["steps"]:
            axes[0, 0].plot(hist["steps"], hist["test_acc"], label=meta["label"], color=colors[key])
            axes[0, 1].plot(hist["steps"], hist["trig_fve"], label=meta["label"], color=colors[key])
            axes[1, 0].plot(hist["steps"], hist["trig_fve_faithful"], label=meta["label"], color=colors[key])

    axes[0, 0].set_title("Test accuracy vs step")
    axes[0, 0].legend()
    axes[0, 1].set_title("Logit trig-FVE (cos-only) vs step")
    axes[0, 1].legend()
    axes[1, 0].set_title("Logit trig-FVE (faithful cos+sin) vs step")
    axes[1, 0].legend()

    labels, fve_vals, fve_f_vals = [], [], []
    for key, meta in models.items():
        fin = summary["models"].get(key, {}).get("final", {})
        if fin:
            labels.append(meta["label"])
            fve = fin.get("logit_trig_fve", {})
            fve_f = fin.get("logit_trig_fve_faithful", {})
            fve_vals.append(fve.get("fve_mean", 0.0) if isinstance(fve, dict) else float(fve or 0))
            fve_f_vals.append(fve_f.get("fve_mean", 0.0) if isinstance(fve_f, dict) else float(fve_f or 0))
    x = range(len(labels))
    axes[1, 1].bar([i - 0.15 for i in x], fve_vals, width=0.3, label="cos-only FVE")
    axes[1, 1].bar([i + 0.15 for i in x], fve_f_vals, width=0.3, label="faithful FVE")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(labels, rotation=15)
    axes[1, 1].set_title("Final checkpoint FVE")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_path_a_compare.pdf")
    plt.close(fig)

    out = ensure_dir(Path(args.output_dir))
    save_json(out / "path_a_compare.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nanda-dir", default="bmi_path_a/nanda")
    p.add_argument("--trm-minimal-dir", default="bmi_path_a/trm_minimal")
    p.add_argument("--trm-full-dir", default="bmi_path_a/trm_full")
    p.add_argument("--output-dir", default="bmi_path_a/compare")
    args = p.parse_args()
    r = run(args)
    for k, v in r["models"].items():
        fin = v.get("final", {})
        print(f"{k}: test_acc={fin.get('test_accuracy')} fve={fin.get('logit_trig_fve')}")


if __name__ == "__main__":
    main()
