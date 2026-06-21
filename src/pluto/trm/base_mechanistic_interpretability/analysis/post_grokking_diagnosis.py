#!/usr/bin/env python3
"""Classify post-grokking degradation: Nanda cleanup vs Ren guessing (2601.10679)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.hybrid_aggregate import diagnose_seed_variance


def classify_from_ren_suite(suite: Dict[str, Any]) -> Dict[str, Any]:
    return suite.get("mechanism_verdict") or {}


def classify_run(run_dir: Path) -> Dict[str, Any]:
    ren_path = run_dir / "analysis" / "ren" / "ren_checkpoint_suite.json"
    if ren_path.exists():
        suite = json.loads(ren_path.read_text())
        return {
            "run_dir": str(run_dir),
            "source": "ren_checkpoint_suite",
            **suite.get("mechanism_verdict", {}),
            "nanda_seed_diagnosis": suite.get("nanda_seed_diagnosis", {}),
        }
    history_path = run_dir / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    nanda = diagnose_seed_variance(history, model_name=run_dir.name)
    causes = nanda.get("likely_causes") or []
    primary = "nanda_cleanup" if "cleanup_phase_fve_collapse" in causes else "undertraining" if "incomplete_grokking_by_20k" in causes else "stable_circuit"
    return {
        "run_dir": str(run_dir),
        "source": "nanda_only",
        "primary_mechanism": primary,
        "hypotheses": ["H2_nanda_cleanup"] if primary == "nanda_cleanup" else ["H1_undertraining"] if primary == "undertraining" else ["H_stable_circuit"],
        "ren_applies": False,
        "ren_null_on_minimal": True,
        "nanda_seed_diagnosis": nanda,
    }


def run_results_root(results_root: Path, *, trm_models: List[str] | None = None) -> Dict[str, Any]:
    trm_models = trm_models or ["trm_minimal", "trm_full_b", "trm_full_a"]
    report: Dict[str, Any] = {"results_root": str(results_root), "models": {}}
    for model in trm_models:
        model_dir = results_root / model
        if not model_dir.is_dir():
            continue
        seeds = []
        for seed_dir in sorted(model_dir.glob("seed_*")):
            seeds.append(classify_run(seed_dir))
        report["models"][model] = {"seeds": seeds}
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="bmi_hybrid")
    p.add_argument("--run-dir", default="")
    p.add_argument("--output", default="")
    args = p.parse_args()
    if args.run_dir:
        result = classify_run(Path(args.run_dir))
    else:
        result = run_results_root(Path(args.results_root))
    if args.output:
        save_json(Path(args.output), result)
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
