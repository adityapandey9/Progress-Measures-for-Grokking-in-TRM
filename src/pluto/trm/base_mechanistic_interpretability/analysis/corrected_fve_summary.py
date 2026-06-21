#!/usr/bin/env python3
"""Corrected trig-FVE summary using data-driven (adaptive) key-frequency count.

The legacy metric hard-coded ``top_k=5`` key frequencies, which under-reports the
faithful FVE of models whose sparse-Fourier circuit spans 6-8 frequencies. This
tool recomputes FVE on final checkpoints with the adaptive selector and writes a
per-seed/per-model JSON summary (no retraining required).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.fve_eval import eval_checkpoint_fve_metrics

MODELS = {
    "nanda_a_mlp": "nanda",
    "trm_minimal": "trm_minimal",
    "trm_full_b": "trm_full",
}


@torch.no_grad()
def summarize_model(root: Path, model_dir: str, model_type: str, device: torch.device) -> List[Dict]:
    rows: List[Dict] = []
    for seed_dir in sorted((root / model_dir).glob("seed_*")):
        ck = seed_dir / "checkpoint_final.pt"
        if not ck.exists():
            continue
        metrics = eval_checkpoint_fve_metrics(str(ck), model_type, device)
        rows.append(
            {
                "seed": seed_dir.name,
                "fve_legacy_k5": metrics["fve_legacy_k5"],
                "fve_adaptive": metrics["fve_adaptive"],
                "fve_neuron_keys": metrics["fve_neuron_keys"],
                "n_key_freqs_adaptive": metrics["n_key_freqs_adaptive"],
                "n_key_freqs_neuron": metrics["n_key_freqs_neuron"],
                "key_freqs_neuron": metrics["key_freqs_neuron"],
            }
        )
        print(
            f"{model_dir}/{seed_dir.name}: legacy@5={metrics['fve_legacy_k5']:.3f}  "
            f"adaptive={metrics['fve_adaptive']:.3f}  neuron={metrics['fve_neuron_keys']:.3f}  "
            f"K_adapt={metrics['n_key_freqs_adaptive']} K_neuron={metrics['n_key_freqs_neuron']}"
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="bmi_hybrid_50k")
    ap.add_argument("--out", default="bmi_hybrid_50k/corrected_fve_summary.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.results_root)
    summary: Dict[str, Dict] = {}
    for model_dir, model_type in MODELS.items():
        if not (root / model_dir).exists():
            continue
        rows = summarize_model(root, model_dir, model_type, device)
        if not rows:
            continue
        n = len(rows)
        summary[model_dir] = {
            "seeds": rows,
            "mean_fve_legacy_k5": round(sum(r["fve_legacy_k5"] for r in rows) / n, 4),
            "mean_fve_adaptive": round(sum(r["fve_adaptive"] for r in rows) / n, 4),
            "mean_fve_neuron_keys": round(sum(r["fve_neuron_keys"] for r in rows) / n, 4),
            "n_seeds_adaptive_ge_0.95": sum(r["fve_adaptive"] >= 0.95 for r in rows),
            "n_seeds_neuron_ge_0.95": sum(r["fve_neuron_keys"] >= 0.95 for r in rows),
            "n_seeds_adaptive_ge_0.80": sum(r["fve_adaptive"] >= 0.80 for r in rows),
            "n_seeds": n,
        }
        s = summary[model_dir]
        print(
            f"  => {model_dir}: mean legacy@5={s['mean_fve_legacy_k5']:.3f}  "
            f"mean adaptive={s['mean_fve_adaptive']:.3f}  mean neuron={s['mean_fve_neuron_keys']:.3f}  "
            f"adaptive>=0.95: {s['n_seeds_adaptive_ge_0.95']}/{n}  "
            f"neuron>=0.95: {s['n_seeds_neuron_ge_0.95']}/{n}\n"
        )

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
