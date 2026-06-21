#!/usr/bin/env python3
"""Audit FVE under excluded/adaptive vs Nanda neuron key frequencies (no retraining)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.fve_eval import eval_checkpoint_fve_metrics

MODEL_DIRS: Dict[str, str] = {
    "nanda_a_mlp": "nanda",
    "nanda": "nanda",
    "trm_minimal": "trm_minimal",
    "trm_full": "trm_full",
    "trm_full_b": "trm_full",
    "trm_full_a": "trm_full",
}


def _discover_checkpoints(root: Path) -> List[Tuple[str, str, Path]]:
    """Return (model_label, seed_label, checkpoint_path) tuples."""
    found: List[Tuple[str, str, Path]] = []
    if not root.exists():
        return found

    for ck in sorted(root.rglob("checkpoint_final.pt")):
        rel = ck.relative_to(root)
        parts = rel.parts
        if len(parts) >= 3 and parts[-2].startswith("seed_"):
            found.append((parts[-3], parts[-2], ck))
        elif len(parts) == 2:
            found.append((parts[0], "seed_0", ck))
        else:
            found.append((parts[0] if parts else "unknown", "seed_0", ck))

    # Deduplicate same model/seed keeping deepest path (prefer ep1/wd_* copies last).
    dedup: Dict[Tuple[str, str], Path] = {}
    for model, seed, ck in found:
        dedup[(model, seed)] = ck
    return [(m, s, p) for (m, s), p in sorted(dedup.items())]


def _summarize(rows: List[Dict]) -> Dict:
    n = len(rows)
    if n == 0:
        return {"n_seeds": 0}
    return {
        "n_seeds": n,
        "mean_fve_legacy_k5": round(sum(r["fve_legacy_k5"] for r in rows) / n, 4),
        "mean_fve_adaptive": round(sum(r["fve_adaptive"] for r in rows) / n, 4),
        "mean_fve_neuron_keys": round(sum(r["fve_neuron_keys"] for r in rows) / n, 4),
        "n_seeds_adaptive_ge_0.95": sum(r["fve_adaptive"] >= 0.95 for r in rows),
        "n_seeds_neuron_ge_0.95": sum(r["fve_neuron_keys"] >= 0.95 for r in rows),
        "n_seeds_neuron_gt_adaptive": sum(r["fve_neuron_keys"] > r["fve_adaptive"] + 0.01 for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        default=None,
        help="Workspace root for relative --results-roots (default: parent of pluto package)",
    )
    ap.add_argument(
        "--results-roots",
        nargs="+",
        default=[
            ".bmi-remote-results/nanda_50k_ren",
            ".bmi-remote-results/paper_v2_arc_a/ep1/wd_1.0",
            ".bmi-remote-results/hybrid_rigor",
            ".bmi-remote-results/path_a_20k",
            "bmi_hybrid_50k",
            "paper_v2_arc_a/ep1/wd_1.0",
            "bmi_hybrid",
        ],
    )
    ap.add_argument("--out", default=".bmi-remote-results/neuron_key_freq_fve_audit.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = Path(args.repo_root) if args.repo_root else Path.cwd()
    summary: Dict[str, Dict] = {}

    for root_arg in args.results_roots:
        root = Path(root_arg)
        if not root.is_absolute():
            candidates = [repo / root, Path(__file__).resolve().parents[3] / root]
            root = next((c for c in candidates if c.exists()), candidates[0])
        if not root.exists():
            print(f"SKIP missing root: {root}")
            continue
        for model_dir, seed, ckpt in _discover_checkpoints(root):
            model_type = MODEL_DIRS.get(model_dir, "trm_minimal")
            key = f"{root.name}/{model_dir}"
            summary.setdefault(key, {"results_root": str(root), "model_dir": model_dir, "seeds": []})
            existing = {r["seed"] for r in summary[key]["seeds"]}
            if seed in existing:
                continue
            try:
                row = eval_checkpoint_fve_metrics(str(ckpt), model_type, device)
            except Exception as exc:  # noqa: BLE001 — audit should continue across seeds
                print(f"SKIP {ckpt}: {exc}")
                continue
            row["seed"] = seed
            summary[key]["seeds"].append(row)
            print(
                f"{model_dir}/{seed}: legacy={row['fve_legacy_k5']:.3f} "
                f"adaptive={row['fve_adaptive']:.3f} neuron={row['fve_neuron_keys']:.3f} "
                f"K_neuron={row['n_key_freqs_neuron']} freqs={row['key_freqs_neuron'][:8]}"
            )

    for key, block in summary.items():
        block["aggregate"] = _summarize(block["seeds"])
        agg = block["aggregate"]
        if agg.get("n_seeds"):
            print(
                f"\n==> {key}: adaptive>={agg['n_seeds_adaptive_ge_0.95']}/{agg['n_seeds']}  "
                f"neuron>={agg['n_seeds_neuron_ge_0.95']}/{agg['n_seeds']}  "
                f"neuron>adaptive: {agg['n_seeds_neuron_gt_adaptive']}/{agg['n_seeds']}"
            )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo / out_path
    ensure_dir(out_path.parent)
    save_json(out_path, summary)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
