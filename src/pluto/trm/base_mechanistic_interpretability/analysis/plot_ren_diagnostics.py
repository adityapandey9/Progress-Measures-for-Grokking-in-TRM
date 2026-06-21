#!/usr/bin/env python3
"""Plot Ren (2601.10679) diagnostic figures for the hybrid paper.

Data layouts supported (whichever exists):
  * per-seed:  <root>/<model>/seed_*/analysis/ren/ren_checkpoint_suite.json
  * flat sync: <root>/<model>/ren_checkpoint_suite.json   (results synced back)
The fixed-point-violation rate and primary-mechanism labels are read directly
from the aggregate ren_diagnosis_report.json (found at <root>/ren_diagnosis_report.json
or <root>/aggregate/ren_diagnosis_report.json), so no per-seed suite files are
required for those two figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from pluto.trm.base_mechanistic_interpretability.analysis import figstyle
from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir

MODELS = ["trm_minimal", "trm_full_b", "trm_full_a", "nanda_a_mlp"]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _find_suites(results_root: Path, model: str) -> List[Tuple[str, Path]]:
    """Return (label, suite_path) for a model under either supported layout."""
    nested = sorted((results_root / model).glob("seed_*/analysis/ren/ren_checkpoint_suite.json"))
    if nested:
        return [(p.parent.parent.parent.name, p) for p in nested]
    flat = results_root / model / "ren_checkpoint_suite.json"
    return [(model, flat)] if flat.exists() else []


def _model_color(model: str) -> str:
    idx = MODELS.index(model) if model in MODELS else len(figstyle.PALETTE) - 1
    return figstyle.PALETTE[idx % len(figstyle.PALETTE)]


def plot_act_depth_grid(results_root: Path, out: Path) -> None:
    """Exact accuracy vs. max ACT steps (one line per model) — the Ren depth stress test."""
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    plotted = False
    for model in MODELS:
        for label, suite_path in _find_suites(results_root, model):
            suite = _load(suite_path)
            final = suite.get("checkpoints", {}).get("final") or {}
            rows = (final.get("act_depth_ablation") or {}).get("rows") or []
            if not rows:
                continue
            depths = [r["max_steps"] for r in rows]
            acc = [r["exact_accuracy"] for r in rows]
            ax.plot(depths, acc, "o-", color=_model_color(model), label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Max ACT steps")
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.savefig(out / "fig_ren_act_depth.pdf")
    plt.close(fig)


def _seed_fp_rate(seed: Dict[str, Any]) -> float:
    """Fixed-point violation rate, read from the report seed or its suite file."""
    if seed.get("fixed_point_violation_rate") is not None:
        return float(seed["fixed_point_violation_rate"])
    ren_path = Path(seed.get("run_dir", "")) / "analysis/ren/ren_checkpoint_suite.json"
    suite = _load(ren_path)
    final = suite.get("checkpoints", {}).get("final") or {}
    return float((final.get("hrm") or {}).get("fixed_point_violation_rate", 0.0))


def plot_fp_violation_bars(report: Dict[str, Any], out: Path) -> None:
    """Per-seed fixed-point violation rate, grouped/coloured by model."""
    xs, vals, colors, ticklabels = [], [], [], []
    legend_models: List[str] = []
    x = 0
    for model, payload in report.get("models", {}).items():
        seeds = payload.get("seeds", [])
        if not seeds:
            continue
        legend_models.append(model)
        for i, seed in enumerate(seeds):
            if seed.get("ren_applies") is False and seed.get("ren_null_on_minimal"):
                # minimal TRM legitimately has no ACT loop: report as 0, keep the bar.
                pass
            xs.append(x)
            vals.append(_seed_fp_rate(seed))
            colors.append(_model_color(model))
            ticklabels.append(f"s{i}")
            x += 1
        x += 0.8  # gap between model groups
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.bar(xs, vals, color=colors, width=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticklabels, fontsize=7)
    ax.set_ylabel("Fixed-point violation rate")
    ax.set_ylim(0, 1.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_model_color(m)) for m in legend_models]
    ax.legend(handles, legend_models, ncol=min(3, len(legend_models)), loc="upper left")
    fig.savefig(out / "fig_ren_fp_violation.pdf")
    plt.close(fig)


def plot_mechanism_summary(report: Dict[str, Any], out: Path) -> None:
    """Count of the primary post-grokking mechanism across all seeds."""
    counts: Dict[str, int] = {}
    for payload in report.get("models", {}).values():
        for seed in payload.get("seeds", []):
            primary = seed.get("primary_mechanism", "unknown")
            counts[primary] = counts.get(primary, 0) + 1
    if not counts:
        return
    labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))
    colors = [figstyle.PALETTE[i % len(figstyle.PALETTE)] for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(labels, values, color=colors, width=0.7)
    ax.set_ylabel("Seed count")
    ax.tick_params(axis="x", rotation=18)
    fig.savefig(out / "fig_ren_mechanism_summary.pdf")
    plt.close(fig)


def plot_latent_pca_sample(results_root: Path, out: Path, *, model: str = "trm_full_b") -> None:
    suites = _find_suites(results_root, model)
    if not suites:
        return
    suite = _load(suites[0][1])
    final = suite.get("checkpoints", {}).get("final") or {}
    pca = (final.get("hrm") or {}).get("latent_pca_sample0") or []
    if len(pca) < 2:
        return
    xs, ys = zip(*pca)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(xs, ys, "o-", color=figstyle.PALETTE[3])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out / "fig_latent_pca_trajectory.pdf")
    plt.close(fig)


def _find_report(root: Path) -> Dict[str, Any]:
    for cand in (root / "ren_diagnosis_report.json", root / "aggregate" / "ren_diagnosis_report.json"):
        if cand.exists():
            return _load(cand)
    return {}


def run(args: argparse.Namespace) -> None:
    figstyle.apply_style()
    root = Path(args.results_root)
    out = ensure_dir(Path(args.figures_dir))
    report = _find_report(root)
    plot_act_depth_grid(root, out)
    if report:
        plot_fp_violation_bars(report, out)
        plot_mechanism_summary(report, out)
    plot_latent_pca_sample(root, out)
    print("ren figures ->", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="bmi_hybrid")
    p.add_argument("--figures-dir", default="bmi_hybrid/aggregate/figures")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
