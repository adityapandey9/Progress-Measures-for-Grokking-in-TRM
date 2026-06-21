#!/usr/bin/env python3
"""Generate hybrid paper figures from aggregate metrics and per-run JSON artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pluto.trm.base_mechanistic_interpretability.analysis.checkpoint_selection import _fve
from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _parse_step(step: Any) -> int:
    if step == "final":
        return 10**6
    return int(step)


def plot_model_comparison(metrics: Dict[str, Any], out: Path) -> None:
    names = list(metrics.get("models", {}).keys())
    fve_final = [metrics["models"][n].get("final_logit_trig_fve_faithful", metrics["models"][n].get("logit_trig_fve_faithful", {})).get("mean", 0.0) for n in names]
    fve_best = [metrics["models"][n].get("best_fve_logit_trig_fve_faithful", {}).get("mean", 0.0) for n in names]
    acc = [metrics["models"][n].get("test_accuracy", {}).get("mean", 0.0) for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(x, acc, color="#3b82f6")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=15, ha="right")
    ax[0].set_ylim(0, 1.05)
    ax[1].bar(x - w / 2, fve_final, w, label="Final FVE", color="#94a3b8")
    ax[1].bar(x + w / 2, fve_best, w, label="Best-FVE checkpoint", color="#22c55e")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=15, ha="right")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend()
    _save(fig, out / "fig_hybrid_model_comparison.pdf")


def plot_placeholder_ladder(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    for x, text, color in [
        (0.1, "Nanda 1-layer\ncalibration", "#dbeafe"),
        (0.5, "TRM minimal\nmain proof", "#dcfce7"),
        (0.9, "TRM full\nlatent extension", "#fee2e2"),
    ]:
        ax.text(x, 0.5, text, ha="center", va="center", bbox=dict(boxstyle="round", fc=color))
    ax.annotate("", xy=(0.39, 0.5), xytext=(0.21, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.79, 0.5), xytext=(0.61, 0.5), arrowprops=dict(arrowstyle="->"))
    _save(fig, out / "fig_hybrid_model_ladder.pdf")


def _representative_run(results_root: Path, model: str, seed: str = "seed_0") -> Optional[Path]:
    p = results_root / model / seed
    return p if p.exists() else None


def plot_phase_combined(results_root: Path, out: Path, *, model: str = "trm_minimal", seed: str = "seed_0") -> None:
    """Nanda Fig 7-style: sq weight norm + trig/excluded loss + Gini vs step."""
    run = _representative_run(results_root, model, seed)
    if not run:
        return
    history = _load(run / "training_history.json")
    if not isinstance(history, list):
        return
    rows = sorted([r for r in history if str(r.get("step", "")).isdigit()], key=lambda r: int(r["step"]))
    steps = [int(r["step"]) for r in rows]
    trig = [float(r.get("trig_loss_test", 0)) for r in rows]
    excluded = [float(r.get("excluded_loss_test", 0)) for r in rows]
    gini_e = [float(r.get("embedding_gini", 0)) for r in rows]
    gini_u = [float(r.get("unembed_gini", 0)) for r in rows]
    fve = [_fve(r) for r in rows]

    wn = _load(run / "analysis/weight_norms/weight_norms.json")
    wn_by_step: Dict[int, float] = {}
    for row in wn.get("weight_norms", []):
        st = _parse_step(row.get("step", -1))
        wn_by_step[st] = float(row.get("total_sq_norm", 0))
    total_norm = [wn_by_step.get(s, np.nan) for s in steps]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(steps, total_norm, "k-", lw=1.5)
    axes[0, 0].set_ylabel("Total sq. weight norm")
    axes[0, 1].plot(steps, trig, label="Trig (restricted)", color="#2563eb")
    axes[0, 1].plot(steps, excluded, label="Excluded", color="#dc2626")
    axes[0, 1].set_ylabel("Test CE")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(steps, gini_e, label="Embedding Gini", color="#059669")
    axes[1, 0].plot(steps, gini_u, label="Unembed Gini", color="#7c3aed")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 0].set_ylabel("Gini")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(steps, fve, color="#ea580c", lw=1.5)
    axes[1, 1].set_xlabel("Training step")
    axes[1, 1].set_ylabel("Faithful trig-FVE")
    axes[1, 1].set_ylim(0, 1.05)
    _save(fig, out / "fig_phase_combined.pdf")


def plot_multi_seed(metrics: Dict[str, Any], out: Path) -> None:
    models = metrics.get("models", {})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    rng = np.random.default_rng(0)
    for model_name, payload in models.items():
        seeds = payload.get("seeds", [])
        xs, err, fves = [], [], []
        for s in seeds:
            m = re.search(r"(\d+)$", s.get("seed", ""))
            if not m:
                continue
            acc = s.get("final_metrics", {}).get("test_accuracy", 0.0)
            xs.append(int(m.group(1)) + rng.uniform(-0.12, 0.12))
            err.append(max(1.0 - float(acc), 1e-4))  # error rate, log-friendly
            fves.append(
                s.get("history_metrics", {}).get("best_fve", {}).get("logit_trig_fve_faithful", 0.0)
            )
        if not xs:
            continue
        axes[0].plot(xs, err, "o", label=model_name, ms=5, alpha=0.85)
        axes[1].plot(xs, fves, "s", label=model_name, ms=5, alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Final test error (1 - acc)")
    axes[0].legend()
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Best-FVE faithful FVE")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    _save(fig, out / "fig_multi_seed.pdf")


def plot_per_seed_grokking(results_root: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    model_order = ["nanda_a_mlp", "trm_minimal", "trm_full_b", "trm_full_a"]
    for ax, model in zip(axes, model_order):
        for seed_dir in sorted((results_root / model).glob("seed_*")):
            hist = _load(seed_dir / "training_history.json")
            if not isinstance(hist, list):
                continue
            rows = sorted([r for r in hist if str(r.get("step", "")).isdigit()], key=lambda r: int(r["step"]))
            steps = [int(r["step"]) for r in rows]
            acc = [float(r.get("test_acc", 0)) for r in rows]
            ax.plot(steps, acc, alpha=0.7, lw=1, label=seed_dir.name.replace("seed_", "s"))
        ax.set_xlabel("Step")
        ax.set_ylabel("Test acc")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, ncol=2)
    _save(fig, out / "fig_hybrid_multiseed_grokking.pdf")


def plot_neuron_histogram(results_root: Path, out: Path, *, model: str = "trm_minimal", seed: str = "seed_0") -> None:
    run = _representative_run(results_root, model, seed)
    if not run:
        return
    data = _load(run / "analysis/neuron_tables/neuron_tables.json")
    rows = data.get("rows", [])
    if not rows:
        return
    freqs = [int(r["best_frequency"]) for r in rows if int(r.get("best_frequency", -1)) >= 0]
    r2s = [float(r.get("best_r2", 0)) for r in rows]
    counts = Counter(freqs)
    top = counts.most_common(15)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    if top:
        labels, vals = zip(*top)
        axes[0].bar([str(l) for l in labels], vals, color="#6366f1")
        axes[0].set_yscale("log")
        axes[0].set_ylim(bottom=0.8)
        axes[0].set_xlabel("Dominant frequency k")
        axes[0].set_ylabel("Neuron count")
    axes[1].hist(r2s, bins=30, color="#14b8a6", edgecolor="white")
    axes[1].set_xlabel("Best $R^2$")
    axes[1].set_ylabel("Count")
    axes[1].axvline(0.5, color="red", ls="--", lw=1, label="$R^2=0.5$")
    axes[1].legend()
    _save(fig, out / "fig_neuron_frequency_tables.pdf")


def plot_all_frequency_grid(results_root: Path, out: Path) -> None:
    for af_path in sorted(results_root.glob("trm_minimal/seed_0/analysis/all_frequency_ablations/all_frequency_ablations.json")):
        payload = _load(af_path)
        rows = payload.get("rows", [])
        if not rows:
            continue
        freqs = [int(r["frequency"]) for r in rows]
        ablate = [r["ablate_loss_test"] for r in rows]
        retain = [r["retain_loss_test"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(freqs, ablate, label="Ablate (excluded) CE")
        ax.plot(freqs, retain, label="Retain (trig) CE")
        ax.set_xlabel("Frequency k")
        ax.set_ylabel("Test CE")
        ax.legend()
        _save(fig, out / "fig_all_frequency_ablations.pdf")
        break


def plot_weight_norm_phases(results_root: Path, out: Path, *, model: str = "trm_minimal", seed: str = "seed_0") -> None:
    run = _representative_run(results_root, model, seed)
    if not run:
        return
    payload = _load(run / "analysis/weight_norms/weight_norms.json")
    rows = payload.get("weight_norms", [])
    if not rows:
        return
    steps = [_parse_step(r["step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in [
        ("mlp_sq_norm", "MLP"),
        ("embedding_sq_norm", "Embedding"),
        ("unembedding_sq_norm", "Unembed"),
        ("total_sq_norm", "Total"),
    ]:
        ax.plot(steps, [r.get(key, 0) for r in rows], label=label)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Squared weight norm")
    ax.legend()
    _save(fig, out / "fig_weight_norm_phases.pdf")


def run(args: argparse.Namespace) -> None:
    metrics = _load(Path(args.metrics))
    out = ensure_dir(Path(args.figures_dir))
    results_root = Path(args.results_root) if args.results_root else Path(metrics.get("results_root", "."))
    plot_placeholder_ladder(out)
    plot_model_comparison(metrics, out)
    plot_phase_combined(results_root, out)
    plot_multi_seed(metrics, out)
    plot_per_seed_grokking(results_root, out)
    plot_neuron_histogram(results_root, out)
    plot_all_frequency_grid(results_root, out)
    plot_weight_norm_phases(results_root, out)
    print(f"hybrid figures -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", required=True)
    p.add_argument("--figures-dir", default="bmi_hybrid/aggregate/figures")
    p.add_argument("--results-root", default="")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
