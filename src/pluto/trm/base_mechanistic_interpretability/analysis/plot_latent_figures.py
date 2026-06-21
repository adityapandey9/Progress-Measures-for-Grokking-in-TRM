#!/usr/bin/env python3
"""Latent-contrast figures: per-step FVE, latent progress vs training, causal ablation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir


def _load(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text()) if p.exists() else {}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Skip tight_layout when the figure already uses constrained_layout to
    # avoid the "incompatible layout engine" warning.
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_latent_step_fve(full_json: Path, minimal_json: Path, out: Path) -> None:
    """Contrast the minimal TRM's single clean latent readout against the
    full TRM's per-ACT-step readout, which collapses to ~0 at every step.

    A grouped bar chart is used (rather than a line plot) so the single
    minimal data point and the flat-at-zero full-TRM cluster both read clearly.
    """
    full = _load(full_json).get("per_step", [])
    mini = _load(minimal_json).get("per_step", [])

    labels: list[str] = []
    heights: list[float] = []
    colors: list[str] = []
    for s in mini:  # Minimal TRM: single effective ACT step.
        labels.append("Minimal\n(1 step)")
        heights.append(max(float(s.get("logit_trig_fve_adaptive", 0.0)), 0.0))
        colors.append("#2563eb")
    n_minimal = len(heights)
    for s in full:  # Full TRM: one bar per ACT step.
        labels.append(f"Full\ns{s['act_step']}")
        heights.append(max(float(s.get("logit_trig_fve_adaptive", 0.0)), 0.0))
        colors.append("#dc2626")

    # Visual gap between the minimal bar and the full-TRM cluster.
    xs = [i if i < n_minimal else i + 0.8 for i in range(len(heights))]

    # Wider figure + constrained_layout give headroom for the value label
    # on the minimal bar and prevent annotation text from overlapping bars.
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.bar(xs, heights, width=0.7, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.95, color="gray", ls="--", lw=0.9)
    if xs:
        ax.text(xs[-1], 0.96, "Nanda-clean threshold (0.95)",
                ha="right", va="bottom", fontsize=7, color="gray")
        for xi, h, c in zip(xs, heights, colors):
            if c == "#2563eb":  # value label on the tall minimal bar
                ax.text(xi, h + 0.03, f"{h:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#1e40af")
        full_xs = [xi for xi, c in zip(xs, colors) if c == "#dc2626"]
        if full_xs:
            cx = sum(full_xs) / len(full_xs)
            # Annotate well above the bars (y=0.58) so text does not
            # overlap the bars sitting near zero.
            ax.annotate("Full TRM: $\\approx 0$ at all ACT steps\n(latent readout collapsed)",
                        xy=(cx, 0.03), xytext=(cx, 0.58), ha="center",
                        fontsize=8, color="#b91c1c",
                        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.9))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Latent readout trig-FVE (adaptive)")
    # Raise ceiling to 1.15 so the value annotation above the minimal bar
    # (height ≈ 1.0 + 0.03) has clear headroom.
    ax.set_ylim(0, 1.15)
    ax.legend(
        handles=[
            Patch(facecolor="#2563eb", edgecolor="black", label="Minimal TRM (logit-visible circuit)"),
            Patch(facecolor="#dc2626", edgecolor="black", label="Full TRM (latent ACT loop)"),
        ],
        fontsize=8, loc="upper right",
    )
    _save(fig, out / "fig_latent_step_fve.pdf")


def plot_latent_progress_vs_training(traj_json: Path, out: Path) -> None:
    t = _load(traj_json).get("trajectory", [])
    # Drop any sentinel "final" step (older runs encoded it as 1e9) so the
    # log-x axis is not stretched past the real training horizon.
    t = [x for x in t if int(x.get("step", 0)) < 10**8]
    if not t:
        return
    steps = [x["step"] for x in t]
    # Trim height and use constrained_layout to reduce excess whitespace.
    fig, ax = plt.subplots(figsize=(6, 3.0), constrained_layout=True)
    ax.plot(steps, [x["latent_progress"] for x in t], "o-", color="#059669", label="Latent progress (peak FVE)")
    accs = [x.get("test_acc") for x in t]
    if any(a is not None for a in accs):
        ax.plot(steps, [a if a is not None else float("nan") for a in accs], "s--",
                color="#7c3aed", label="Test accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log)")
    ax.set_ylabel("Measure")
    ax.set_ylim(0, 1.05)
    ax.margins(x=0.01)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, out / "fig_latent_progress_vs_training.pdf")


def plot_latent_causal(full_json: Path, out: Path, modulus: int = 113) -> None:
    full = _load(full_json).get("per_step", [])
    if not full:
        return
    steps = [s["act_step"] for s in full]
    full_ce = [float(s.get("full_loss_test", 0.0)) for s in full]
    abl_ce = [float(s.get("ablate_key_freqs_test_ce", s.get("full_loss_test", 0.0))) for s in full]
    x = list(range(len(steps)))
    width = 0.4
    # constrained_layout + wider figure prevent legend from colliding with
    # the chance reference line.
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    ax.bar([i - width / 2 for i in x], full_ce, width, color="#2563eb", label="Readout test CE")
    ax.bar([i + width / 2 for i in x], abl_ce, width, color="#dc2626", label="After key-freq ablation")
    chance = math.log(modulus)
    ax.axhline(chance, color="gray", ls="--", lw=0.9)
    # Label the chance line on the right margin so it does not overlap bars.
    ax.text(len(x) - 0.5, chance + 0.05, f"Chance ($\\log P={chance:.2f}$)",
            ha="right", va="bottom", fontsize=7, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.set_xlabel("ACT step")
    ax.set_ylabel("Per-step logit-readout test CE")
    # Legend outside the bars (lower right) where y-axis is clear.
    ax.legend(fontsize=8, loc="lower right")
    # The readout sits at/above chance every step and ablation leaves it
    # unchanged (degenerate key set): no logit-level circuit to remove.
    _save(fig, out / "fig_latent_causal_ablation.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-json", required=True)
    ap.add_argument("--minimal-json", required=True)
    ap.add_argument("--trajectory-json", required=True)
    ap.add_argument("--figures-dir", required=True)
    args = ap.parse_args()
    out = ensure_dir(Path(args.figures_dir))
    plot_latent_step_fve(Path(args.full_json), Path(args.minimal_json), out)
    plot_latent_progress_vs_training(Path(args.trajectory_json), out)
    plot_latent_causal(Path(args.full_json), out)
    print(f"latent figures -> {out}")


if __name__ == "__main__":
    main()
