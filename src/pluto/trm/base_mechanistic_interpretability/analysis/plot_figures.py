#!/usr/bin/env python3
"""Generate Nanda-comparable figures for latent grokking mechanistic interpretability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def plot_grokking_curves(history_path: Path, out_dir: Path) -> None:
    if not history_path.exists():
        return
    hist = json.loads(history_path.read_text())
    steps = [h["step"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, [h["train_acc"] for h in hist], label="train", color="#1f77b4")
    ax[0].plot(steps, [h["test_acc"] for h in hist], label="test", color="#ff7f0e")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].set_ylim(-0.05, 1.05)
    ax[1].plot(steps, [h["train_loss"] for h in hist], label="train", color="#1f77b4")
    ax[1].plot(steps, [h["test_loss"] for h in hist], label="test", color="#ff7f0e")
    if hist and "trig_loss_test" in hist[0]:
        ax[1].plot(
            [h["step"] for h in hist if "trig_loss_test" in h],
            [h["trig_loss_test"] for h in hist if "trig_loss_test" in h],
            label="trig (test)",
            color="#2ca02c",
            linestyle="--",
        )
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_grokking_curves.pdf")
    plt.close(fig)


def plot_progress(pm: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    emb = pm.get("embedding_fourier_norms", [])[:56]
    un = pm.get("unembed_fourier_norms", [])[:56]
    axes[0, 0].bar(range(len(emb)), emb, color="#446688", width=1.0)
    axes[0, 0].set_xlabel("Frequency component")
    axes[0, 1].bar(range(len(un)), un, color="#884466", width=1.0)
    axes[0, 1].set_xlabel("Frequency component")
    labels = ["full", "trig", "excluded"]
    vals = [
        pm.get("test_loss", pm.get("full_loss_test", 0)),
        pm.get("trig_loss_test", 0),
        pm.get("excluded_loss_test", 0),
    ]
    axes[1, 0].bar(labels, vals, color=["#333", "#2a2", "#a22"])
    axes[1, 0].set_ylabel("Test CE")
    per = pm.get("excluded_loss_test_per_freq", [])
    if per:
        axes[1, 1].plot(range(len(per)), per, marker="o", color="#c44")
        axes[1, 1].set_xlabel("Key freq removed (sequential)")
        axes[1, 1].set_ylabel("Test CE after exclusion")
    else:
        axes[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_progress_measures.pdf")
    plt.close(fig)


def plot_trajectory(traj: dict, out_dir: Path) -> None:
    rows = traj.get("trajectory", [])
    if not rows:
        return
    steps: List[int] = []
    for r in rows:
        s = r.get("step", "0")
        steps.append(99999 if s == "final" else int(s))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(steps, [r["trig_loss_test"] for r in rows], marker="o", label="trig")
    ax[0].plot(steps, [r["excluded_loss_test"] for r in rows], marker="s", label="excluded")
    ax[0].plot(steps, [r["full_loss_test"] for r in rows], marker=".", label="full", alpha=0.6)
    ax[0].set_xlabel("Checkpoint step")
    ax[0].set_ylabel("Test CE")
    ax[0].legend()
    ax[1].plot(steps, [r["embedding_gini"] for r in rows], marker="o", label="W_E")
    ax[1].plot(steps, [r["unembed_gini"] for r in rows], marker="s", label="W_U")
    ax[1].set_xlabel("Checkpoint step")
    ax[1].set_ylabel("Gini coefficient")
    ax[1].legend()
    ax[2].plot(steps, [r.get("logit_trig_fve_mean", 0) for r in rows], marker="o", color="#636")
    ax[2].set_xlabel("Checkpoint step")
    ax[2].set_ylabel("Mean FVE")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_progress_trajectory.pdf")
    plt.close(fig)


def plot_ablations(ab: dict, out_dir: Path) -> None:
    if not ab:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["full", "trig (key)", "excluded (key)", "trig (non-key)"]
    vals = [
        ab.get("full_loss_test", 0),
        ab.get("trig_loss_key_test", 0),
        ab.get("excluded_loss_key_test", 0),
        ab.get("trig_loss_nonkey_test", 0),
    ]
    colors = ["#333", "#2a2", "#a22", "#888"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Test CE")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_frequency_ablations.pdf")
    plt.close(fig)


def plot_reasoning(hrm: dict, out_dir: Path) -> None:
    ce = hrm.get("mean_field_ce_by_act_step", [])
    exact = hrm.get("mean_field_exact_by_act_step", [])
    if not ce and not exact:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    if ce:
        ax[0].plot(list(range(len(ce))), ce, marker="o")
    ax[0].set_xlabel("ACT step")
    ax[0].set_ylabel("Mean-field CE")

    if exact:
        ax[1].plot(list(range(len(exact))), exact, marker="s", color="#d62728")
    ax[1].set_xlabel("ACT step")
    ax[1].set_ylabel("Fraction correct")
    ax[1].set_ylim(0, 1.02)
    fpv = hrm.get("fixed_point_violation_rate")
    if fpv is not None:
        ax[1].annotate(
            f"fixed-point violation = {float(fpv):.2f}",
            xy=(0.5, 0.08), xycoords="axes fraction", ha="center", fontsize=7,
        )
    fig.savefig(out_dir / "fig_latent_reasoning_modes.pdf")
    plt.close(fig)

    pca = hrm.get("latent_pca_sample0", [])
    if len(pca) > 1:
        fig2, ax2 = plt.subplots(figsize=(3.6, 3.0))
        ax2.plot([p[0] for p in pca], [p[1] for p in pca], marker=".")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        fig2.savefig(out_dir / "fig_latent_pca_trajectory.pdf")
        plt.close(fig2)


def plot_mechanistic(mech: dict, out_dir: Path) -> None:
    if not mech:
        return
    tables = mech.get("tables", {})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    emb = tables.get("embedding_fourier_norms_top10", {})
    un = tables.get("unembed_fourier_norms_top10", {})
    if emb:
        ks = sorted(emb.keys(), key=int)
        ax[0].bar([int(k) for k in ks], [emb[k] for k in ks], color="#446")
    if un:
        ks = sorted(un.keys(), key=int)
        ax[1].bar([int(k) for k in ks], [un[k] for k in ks], color="#644")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mechanistic_circuit.pdf")
    plt.close(fig)


def plot_reverse_engineering(re: dict, out_dir: Path) -> None:
    if not re:
        return
    wl = re.get("W_L_analysis", {}).get("W_L_final", {})
    norms = wl.get("fourier_norms", [])[:56]
    if norms:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # Drop the DC component (index 0) and use log-y so the secondary
        # frequencies are visible instead of being flattened by one spike.
        comp = list(range(1, len(norms)))
        vals = [max(float(v), 1e-6) for v in norms[1:]]
        ax[0].bar(comp, vals, width=1.0)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Frequency component (DC dropped)")
        ax[0].set_ylabel("$W_L$ Fourier norm (log)")
        clusters = re.get("mlp_neuron_clustering", {})
        layer_key = "layer1_mlp_act" if "layer1_mlp_act" in clusters else next(iter(clusters), None)
        if layer_key:
            per = clusters[layer_key].get("neurons_per_freq", {})
            if per:
                ax[1].bar(list(per.keys()), list(per.values()), color="#885533")
                ax[1].set_xlabel("Frequency k")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_wl_mlp_reverse_engineering.pdf")
        plt.close(fig)

    attn = re.get("attention_decomposition", {}).get("layers", [])
    if attn:
        fig, ax = plt.subplots(figsize=(6, 4))
        layer0 = attn[0]
        heads_a = layer0.get("heads_attn_to_a", [])
        heads_b = layer0.get("heads_attn_to_b", [])
        x = np.arange(len(heads_a))
        w = 0.35
        ax.bar(x - w / 2, heads_a, w, label="a", color="#447")
        ax.bar(x + w / 2, heads_b, w, label="b", color="#744")
        ax.set_xlabel("Head")
        ax.set_ylabel("Mean attention from '='")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig_attention_decomposition.pdf")
        plt.close(fig)


def plot_multi_seed(ms: dict, out_dir: Path) -> None:
    seeds = ms.get("seeds", [])
    if not seeds:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(range(len(seeds)), [s["final_test_acc"] for s in seeds], color="#336699")
    ax[0].set_xlabel("Seed")
    ax[0].set_ylabel("Final test accuracy")
    grok_steps = [s.get("grokking_step") or 0 for s in seeds]
    ax[1].bar(range(len(seeds)), grok_steps, color="#996633")
    ax[1].set_xlabel("Seed")
    ax[1].set_ylabel("Step test acc > 90%")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_multi_seed.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", default="bmi_analysis")
    p.add_argument("--figures-dir", default="bmi_analysis/figures")
    p.add_argument("--training-history", default="bmi_grokking_runs/default/training_history.json")
    args = p.parse_args()
    analysis = Path(args.analysis_dir)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_grokking_curves(Path(args.training_history), fig_dir)
    pm = _load_json(analysis / "progress" / "progress_measures_grokking.json")
    if pm:
        plot_progress(pm, fig_dir)
    traj = _load_json(analysis / "trajectory" / "progress_trajectory.json")
    if traj:
        plot_trajectory(traj, fig_dir)
    ab = _load_json(analysis / "ablations" / "frequency_ablations.json")
    if ab:
        plot_ablations(ab, fig_dir)
    mech = _load_json(analysis / "mechanistic" / "mechanistic_circuit.json")
    if mech:
        plot_mechanistic(mech, fig_dir)
    hrm = _load_json(analysis / "reasoning" / "latent_reasoning_probes.json")
    if hrm:
        plot_reasoning(hrm, fig_dir)
    re_ = _load_json(analysis / "reverse_engineering" / "reverse_engineering.json")
    if re_:
        plot_reverse_engineering(re_, fig_dir)
    ms = _load_json(analysis / "multi_seed" / "multi_seed_aggregate.json")
    if ms:
        plot_multi_seed(ms, fig_dir)
    print(f"Figures -> {fig_dir}")


if __name__ == "__main__":
    main()
