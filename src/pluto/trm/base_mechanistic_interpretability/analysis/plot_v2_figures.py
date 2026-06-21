"""Paper v2 figure generators (Nanda-style, Arc A)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from pluto.trm.base_mechanistic_interpretability.analysis.checkpoint_selection import _fve
from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir
from pluto.trm.base_mechanistic_interpretability.analysis import plot_figures as pf


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


def _ema(xs: List[float], alpha: float = 0.3) -> List[float]:
    out: List[float] = []
    acc: Optional[float] = None
    for x in xs:
        acc = x if acc is None else alpha * x + (1 - alpha) * acc
        out.append(acc)
    return out


def _mainline_run(metrics: Dict[str, Any], results_root: Path) -> Optional[Path]:
    trm = metrics.get("models", {}).get("trm_minimal", {})
    ml = trm.get("mainline_seed_run", trm.get("best_seed_run", {}))
    run_dir = ml.get("run_dir")
    if run_dir and Path(run_dir).exists():
        return Path(run_dir)
    seed = ml.get("seed", "seed_0")
    for candidate in [
        results_root / "ep1" / f"wd_{metrics.get('best_weight_decay', 1.0)}" / "trm_minimal" / seed,
        results_root / "trm_minimal" / seed,
    ]:
        if candidate.exists():
            return candidate
    return None


def _seed_with_artifact(run_dir: Path, rel: str) -> Path:
    """Return ``run_dir`` if it holds ``rel``, else a sibling ``seed_*`` that does.

    The dense trajectory / weight-norm analyses are only computed for the
    analyzed seed, which need not be the aggregator's mainline (highest-FVE)
    seed. Falling back keeps the progress-measure figure from rendering empty
    panels when the mainline seed lacks those artifacts.
    """
    if (run_dir / rel).exists():
        return run_dir
    for sib in sorted(run_dir.parent.glob("seed_*")):
        if (sib / rel).exists():
            return sib
    return run_dir


def plot_v2_algorithm_schematic(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Fill the axes area with the schematic content so tight_layout
    # preserves the full figure height.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08)
    steps = [
        (0.08, "Embed $a,b$\n(sparse Fourier $W_E$)", "#dbeafe"),
        (0.30, "Attention\nroute '=' $\\to$ $a,b$", "#e0e7ff"),
        (0.52, "MLP SwiGLU\n$\\cos/\\sin(w_k(a{+}b))$", "#dcfce7"),
        (0.74, "Unembed $W_U$\n$\\cos(w_k(a{+}b{-}c))$", "#fef3c7"),
        (0.94, "Logits", "#f3f4f6"),
    ]
    for x, text, color in steps:
        ax.text(
            x, 0.5, text, ha="center", va="center", fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="#aaaaaa", lw=0.8),
        )
    for x0, x1 in [(0.16, 0.22), (0.38, 0.44), (0.60, 0.66), (0.82, 0.88)]:
        ax.annotate(
            "", xy=(x1, 0.5), xytext=(x0, 0.5),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
    # Save without tight_layout (subplots_adjust already positions content)
    out_path = out / "fig_algorithm_schematic.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches=None)
    plt.close(fig)


def plot_v2_grokking(results_root: Path, metrics: Dict[str, Any], out: Path) -> None:
    trm = metrics.get("models", {}).get("trm_minimal", {})
    wd = metrics.get("best_weight_decay", 1.0)
    model_root = results_root / "ep1" / f"wd_{wd}" / "trm_minimal"
    if not model_root.exists():
        model_root = results_root / "trm_minimal"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (ylabel, key) in zip(axes.flatten(), [("Train acc", "train_acc"), ("Test acc", "test_acc"), ("Train loss", "train_loss"), ("Test loss", "test_loss")]):
        for seed_dir in sorted(model_root.glob("seed_*")):
            hist = _load(seed_dir / "training_history.json")
            if not isinstance(hist, list):
                continue
            rows = sorted([r for r in hist if str(r.get("step", "")).isdigit()], key=lambda r: int(r["step"]))
            steps = [max(int(r["step"]), 1) for r in rows]
            vals = _ema([float(r.get(key, 0)) for r in rows], alpha=0.25)
            ax.plot(steps, vals, lw=1.2, alpha=0.85, label=seed_dir.name.replace("seed_", "s"))
        ax.set_xscale("log")
        ax.set_xlabel("Step (log)")
        ax.set_ylabel(ylabel)
        if key.endswith("acc"):
            ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, ncol=2)
    _save(fig, out / "fig_grokking_curves.pdf")


def plot_v2_we_fourier(re: Dict[str, Any], out: Path) -> None:
    emb = re.get("key_frequencies_embedding", [])
    wl = re.get("W_L_analysis", {}).get("W_L_final", {})
    norms = wl.get("fourier_norms", [])
    if not norms:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    comp = list(range(1, min(len(norms), 56)))
    vals = [max(float(norms[i]), 1e-6) for i in comp]
    ax.bar(comp, vals, width=1.0, color="#3b82f6")
    ax.set_yscale("log")
    ax.set_xlabel("Fourier component (DC dropped)")
    ax.set_ylabel("$W_E$ / embedding energy (log)")
    ax.set_title(f"Key freqs (progress): {emb}")
    _save(fig, out / "fig_embedding_fourier.pdf")


def plot_v2_wl_fourier(re: Dict[str, Any], out: Path) -> None:
    wl = re.get("W_L_analysis", {}).get("W_L_final", {})
    norms = wl.get("fourier_norms", [])
    if not norms:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    comp = list(range(1, min(len(norms), 56)))
    vals = [max(float(norms[i]), 1e-6) for i in comp]
    ax.bar(comp, vals, width=1.0, color="#6366f1")
    ax.set_yscale("log")
    ax.set_xlabel("Fourier component (DC dropped)")
    ax.set_ylabel("$W_L$ Fourier norm (log)")
    kf = wl.get("key_frequencies", [])
    ax.set_title(f"$W_L$ key frequencies: {kf}")
    _save(fig, out / "fig_wl_fourier.pdf")


def plot_v2_attention_neuron_heatmaps(re: Dict[str, Any], run_dir: Path, out: Path) -> None:
    clusters = re.get("mlp_neuron_clustering", {})
    layer_key = "layer1_mlp_act" if "layer1_mlp_act" in clusters else next(iter(clusters), None)
    attn = re.get("attention_decomposition", {}).get("layers", [])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    if layer_key:
        per = clusters[layer_key].get("neurons_per_freq", {})
        if per:
            keys = sorted(per.keys(), key=lambda x: int(x))
            axes[0].bar([str(k) for k in keys], [per[k] for k in keys], color="#14b8a6")
            axes[0].set_xlabel("Frequency k")
            axes[0].set_ylabel("Neuron count")
            axes[0].set_title("MLP neurons per frequency")
    if attn:
        layer0 = attn[0]
        heads_a = layer0.get("heads_attn_to_a", [])
        heads_b = layer0.get("heads_attn_to_b", [])
        x = np.arange(len(heads_a))
        w = 0.35
        axes[1].bar(x - w / 2, heads_a, w, label="attn to $a$", color="#4477aa")
        axes[1].bar(x + w / 2, heads_b, w, label="attn to $b$", color="#aa7744")
        axes[1].set_xlabel("Head")
        axes[1].set_ylabel("Mean attention from '='")
        axes[1].legend(fontsize=8)
    _save(fig, out / "fig_activations_periodic.pdf")


def plot_v2_neuron_variance_explained(run_dir: Path, out: Path) -> None:
    swiglu_path = run_dir / "analysis/swiglu_neurons/summary.json"
    rows: List[Dict[str, Any]] = []
    if swiglu_path.exists():
        summary = _load(swiglu_path)
        rows = summary.get("per_channel", [])
        if not rows:
            channels = _load(run_dir / "analysis/swiglu_neurons/channels.json")
            rows = channels.get("rows", [])
        r2s = [float(r.get("best_r2", 0)) for r in rows]
        frac85 = float(summary.get("fraction_ge_85", 0))
        title = f"SwiGLU channels with $R^2\\geq 0.85$: {frac85:.1%}"
    else:
        data = _load(run_dir / "analysis/neuron_tables/neuron_tables.json")
        rows = data.get("rows", [])
        if not rows:
            re_ = _load(run_dir / "analysis/reverse_engineering/reverse_engineering.json")
            clusters = re_.get("mlp_neuron_clustering", {})
            layer = next(iter(clusters.values()), {})
            rows = layer.get("neurons", [])
        r2s = [float(r.get("best_r2", 0)) for r in rows]
        frac85 = sum(1 for r in r2s if r >= 0.85) / max(len(r2s), 1)
        title = f"Post-SwiGLU neurons with $R^2\\geq 0.85$: {frac85:.1%} (distributed circuit)"
    if not r2s:
        return
    freqs = [int(r.get("best_frequency", -1)) for r in rows if int(r.get("best_frequency", -1)) >= 0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(r2s, bins=30, color="#14b8a6", edgecolor="white")
    axes[0].axvline(0.85, color="red", ls="--", lw=1, label="$R^2=0.85$")
    axes[0].set_xlabel("Variance explained ($R^2$)")
    axes[0].set_ylabel("Channel count")
    axes[0].legend()
    counts = Counter(freqs)
    if counts:
        labels, vals = zip(*counts.most_common(12))
        axes[1].bar([str(l) for l in labels], vals, color="#6366f1")
        axes[1].set_yscale("log")
        axes[1].set_ylim(bottom=0.8)
        axes[1].set_xlabel("Dominant frequency $k$")
        axes[1].set_ylabel("Count")
    fig.suptitle(title, fontsize=10, y=1.02)
    _save(fig, out / "fig_neuron_trig_fit.pdf")


def plot_v2_frequency_ablation_grid(run_dir: Path, out: Path, key_freqs: list[int]) -> None:
    # Fallback to a sibling seed that has the ablation artifact when the
    # mainline seed does not (artifacts are only computed for the analyzed seed).
    run_dir = _seed_with_artifact(run_dir, "analysis/all_frequency_ablations/all_frequency_ablations.json")
    path = run_dir / "analysis/all_frequency_ablations/all_frequency_ablations.json"
    if not path.exists():
        return
    data = _load(path)
    rows = data.get("rows", [])
    if not rows:
        return
    key_set = set(key_freqs)
    freqs = [int(r["frequency"]) for r in rows]
    losses = [float(r["ablate_loss_test"]) for r in rows]
    # Key-frequency bars: red with hatch; non-key: muted blue
    colors = ["#c0392b" if f in key_set else "#5b8db8" for f in freqs]
    hatches = ["//" if f in key_set else "" for f in freqs]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bars = ax.bar(freqs, losses, color=colors, width=0.8, edgecolor="white", linewidth=0.4)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.axhline(y=np.log(113), color="k", linestyle="--", lw=1.2, label="chance ($\\ln P$)")
    ax.set_xlabel("Frequency $k$", fontsize=12)
    ax.set_ylabel("Test CE after ablating freq $k$", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    # Rotate x-ticks so dense frequency indices don't overlap
    ax.set_xticks(freqs[::2] if len(freqs) > 30 else freqs)
    ax.tick_params(axis="x", rotation=45)
    # Legend outside data area (upper right, outside axes) to avoid overlap
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#c0392b", hatch="//", edgecolor="#888", label="Key frequency"),
        Patch(facecolor="#5b8db8", edgecolor="white", label="Non-key freq"),
        plt.Line2D([0], [0], color="k", linestyle="--", lw=1.2, label="Chance ($\\ln P$)"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="center right", framealpha=0.9)
    _save(fig, out / "fig_frequency_ablation_grid.pdf")


def plot_v2_key_freq_ablation(run_dir: Path, out: Path) -> None:
    ab = _load(run_dir / "analysis/ablations/frequency_ablations.json")
    if not ab:
        return
    pf.plot_ablations(ab, out)


def plot_v2_progress_phases(run_dir: Path, out: Path) -> None:
    # The progress-measure story needs the dense trajectory + weight-norm
    # analyses, which are only computed for the analyzed seed (not necessarily
    # the aggregator's highest-FVE mainline seed). Resolve to that seed so all
    # four panels come from one coherent run instead of rendering empty.
    run_dir = _seed_with_artifact(run_dir, "analysis/trajectory/progress_trajectory.json")
    history = _load(run_dir / "training_history.json")
    if not isinstance(history, list):
        return
    rows = sorted([r for r in history if str(r.get("step", "")).isdigit()], key=lambda r: int(r["step"]))
    steps = [max(int(r["step"]), 1) for r in rows]
    acc = [float(r.get("test_acc", 0)) for r in rows]

    wn_dir = _seed_with_artifact(run_dir, "analysis/weight_norms/weight_norms.json")
    wn = _load(wn_dir / "analysis/weight_norms/weight_norms.json")
    wn_rows = [w for w in wn.get("weight_norms", []) if str(w.get("step", "")).strip().isdigit()]
    wn_rows.sort(key=lambda w: int(w["step"]))
    wn_steps = [max(int(w["step"]), 1) for w in wn_rows]
    wn_vals = [float(w.get("total_sq_norm", 0.0)) for w in wn_rows]

    traj = _load(run_dir / "analysis/trajectory/progress_trajectory.json").get("trajectory", [])
    traj = [t for t in traj if str(t.get("step", "")).strip().isdigit()]
    traj.sort(key=lambda t: int(t["step"]))
    t_steps = [max(int(t["step"]), 1) for t in traj]
    trig = [float(t.get("trig_loss_test", 0.0)) for t in traj]
    excluded = [float(t.get("excluded_loss_test", 0.0)) for t in traj]
    fve = [float(t.get("logit_trig_fve_faithful_mean", t.get("logit_trig_fve_mean", 0.0))) for t in traj]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    if wn_steps:
        axes[0, 0].plot(wn_steps, wn_vals, "k-o", ms=3, lw=1.5)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Step (log)")
    axes[0, 0].set_ylabel("Total sq. weight norm")

    if t_steps:
        axes[0, 1].plot(t_steps, trig, "o-", ms=3, label="Restricted (trig)", color="#2563eb")
        axes[0, 1].plot(t_steps, excluded, "s-", ms=3, label="Excluded", color="#dc2626")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Step (log)")
    axes[0, 1].set_ylabel("Test CE (log)")
    axes[0, 1].legend(fontsize=8)

    if t_steps:
        axes[1, 0].plot(t_steps, fve, "o-", ms=3, color="#059669")
    axes[1, 0].axhline(0.95, color="gray", ls="--", lw=0.8)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xlabel("Step (log)")
    axes[1, 0].set_ylabel("Logit trig-FVE")

    axes[1, 1].plot(steps, _ema(acc, 0.25), color="#7c3aed", lw=1.5)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xlabel("Step (log)")
    axes[1, 1].set_ylabel("Test accuracy")

    grok_steps = [s for s, a in zip(steps, acc) if a >= 0.99]
    if grok_steps:
        for ax in axes.flat:
            ax.axvline(grok_steps[0], color="#94a3b8", ls=":", lw=1.0, alpha=0.8)
    for ax in axes.flat:
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _save(fig, out / "fig_progress_measures.pdf")


def plot_v2_multiseed(metrics: Dict[str, Any], out: Path) -> None:
    trm = metrics.get("models", {}).get("trm_minimal", {})
    seeds = trm.get("seeds", [])
    if not seeds:
        return
    _corr_by_seed = {cs.get("seed"): cs for cs in trm.get("corrected_fve", {}).get("seeds", [])}
    # Wider figure + constrained_layout so the external legend isn't clipped.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    xs, fve_final, fve_best, trig = [], [], [], []
    for s in seeds:
        m = re.search(r"(\d+)$", s.get("seed", ""))
        if not m:
            continue
        xs.append(int(m.group(1)))
        fve_final.append(
            float(_corr_by_seed.get(s.get("seed"), {}).get("fve_adaptive",
                  s.get("history_metrics", {}).get("final", {}).get("logit_trig_fve_adaptive", 0.0)))
        )
        fve_best.append(s.get("history_metrics", {}).get("best_fve", {}).get("logit_trig_fve_faithful", 0.0))
        trig.append(s.get("final_metrics", {}).get("trig_loss_test", 0.0))
    bar_w = 0.35
    axes[0].bar([x - bar_w / 2 for x in xs], fve_final, width=bar_w, color="#94a3b8", label="Final adaptive FVE")
    axes[0].bar([x + bar_w / 2 for x in xs], fve_best, width=bar_w, color="#22c55e", label="Best-FVE checkpoint")
    axes[0].axhline(0.95, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("FVE")
    axes[0].set_ylim(0, 1.12)
    # Legend outside the axes to avoid overlapping bars.
    axes[0].legend(
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    axes[1].plot(xs, trig, "o-", color="#2563eb")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Final restricted (trig) CE")
    # Save with bbox_inches='tight' so the external legend is not clipped.
    path = out / "fig_multiseed_summary.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_v2_fve_calibration(results_root: Path, metrics: Dict[str, Any], out: Path) -> None:
    """FVE vs.\\ number of key frequencies, one panel per model with checkpoints.

    ``torch`` and the curve helper are imported lazily so the JSON-only figure
    functions in this module remain importable on machines without torch.
    """
    from pluto.trm.base_mechanistic_interpretability.analysis.plot_fve_calibration import (
        KS,
        curve,
    )
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wd = metrics.get("best_weight_decay", 1.0)
    configs = [
        ("Nanda 1-layer (calibration)", "nanda", results_root / "nanda_a_mlp"),
        ("TRM minimal", "trm_minimal", results_root / "ep1" / f"wd_{wd}" / "trm_minimal"),
    ]
    # Render only models that actually have local final checkpoints so we never
    # reserve an empty panel (e.g. when the Nanda calibration run is absent).
    present = [
        (title, mtype, mroot)
        for title, mtype, mroot in configs
        if mroot.exists() and any((sd / "checkpoint_final.pt").exists() for sd in mroot.glob("seed_*"))
    ]
    if not present:
        return
    fig, axes = plt.subplots(
        1, len(present), figsize=(4.5 * len(present), 3.6), sharey=True, squeeze=False
    )
    row = axes[0]
    for ax, (title, mtype, mroot) in zip(row, present):
        for seed_dir in sorted(mroot.glob("seed_*")):
            ck = seed_dir / "checkpoint_final.pt"
            if not ck.exists():
                continue
            ys = curve(str(ck), mtype, device)
            ax.plot(KS, ys, marker="o", ms=3, lw=1.3, label=seed_dir.name.replace("seed_", "s"))
        ax.axhline(0.95, color="gray", ls="--", lw=0.8)
        ax.axvline(5, color="crimson", ls=":", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("# key frequencies $K$")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    row[0].set_ylabel("Faithful trig-FVE")
    _save(fig, out / "fig_fve_vs_k.pdf")


def plot_v2_data_fraction(metrics: Dict[str, Any], out: Path) -> None:
    rows = metrics.get("ep3_data_fraction_sweep", [])
    if not rows:
        return
    from collections import defaultdict

    by_frac: dict[float, list] = defaultdict(list)
    for r in rows:
        by_frac[float(r["frac_train"])].append(r)
    fracs = sorted(by_frac.keys())
    grok = [
        sum(int(g.get("grokking_step", -1)) for g in by_frac[f] if int(g.get("grokking_step", -1)) >= 0)
        / max(len([g for g in by_frac[f] if int(g.get("grokking_step", -1)) >= 0]), 1)
        for f in fracs
    ]
    fve = [sum(float(g.get("final_fve", 0)) for g in by_frac[f]) / len(by_frac[f]) for f in fracs]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(fracs, grok, "o-", color="#2563eb")
    axes[0].set_xlabel("Train data fraction")
    axes[0].set_ylabel("Grokking step")
    axes[1].plot(fracs, fve, "s-", color="#22c55e")
    axes[1].axhline(0.95, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("Train data fraction")
    axes[1].set_ylabel("Final adaptive FVE")
    axes[1].set_ylim(0, 1.05)
    _save(fig, out / "fig_data_fraction_sweep.pdf")


def plot_v2_weight_decay(metrics: Dict[str, Any], out: Path) -> None:
    # Combine the hi-prec ep3 sweep with the main lambda=1.0 hero result so the
    # panel shows a real lambda comparison (not a single bar) and stays consistent
    # with the headline 5/5 number. Fall back to the ep1 sweep only if neither is
    # available.
    ep3 = metrics.get("ep3_weight_decay_sweep", [])
    by_wd: Dict[float, List[float]] = defaultdict(list)
    for r in ep3:
        wd = float(str(r.get("weight_decay", r.get("tag", "0"))).replace("wd_", ""))
        by_wd[wd].append(float(r.get("final_fve", 0.0)))
    # point := (weight_decay, fraction_clean, mean_fve, n_seeds)
    points: List[tuple] = [
        (
            wd,
            sum(1 for f in by_wd[wd] if f >= 0.95) / len(by_wd[wd]),
            sum(by_wd[wd]) / len(by_wd[wd]),
            len(by_wd[wd]),
        )
        for wd in sorted(by_wd)
    ]
    main = metrics.get("corrected_fve", {}).get("trm_minimal", {})
    if main and not any(abs(wd - 1.0) < 1e-9 for wd, *_ in points):
        n = int(main.get("n_seeds", 0)) or 1
        points.append(
            (
                1.0,
                int(main.get("n_seeds_adaptive_ge_0.95", 0)) / n,
                float(main.get("mean_fve_adaptive", 0.0)),
                n,
            )
        )
    points.sort()
    if points:
        wds = [p[0] for p in points]
        frac = [p[1] for p in points]
        mean_fve = [p[2] for p in points]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        axes[0].bar([str(w) for w in wds], frac, color="#6366f1", width=0.5)
        axes[0].set_xlabel("Weight decay $\\lambda$")
        axes[0].set_ylabel("Fraction of seeds, FVE $\\geq 0.95$")
        axes[0].set_ylim(0, 1.05)
        axes[1].plot(wds, mean_fve, "o-", color="#22c55e")
        axes[1].axhline(0.95, color="gray", ls="--", lw=0.8)
        axes[1].set_xlabel("Weight decay $\\lambda$")
        axes[1].set_ylabel("Mean final adaptive FVE")
        axes[1].set_ylim(0, 1.05)
        for ax in axes:
            ax.grid(alpha=0.25)
        _save(fig, out / "fig_weight_decay_sweep.pdf")
        return
    candidates = metrics.get("ep1_weight_decay_sweep", [])
    if not candidates:
        return
    wds = [float(c["weight_decay"]) for c in candidates]
    clean = [int(c.get("n_clean_final", 0)) for c in candidates]
    mean_fve = [float(c.get("mean_fve_adaptive", 0)) for c in candidates]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].bar([str(w) for w in wds], clean, color="#6366f1")
    axes[0].set_xlabel("Weight decay $\\lambda$")
    axes[0].set_ylabel("Seeds with FVE $\\geq 0.95$")
    axes[0].set_ylim(0, 5.5)
    axes[1].plot(wds, mean_fve, "o-", color="#22c55e")
    axes[1].axhline(0.95, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("Weight decay $\\lambda$")
    axes[1].set_ylabel("Mean final adaptive FVE")
    axes[1].set_ylim(0, 1.05)
    _save(fig, out / "fig_weight_decay_sweep.pdf")
