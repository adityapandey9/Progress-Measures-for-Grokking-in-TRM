#!/usr/bin/env python3
"""Build all paper v2 figures from Arc A result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pluto.trm.base_mechanistic_interpretability.analysis import figstyle
from pluto.trm.base_mechanistic_interpretability.analysis import plot_v2_figures as pv2
from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--figures-dir", required=True)
    ap.add_argument("--best-wd", required=True, type=float)
    args = ap.parse_args()

    figstyle.apply_style()
    root = Path(args.results_root)
    out = ensure_dir(Path(args.figures_dir))
    metrics_path = root / "aggregate" / "paper_v2_metrics.json"
    metrics = _load(metrics_path)
    if not metrics:
        metrics = {"best_weight_decay": args.best_wd, "models": {}}

    pv2.plot_v2_algorithm_schematic(out)
    pv2.plot_v2_grokking(root, metrics, out)
    pv2.plot_v2_multiseed(metrics, out)
    pv2.plot_v2_weight_decay(metrics, out)
    pv2.plot_v2_data_fraction(metrics, out)
    pv2.plot_v2_fve_calibration(root, metrics, out)

    run_dir = pv2._mainline_run(metrics, root)
    if run_dir:
        re_ = _load(run_dir / "analysis/reverse_engineering/reverse_engineering.json")
        if re_:
            pv2.plot_v2_we_fourier(re_, out)
            pv2.plot_v2_wl_fourier(re_, out)
            pv2.plot_v2_attention_neuron_heatmaps(re_, run_dir, out)
        pv2.plot_v2_neuron_variance_explained(run_dir, out)
        pv2.plot_v2_key_freq_ablation(run_dir, out)
        re_ = _load(run_dir / "analysis/reverse_engineering/reverse_engineering.json")
        kf = re_.get("key_frequencies_progress", [])
        if kf:
            pv2.plot_v2_frequency_ablation_grid(run_dir, out, kf)
        pv2.plot_v2_progress_phases(run_dir, out)
        # Manifest name is singular; plot helper writes plural.
        ablation_plural = out / "fig_frequency_ablations.pdf"
        ablation_singular = out / "fig_frequency_ablation.pdf"
        if ablation_plural.exists() and not ablation_singular.exists():
            ablation_singular.write_bytes(ablation_plural.read_bytes())

    print(f"paper v2 figures -> {out}")


if __name__ == "__main__":
    main()
