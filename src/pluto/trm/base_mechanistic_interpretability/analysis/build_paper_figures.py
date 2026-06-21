#!/usr/bin/env python3
"""Regenerate every paper figure from local .bmi-remote-results artifacts.

No checkpoints/GPU required: this only re-plots existing JSON outputs using the
shared Faithful-Nanda stylesheet into paper/figures/.

The single exception is fig_fve_calibration.pdf (Fig 1), which is recomputed
from 50k final checkpoints that are not synced locally; its existing rendering
is retained. Run plot_fve_calibration.py on the machine that holds the 50k
checkpoints to refresh it.
"""
from __future__ import annotations

import json
from pathlib import Path

from pluto.trm.base_mechanistic_interpretability.analysis import figstyle
from pluto.trm.base_mechanistic_interpretability.analysis import plot_figures as pf
from pluto.trm.base_mechanistic_interpretability.analysis import plot_hybrid_figures as ph
from pluto.trm.base_mechanistic_interpretability.analysis import plot_ren_diagnostics as prd

# Repo root is the project dir that contains .bmi-remote-results. This file lives at
# <repo>/pluto/trm/base_mechanistic_interpretability/analysis/build_paper_figures.py,
# so the project root is parents[4].
REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / ".bmi-remote-results"
assert RESULTS.exists(), f"results dir not found: {RESULTS}"
AGG = RESULTS / "nanda_50k_ren" / "aggregate"
RIGOR = RESULTS / "hybrid_rigor"
REN = RESULTS / "hybrid_rigor_ren"
FIGS = REPO / "pluto/trm/base_mechanistic_interpretability/paper/figures"


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    figstyle.apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    metrics = _load(AGG / "hybrid_metrics.json")

    # Aggregate (50k) figures
    if metrics:
        ph.plot_model_comparison(metrics, FIGS)        # Fig 2
        ph.plot_multi_seed(metrics, FIGS)              # Fig 3
    ph.plot_placeholder_ladder(FIGS)                   # Fig 10

    # Per-seed mechanistic figures (representative seeds from the rigor run)
    ph.plot_phase_combined(RIGOR, FIGS, model="trm_minimal", seed="seed_0")        # Fig 7
    ph.plot_per_seed_grokking(RIGOR, FIGS)                                          # Fig 9
    ph.plot_neuron_histogram(RIGOR, FIGS, model="trm_minimal", seed="seed_0")      # Fig 6
    ph.plot_all_frequency_grid(RIGOR, FIGS)                                         # Fig 8
    ph.plot_weight_norm_phases(RIGOR, FIGS, model="trm_minimal", seed="seed_0")    # Fig 25

    rep = RIGOR / "trm_minimal" / "seed_0" / "analysis"
    re_ = _load(rep / "reverse_engineering" / "reverse_engineering.json")
    if re_:
        pf.plot_reverse_engineering(re_, FIGS)         # Figs 4, 5
    pm = _load(rep / "progress" / "progress_measures_grokking.json")
    if pm:
        pf.plot_progress(pm, FIGS)                     # Fig 17
    traj = _load(rep / "trajectory" / "progress_trajectory.json")
    if traj:
        pf.plot_trajectory(traj, FIGS)                 # Fig 18
    ab = _load(rep / "ablations" / "frequency_ablations.json")
    if ab:
        pf.plot_ablations(ab, FIGS)                    # Fig 19
    mech = _load(rep / "mechanistic" / "mechanistic_circuit.json")
    if mech:
        pf.plot_mechanistic(mech, FIGS)               # Fig 20
    hist = rep.parent / "training_history.json"
    if hist.exists():
        pf.plot_grokking_curves(hist, FIGS)           # Fig 16

    # Fig 11: full-TRM latent dynamics from the richer hrm json
    hrm = _load(RIGOR / "trm_full_b" / "seed_0" / "analysis" / "hrm" / "hrm_reasoning_guessing.json")
    if hrm:
        pf.plot_reasoning(hrm, FIGS)                   # Figs 11, 14 (also writes PCA trajectory)

    # Ren diagnostics (Figs 12, 13, 15) from the flat hybrid_rigor_ren suites + report.
    prd.plot_act_depth_grid(REN, FIGS)                 # Fig 12
    ren_report = _load(AGG / "ren_diagnosis_report.json")
    if ren_report:
        prd.plot_fp_violation_bars(ren_report, FIGS)   # Fig 13
        prd.plot_mechanism_summary(ren_report, FIGS)   # Fig 15

    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
