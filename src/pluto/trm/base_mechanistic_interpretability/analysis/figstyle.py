#!/usr/bin/env python3
"""Shared 'Faithful Nanda' matplotlib style for all paper figures.

Call apply_style() once at the top of any figure build before plotting.
Matches the aesthetic of Nanda et al. (arXiv:2301.05217): serif text,
muted palette, light grid, log-friendly loss axes, no in-figure titles.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Muted, print-safe palette (matplotlib C0/C1/... family).
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
HEATMAP_CMAP = "RdBu_r"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # editable/embeddable text in PDF
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "legend.frameon": False,
            "figure.autolayout": True,
        }
    )
