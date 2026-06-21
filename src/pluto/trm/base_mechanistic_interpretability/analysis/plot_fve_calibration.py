#!/usr/bin/env python3
"""Plot faithful trig-FVE vs number of key frequencies (calibration validation).

Shows that the Nanda one-layer baseline and the clean TRM-minimal seeds reach
~0.99 FVE once enough key frequencies are included (the circuit spans 6-8
frequencies, not 5), validating the progress-measure stack; degraded TRM seeds
and the recursive TRM-full plateau far below, evidencing a non-Fourier mechanism.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from pluto.trm.base_mechanistic_interpretability.analysis import figstyle  # noqa: E402
from pluto.trm.base_mechanistic_interpretability.analysis.common import eval_all_pairs_logits_from_checkpoint  # noqa: E402
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset  # noqa: E402
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (  # noqa: E402
    fit_trig_logits_fve_bias_corrected,
    identify_key_frequencies_by_excluded,
    logits_grid,
)

MODELS = [("nanda_a_mlp", "nanda", "Nanda 1-layer (calibration)"),
          ("trm_minimal", "trm_minimal", "TRM minimal"),
          ("trm_full_b", "trm_full", "TRM full (ACT)")]
KS = list(range(1, 13))


@torch.no_grad()
def curve(ck: str, model_type: str, device) -> list[float]:
    logits, cfg, _ = eval_all_pairs_logits_from_checkpoint(ck, model_type, device)
    ds = ModAddFullDataset(cfg)
    lab, tr, te = ds.labels[:, 2].to(device), ds.train_mask.to(device), ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)
    ranked = identify_key_frequencies_by_excluded(grid, lab, tr, te, cfg.p, top_k=max(KS))
    return [fit_trig_logits_fve_bias_corrected(grid, ranked[:k], cfg.p)["fve_mean"] for k in KS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="bmi_hybrid_50k")
    ap.add_argument("--out", default="bmi_hybrid_50k/aggregate/figures/fig_fve_calibration.pdf")
    args = ap.parse_args()
    figstyle.apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.results_root)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (mdir, mtype, title) in zip(axes, MODELS):
        for seed_dir in sorted((root / mdir).glob("seed_*")):
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
    axes[0].set_ylabel("Faithful trig-FVE")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
