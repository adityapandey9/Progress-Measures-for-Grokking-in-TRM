#!/usr/bin/env python3
"""Mechanistic circuit analysis: W_E, W_U, latent readout, FVE (Nanda §4)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import (
    ensure_dir,
    eval_all_pairs_logits_and_latent,
    load_analysis_bundle,
    save_json,
)
from pluto.trm.models.losses import ACTLossHead
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    embedding_fourier_norms,
    fit_trig_logits_fve,
    gini_coefficient,
    identify_key_frequencies,
    latent_fve_along_freq,
    logits_fourier_norm_map,
    logits_grid,
    progress_measure_bundle,
    unembed_fourier_norms,
)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg, w_e, w_u = load_analysis_bundle(args.checkpoint, args.model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("Mechanistic circuit analysis requires TRM checkpoints")
    ds = ModAddFullDataset(cfg)

    logits_flat, z_h = eval_all_pairs_logits_and_latent(model, cfg, device)
    grid = logits_grid(logits_flat, cfg.p)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)

    emb_norms = embedding_fourier_norms(w_e, cfg.p)
    key_freqs = identify_key_frequencies(emb_norms, top_k=5)

    bundle = progress_measure_bundle(grid, labels, train_m, test_m, key_freqs, w_e, w_u)
    latent_fve = latent_fve_along_freq(z_h, key_freqs, cfg.p)

    # W_U rank-10 style: top Fourier directions on output axis
    un_norms = unembed_fourier_norms(w_u, cfg.p)
    log_f_map = logits_fourier_norm_map(grid, cfg.p)

    results: Dict[str, Any] = {
        "paper": "2301.05217",
        "checkpoint": args.checkpoint,
        "key_frequencies": key_freqs,
        "progress_measures": bundle,
        "latent_readout_fve": latent_fve,
        "mechanistic_summary": {
            "embedding_top5_freq_energy": float(emb_norms[key_freqs].sum().item()) if key_freqs else 0.0,
            "unembed_top5_freq_energy": float(
                sum(un_norms[2 * k].item() + un_norms[2 * k + 1].item() for k in key_freqs)
            ),
            "logit_trig_fve": bundle["logit_trig_fve"],
            "embedding_gini": bundle["embedding_gini"],
            "unembed_gini": bundle["unembed_gini"],
            "logits_fourier_gini": bundle["logits_fourier_gini"],
        },
        "tables": {
            "embedding_fourier_norms_top10": {
                str(i): float(emb_norms[i].item()) for i in range(min(20, len(emb_norms)))
            },
            "unembed_fourier_norms_top10": {
                str(i): float(un_norms[i].item()) for i in range(min(20, len(un_norms)))
            },
        },
    }

    out = ensure_dir(Path(args.output_dir))
    save_json(out / "mechanistic_circuit.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/mechanistic")
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--frac-train", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-type", default="trm_full", choices=["trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run(args)
    pm = r["progress_measures"]
    print(
        f"trig_test={pm['trig_loss_test']:.4f} excluded_test={pm['excluded_loss_test']:.4f} "
        f"key_freqs={r['key_frequencies']}"
    )


if __name__ == "__main__":
    main()
