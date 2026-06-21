#!/usr/bin/env python3
"""Causal frequency ablations on logits (Nanda §4.4)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import (
    ensure_dir,
    eval_all_pairs_logits,
    eval_all_pairs_logits_from_checkpoint,
    load_analysis_bundle,
    save_json,
)
from pluto.trm.models.losses import ACTLossHead
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    calculate_excluded_loss,
    calculate_trig_loss,
    embedding_fourier_norms,
    fourier_basis,
    identify_key_frequencies,
    logits_grid,
    progress_measure_bundle,
    test_logits,
)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.model_type == "nanda":
        logits, cfg, model = eval_all_pairs_logits_from_checkpoint(args.checkpoint, args.model_type, device)
    else:
        model, cfg, w_e, w_u = load_analysis_bundle(args.checkpoint, args.model_type, device)
        if not isinstance(model, ACTLossHead):
            raise ValueError("Frequency ablations require TRM or nanda via checkpoint loader")
        logits = eval_all_pairs_logits(model, cfg, device)
    ds = ModAddFullDataset(cfg)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)

    grid = logits_grid(logits, cfg.p)
    if args.model_type == "nanda":
        from pluto.trm.base_mechanistic_interpretability.models.nanda_one_layer import NandaOneLayerTransformer

        assert isinstance(model, NandaOneLayerTransformer)
        w_e = model.embed_tokens.embedding_weight.detach()
        w_u = model.lm_head.weight.detach()
    else:
        w_e = model.model.inner.embed_tokens.embedding_weight.detach()
        w_u = model.model.inner.lm_head.weight.detach()
    key_freqs = progress_measure_bundle(
        grid, labels, train_m, test_m, [], w_e, w_u
    )["key_frequencies"]
    non_key = [k for k in range(cfg.p // 2) if k not in key_freqs][:5]
    basis = fourier_basis(cfg.p, device)

    full_test = test_logits(logits, labels, train_m, test_m, mode="test")
    full_train = test_logits(logits, labels, train_m, test_m, mode="train")
    trig_test = calculate_trig_loss(grid, key_freqs, labels, train_m, test_m, basis, mode="test")
    trig_train = calculate_trig_loss(grid, key_freqs, labels, train_m, test_m, basis, mode="train")
    excluded_test, _ = calculate_excluded_loss(grid, key_freqs, labels, train_m, test_m, basis, mode="test")
    excluded_train, _ = calculate_excluded_loss(grid, key_freqs, labels, train_m, test_m, basis, mode="train")

    # Ablate non-key only (should preserve if circuit is key-only)
    trig_nonkey = calculate_trig_loss(grid, non_key, labels, train_m, test_m, basis, mode="test")
    excluded_nonkey, _ = calculate_excluded_loss(grid, non_key, labels, train_m, test_m, basis, mode="test")

    results: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "key_frequencies": key_freqs,
        "non_key_control": non_key,
        "full_loss_test": full_test,
        "full_loss_train": full_train,
        "trig_loss_key_test": trig_test,
        "trig_loss_key_train": trig_train,
        "excluded_loss_key_test": excluded_test,
        "excluded_loss_key_train": excluded_train,
        "trig_loss_nonkey_test": trig_nonkey,
        "excluded_loss_nonkey_test": excluded_nonkey,
        "interpretation": {
            "key_trig_beats_full": trig_test <= full_test,
            "key_exclude_hurts": excluded_test > full_test,
            "nonkey_trig_near_full": abs(trig_nonkey - full_test) < 1.0,
        },
    }

    out = ensure_dir(Path(args.output_dir))
    save_json(out / "frequency_ablations.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/ablations")
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--frac-train", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-type", default="trm_full", choices=["nanda", "trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run(args)
    print(r["interpretation"])


if __name__ == "__main__":
    main()
