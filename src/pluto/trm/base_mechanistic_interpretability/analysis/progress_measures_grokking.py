#!/usr/bin/env python3
"""Progress measures for grokking (arXiv:2301.05217) on base TRM + modular addition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import (
    ensure_dir,
    eval_all_pairs_logits_from_checkpoint,
    save_json,
)
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig
from pluto.trm.base_mechanistic_interpretability.models.nanda_one_layer import NandaOneLayerTransformer
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    embedding_fourier_norms,
    identify_key_frequencies,
    logits_fourier_norms,
    logits_grid,
    progress_measure_bundle,
)


def training_phases(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic three-phase split: memorization, circuit formation, cleanup (Nanda §5)."""
    if len(history) < 3:
        return {"memorization_end": 0, "circuit_end": 0, "cleanup_start": 0}
    train_acc = [h["train_acc"] for h in history]
    test_acc = [h["test_acc"] for h in history]
    mem_end = next((i for i, a in enumerate(train_acc) if a > 0.95), len(train_acc) // 3)
    circ_end = next((i for i, a in enumerate(test_acc[mem_end:], start=mem_end) if a > 0.5), len(test_acc) * 2 // 3)
    return {
        "memorization_end_step": history[mem_end]["step"] if mem_end < len(history) else 0,
        "circuit_formation_end_step": history[circ_end]["step"] if circ_end < len(history) else 0,
        "cleanup_start_step": history[circ_end]["step"] if circ_end < len(history) else 0,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = ModAddGrokkingConfig(p=args.p, frac_train=args.frac_train, seed=args.seed)
    logits, ds_cfg, model = eval_all_pairs_logits_from_checkpoint(args.checkpoint, args.model_type, device)
    cfg = ds_cfg

    ds = ModAddFullDataset(cfg)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)

    grid = logits_grid(logits, cfg.p)
    if isinstance(model, NandaOneLayerTransformer):
        w_e = model.embed_tokens.embedding_weight.detach()
        w_u = model.lm_head.weight.detach()
        variant = "nanda_baseline"
    else:
        w_e = model.model.inner.embed_tokens.embedding_weight.detach()
        w_u = model.model.inner.lm_head.weight.detach()
        variant = args.model_type
    bundle = progress_measure_bundle(grid, labels, train_m, test_m, [], w_e, w_u)
    key_freqs = bundle["key_frequencies"]
    log_f_norms = logits_fourier_norms(grid, cfg.p).reshape(-1).cpu().tolist()

    history: List[Dict[str, Any]] = []
    hist_path = Path(args.training_history)
    if hist_path.exists():
        history = json.loads(hist_path.read_text())

    results: Dict[str, Any] = {
        "paper": "2301.05217",
        "variant": variant,
        "model_type": args.model_type,
        "task": "modular_addition",
        "p": cfg.p,
        "frac_train": cfg.frac_train,
        "checkpoint": args.checkpoint,
        "train_loss": bundle["full_loss_train"],
        "test_loss": bundle["full_loss_test"],
        "train_accuracy": (logits[train_m].argmax(-1) == labels[train_m]).float().mean().item(),
        "test_accuracy": (logits[test_m].argmax(-1) == labels[test_m]).float().mean().item(),
        "key_frequencies": key_freqs,
        "embedding_fourier_norms": bundle["embedding_fourier_norms"],
        "unembed_fourier_norms": bundle["unembed_fourier_norms"],
        "logits_fourier_norms_flat": log_f_norms[:56],
        "trig_loss_train": bundle["trig_loss_train"],
        "trig_loss_test": bundle["trig_loss_test"],
        "excluded_loss_train": bundle["excluded_loss_train"],
        "excluded_loss_test": bundle["excluded_loss_test"],
        "excluded_loss_test_per_freq": bundle["excluded_loss_test_per_freq"],
        "embedding_gini": bundle["embedding_gini"],
        "unembed_gini": bundle["unembed_gini"],
        "logits_fourier_gini": bundle["logits_fourier_gini"],
        "logit_trig_fve": bundle["logit_trig_fve"],
        "logit_trig_fve_faithful": bundle["logit_trig_fve_faithful"],
        "training_phases": training_phases(history),
    }

    out = ensure_dir(Path(args.output_dir))
    save_json(out / "progress_measures_grokking.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/progress")
    p.add_argument("--training-history", default="bmi_grokking_runs/default/training_history.json")
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--frac-train", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-type", default="trm_full", choices=["nanda", "trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run(args)
    print(
        f"test_acc={r['test_accuracy']:.4f} trig_test={r['trig_loss_test']:.4f} "
        f"excluded_test={r['excluded_loss_test']:.4f} key_freqs={r['key_frequencies']}"
    )


if __name__ == "__main__":
    main()
