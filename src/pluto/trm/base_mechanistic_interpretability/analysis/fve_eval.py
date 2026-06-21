#!/usr/bin/env python3
"""Unified checkpoint FVE evaluation: excluded/adaptive vs Nanda neuron key frequencies."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.mlp_activations import (
    collect_mlp_neuron_acts_at_equals,
)
from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis
from pluto.trm.base_mechanistic_interpretability.config import mod_add_dataset_config
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset, all_pairs_batch
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    calculate_key_freqs_from_mlp_acts,
    fit_trig_logits_fve_bias_corrected,
    identify_key_frequencies_adaptive,
    identify_key_frequencies_by_excluded,
    logits_grid,
)
from pluto.trm.base_mechanistic_interpretability.models.trm_direct import TrmDirectModel
from pluto.trm.models.losses import ACTLossHead


def _logits_at_equals(model: Any, batch: dict[str, torch.Tensor], cfg, device: torch.device) -> torch.Tensor:
    batch_d = {k: v.to(device) for k, v in batch.items()}
    if isinstance(model, TrmDirectModel):
        return model(batch_d)[:, 2, : cfg.p]
    if isinstance(model, ACTLossHead):
        carry = model.initial_carry(batch_d)
        carry, outputs = model.model(carry=carry, batch=batch_d)
        return outputs["logits"][:, 2, : cfg.p]
    return model(batch_d["inputs"])[:, 2, : cfg.p]


@torch.no_grad()
def eval_checkpoint_fve_metrics(
    checkpoint: str,
    model_type: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Return legacy/adaptive/neuron-key FVE metrics for one checkpoint."""
    model, raw_cfg = load_model_for_analysis(checkpoint, model_type, device)  # type: ignore[arg-type]
    cfg = mod_add_dataset_config(raw_cfg)
    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    logits = _logits_at_equals(model, batch, cfg, device)

    ds = ModAddFullDataset(cfg)
    lab = ds.labels[:, 2].to(device)
    tr = ds.train_mask.to(device)
    te = ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)

    legacy_freqs = identify_key_frequencies_by_excluded(grid, lab, tr, te, cfg.p, top_k=5)
    adaptive_freqs = identify_key_frequencies_adaptive(grid, lab, tr, te, cfg.p)

    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    mlp_acts = collect_mlp_neuron_acts_at_equals(model, batch)
    neuron_freqs = calculate_key_freqs_from_mlp_acts(mlp_acts, cfg.p)

    fve_legacy = float(fit_trig_logits_fve_bias_corrected(grid, legacy_freqs, cfg.p)["fve_mean"])
    fve_adaptive = float(fit_trig_logits_fve_bias_corrected(grid, adaptive_freqs, cfg.p)["fve_mean"])
    fve_neuron = (
        float(fit_trig_logits_fve_bias_corrected(grid, neuron_freqs, cfg.p)["fve_mean"])
        if neuron_freqs
        else 0.0
    )

    return {
        "checkpoint": checkpoint,
        "fve_legacy_k5": round(fve_legacy, 4),
        "fve_adaptive": round(fve_adaptive, 4),
        "fve_neuron_keys": round(fve_neuron, 4),
        "n_key_freqs_legacy": len(legacy_freqs),
        "n_key_freqs_adaptive": len(adaptive_freqs),
        "n_key_freqs_neuron": len(neuron_freqs),
        "key_freqs_adaptive": adaptive_freqs,
        "key_freqs_neuron": neuron_freqs,
    }
