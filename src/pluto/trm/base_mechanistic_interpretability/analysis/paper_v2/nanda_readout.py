"""Nanda Table 1-style W_L readout analysis for paper v2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, load_analysis_bundle, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.reverse_engineering import (
    _collect_mlp_activations,
    _neuron_logit_map_wl,
)
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import fft1d, fourier_basis, logits_grid
from pluto.trm.models.losses import ACTLossHead


def _fit_single_trig(
    y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, k: int, p: int, *, sin_basis: bool
) -> Tuple[float, float]:
    if k <= 0:
        return 0.0, 0.0
    w = 2.0 * math.pi * k / p
    if sin_basis:
        feat = torch.sin(w * (a + b).double())
    else:
        feat = torch.cos(w * (a + b).double())
    feat = feat - feat.mean()
    yc = y.double() - y.double().mean()
    feat_var = feat.pow(2).sum().clamp_min(1e-12)
    yc_var = yc.pow(2).sum().clamp_min(1e-12)
    dot = float((feat @ yc).item())
    coef = dot / feat_var.item()
    fve = (dot * dot) / (feat_var.item() * yc_var.item())
    return coef, fve


def _wl_projection_fve(grid: torch.Tensor, k: int, p: int, *, sin_basis: bool) -> Tuple[float, float]:
    """FVE of W_L cos/sin(w_k c) logit projection vs cos/sin(w_k(a+b))."""
    w = 2.0 * math.pi * k / p
    c = torch.arange(p, dtype=torch.float64)
    basis_c = torch.sin(w * c) if sin_basis else torch.cos(w * c)
    centered = grid.double().cpu() - grid.double().cpu().mean(dim=-1, keepdim=True)
    proj = (centered * basis_c.view(1, 1, p)).sum(dim=-1).flatten()
    a = torch.arange(p, dtype=torch.float64).repeat_interleave(p)
    b = torch.arange(p, dtype=torch.float64).repeat(p)
    feat = torch.sin(w * (a + b)) if sin_basis else torch.cos(w * (a + b))
    feat = feat - feat.mean()
    yc = proj - proj.mean()
    feat_var = feat.pow(2).sum().clamp_min(1e-12)
    yc_var = yc.pow(2).sum().clamp_min(1e-12)
    dot = feat @ yc
    coef = dot / feat_var
    fve = (dot * dot) / (feat_var * yc_var)
    return float(coef.item()), float(fve.item())


def _best_neuron_for_direction(
    mlp_acts: torch.Tensor, k: int, p: int, *, sin_basis: bool
) -> Tuple[int, float, float]:
    """Find neuron whose activation best matches cos/sin(w_k(a+b))."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    best_idx, best_coef, best_fve = 0, 0.0, -1.0
    for n in range(mlp_acts.shape[1]):
        coef, fve = _fit_single_trig(mlp_acts[:, n], a, b, k, p, sin_basis=sin_basis)
        if fve > best_fve:
            best_idx, best_coef, best_fve = n, coef, fve
    return best_idx, best_coef, best_fve


def build_nanda_readout_table(
    grid: torch.Tensor,
    wl: torch.Tensor,
    mlp_acts: torch.Tensor,
    key_freqs: List[int],
    p: int,
) -> List[Dict[str, Any]]:
    """Table 1 rows: logit-space FVE per W_L direction + best matching neuron."""
    device = grid.device
    basis = fourier_basis(p, device)
    coeffs = fft1d(wl, basis)

    rows: List[Dict[str, Any]] = []
    for k in key_freqs:
        if k <= 0:
            continue
        for sin_basis, label in ((False, f"cos(w_{k}c)"), (True, f"sin(w_{k}c)")):
            col = 2 * k + (1 if sin_basis else 0)
            if col >= coeffs.shape[1]:
                continue
            u = coeffs[:, col]
            top_neurons = torch.topk(u.abs(), min(32, u.numel())).indices.tolist()
            logit_coef, logit_fve = _wl_projection_fve(grid, k, p, sin_basis=sin_basis)
            best_idx, best_coef, best_neuron_fve = 0, 0.0, -1.0
            for direction_idx in top_neurons:
                proj = mlp_acts[:, direction_idx]
                a = torch.arange(p).repeat_interleave(p)
                b = torch.arange(p).repeat(p)
                coef, fve = _fit_single_trig(proj, a, b, k, p, sin_basis=sin_basis)
                if fve > best_neuron_fve:
                    best_idx, best_coef, best_neuron_fve = direction_idx, coef, fve
            if best_neuron_fve < 0:
                best_idx, best_coef, best_neuron_fve = _best_neuron_for_direction(
                    mlp_acts, k, p, sin_basis=sin_basis
                )
            # Report logit-direction FVE (matches grokked circuit); neuron fit is auxiliary.
            rows.append(
                {
                    "component": label,
                    "frequency": k,
                    "neuron_index": int(best_idx),
                    "coefficient": round(logit_coef, 2),
                    "fve": round(logit_fve * 100, 1),
                    "neuron_fve": round(best_neuron_fve * 100, 1),
                }
            )
    return rows


def wl_directions_from_checkpoint(model, cfg, device: torch.device) -> Tuple[torch.Tensor, List[int]]:
    wl_info = _neuron_logit_map_wl(model, cfg, device)
    inner = model.model.inner
    w_u = inner.lm_head.weight[: cfg.p].detach().to(device)
    block = inner.L_level.layers[-1]
    down = block.mlp.down_proj.weight.detach().to(device)
    wl = down.T @ w_u.T
    key_freqs = wl_info["W_L_final"]["key_frequencies"]
    return wl, key_freqs


def run_readout(checkpoint: str, model_type: str, output_dir: str, *, cpu: bool = False) -> Dict[str, Any]:
    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    from pluto.trm.base_mechanistic_interpretability.analysis.common import eval_all_pairs_logits_from_checkpoint
    from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import identify_key_frequencies_adaptive
    from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset

    model, cfg, _, _ = load_analysis_bundle(checkpoint, model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("Readout requires TRM checkpoints")

    logits, _, _ = eval_all_pairs_logits_from_checkpoint(checkpoint, model_type, device, prefer_ptrm=True)
    ds = ModAddFullDataset(cfg)
    lab = ds.labels[:, 2].to(device)
    tr = ds.train_mask.to(device)
    te = ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)
    adaptive_keys = [k for k in identify_key_frequencies_adaptive(grid, lab, tr, te, cfg.p) if k > 0]

    wl, wl_keys = wl_directions_from_checkpoint(model, cfg, device)
    key_freqs = sorted(set(wl_keys) | set(adaptive_keys))
    captures = _collect_mlp_activations(model, cfg, device)
    layer_key = "layer1_mlp_act" if "layer1_mlp_act" in captures["mlp"] else next(iter(captures["mlp"]))
    mlp_acts = captures["mlp"][layer_key]
    rows = build_nanda_readout_table(grid, wl, mlp_acts, key_freqs, cfg.p)
    n_above_90 = sum(1 for r in rows if r["fve"] >= 90.0)
    results = {
        "checkpoint": checkpoint,
        "key_frequencies": key_freqs,
        "readout_rows": rows,
        "n_directions_above_90pct": n_above_90,
        "n_directions": len(rows),
        "readout_method": "wl_projection_trig_fve",
    }
    out = ensure_dir(Path(output_dir))
    save_json(out / "nanda_readout.json", results)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-type", default="trm_minimal")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    r = run_readout(args.checkpoint, args.model_type, args.output_dir, cpu=args.cpu)
    print(f"readout: {r['n_directions_above_90pct']}/{r['n_directions']} directions >= 90% FVE")


if __name__ == "__main__":
    main()
