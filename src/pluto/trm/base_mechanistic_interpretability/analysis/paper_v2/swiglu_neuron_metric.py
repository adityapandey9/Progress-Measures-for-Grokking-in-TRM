"""SwiGLU channel trig-FVE metric for Fig 5 / Gate G9."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, load_analysis_bundle, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.reverse_engineering import (
    identify_key_frequencies_by_excluded_from_norms,
)
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import fft1d, fourier_basis
from pluto.trm.models.losses import ACTLossHead


def _fit_channel_trig(
    y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, key_freqs: List[int], p: int
) -> Dict[str, Any]:
    best_k, best_r2, best_basis = -1, 0.0, "unknown"
    for k in key_freqs:
        w = 2.0 * math.pi * k / p
        for basis_name, feat_fn in (("cos", torch.cos), ("sin", torch.sin)):
            feat = feat_fn(w * (a + b).double())
            feat = feat - feat.mean()
            yc = y.double() - y.double().mean()
            denom = (feat.pow(2).sum() * yc.pow(2).sum()).clamp_min(1e-12)
            r2 = float((feat @ yc).pow(2).item() / denom.item())
            if r2 > best_r2:
                best_r2, best_k, best_basis = r2, k, basis_name
    return {"best_frequency": int(best_k), "best_r2": float(best_r2), "basis": best_basis}


@torch.no_grad()
def collect_swiglu_channels(model, cfg, device: torch.device) -> torch.Tensor:
    """Return h = silu(gate) * up at '=' token, shape [n_pairs, n_channels]."""
    inner = model.model.inner
    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    storage: Dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        h = inputs[0]
        storage["h"] = h[:, 2, :].detach().cpu()

    block = inner.L_level.layers[-1]
    handle = block.mlp.down_proj.register_forward_hook(hook)
    carry = model.initial_carry(batch)
    carry, _ = model.model(carry=carry, batch=batch)
    handle.remove()
    return storage["h"]


def swiglu_neuron_trig_fve(model, cfg, device: torch.device, key_freqs: List[int]) -> Dict[str, Any]:
    """Fraction of SwiGLU channels with best_r2 >= 0.85 on h = silu(g)*u."""
    h = collect_swiglu_channels(model, cfg, device)
    p = cfg.p
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    rows: List[Dict[str, Any]] = []
    for ch in range(h.shape[1]):
        row = _fit_channel_trig(h[:, ch], a, b, key_freqs, p)
        row["channel"] = ch
        rows.append(row)
    n = len(rows)
    n85 = sum(1 for r in rows if r["best_r2"] >= 0.85)
    return {
        "fraction_ge_85": n85 / max(n, 1),
        "n_channels": n,
        "n_ge_85": n85,
        "per_channel": sorted(rows, key=lambda r: r["best_r2"], reverse=True)[:20],
    }


def run(checkpoint: str, model_type: str, output_dir: str) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _, _ = load_analysis_bundle(checkpoint, model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("swiglu_neuron_metric requires TRM checkpoints")

    inner = model.model.inner
    w_u = inner.lm_head.weight[: cfg.p].detach().to(device)
    block = inner.L_level.layers[-1]
    down = block.mlp.down_proj.weight.detach().to(device)
    wl = down.T @ w_u.T
    basis = fourier_basis(cfg.p, device)
    coeffs = fft1d(wl, basis)
    norms = coeffs.pow(2).sum(0).sqrt().cpu().tolist()
    key_freqs = identify_key_frequencies_by_excluded_from_norms(norms, top_k=10)

    summary = swiglu_neuron_trig_fve(model, cfg, device, key_freqs)
    summary["key_frequencies"] = key_freqs
    summary["checkpoint"] = checkpoint
    out = ensure_dir(Path(output_dir))
    save_json(out / "summary.json", summary)
    save_json(out / "channels.json", {"rows": summary["per_channel"]})
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-type", default="trm_minimal")
    args = ap.parse_args()
    r = run(args.checkpoint, args.model_type, args.output_dir)
    print(f"SwiGLU channels >=85% FVE: {r['n_ge_85']}/{r['n_channels']} ({r['fraction_ge_85']:.1%})")


if __name__ == "__main__":
    main()
