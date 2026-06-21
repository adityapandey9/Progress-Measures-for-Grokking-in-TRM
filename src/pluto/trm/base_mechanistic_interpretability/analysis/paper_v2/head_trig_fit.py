"""Nanda Table 2: attention head trig polynomial fits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import einops
import torch
import torch.nn.functional as F

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, load_analysis_bundle, save_json
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch
from pluto.trm.models.losses import ACTLossHead


def fit_head_trig_poly(
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    p: int,
    key_freqs: List[int] | None = None,
) -> Dict[str, Any]:
    key_freqs = key_freqs or list(range(1, min(p // 2, 20)))
    best: Dict[str, Any] = {
        "dominant_frequency": 0,
        "fve_pct": 0.0,
        "coef_cos": 0.0,
        "coef_sin": 0.0,
        "phase": "a+b",
    }
    yc = y.double() - y.double().mean()
    yv = yc.pow(2).sum().clamp_min(1e-12)
    if float(yv) < 1e-12:
        return best
    phases = {
        "a+b": (a + b).double(),
        "a": a.double(),
        "b": b.double(),
        "a-b": (a - b).double(),
    }
    for phase_name, phase in phases.items():
        for k in key_freqs:
            w = 2.0 * math.pi * k / p
            for sin_basis, coef_key in ((False, "coef_cos"), (True, "coef_sin")):
                feat = torch.sin(w * phase) if sin_basis else torch.cos(w * phase)
                feat = feat - feat.mean()
                fv = feat.pow(2).sum().clamp_min(1e-12)
                dot = float((feat @ yc).item())
                fve = (dot * dot) / (float(fv) * float(yv))
                if fve > best["fve_pct"] / 100:
                    coef = dot / float(fv)
                    best = {
                        "dominant_frequency": k,
                        "fve_pct": round(fve * 100, 1),
                        "coef_cos": round(coef, 2) if not sin_basis else best.get("coef_cos", 0.0),
                        "coef_sin": round(coef, 2) if sin_basis else best.get("coef_sin", 0.0),
                        "phase": phase_name,
                    }
                    best[coef_key] = round(coef, 2)
    return best


@torch.no_grad()
def _capture_attn_to_a(model, cfg, device: torch.device) -> torch.Tensor:
    """Per-pair attention from '=' to token $a$ per head: [n_pairs, n_heads]."""
    inner = model.model.inner
    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    weights_store: List[torch.Tensor] = []

    def patch_attention(attn_module):
        orig_forward = attn_module.forward

        def wrapped(cos_sin, hidden_states):
            batch_size, seq_len, _ = hidden_states.shape
            qkv = attn_module.qkv_proj(hidden_states)
            nh = attn_module.num_heads
            nkv = attn_module.num_key_value_heads
            hd = attn_module.head_dim
            qkv = qkv.view(batch_size, seq_len, nh + 2 * nkv, hd)
            query = qkv[:, :, :nh]
            key = qkv[:, :, nh : nh + nkv]
            value = qkv[:, :, nh + nkv :]
            if cos_sin is not None:
                from pluto.trm.models.layers import apply_rotary_pos_emb

                cos, sin = cos_sin
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            query = einops.rearrange(query, "B S H D -> B H S D")
            key = einops.rearrange(key, "B S H D -> B H S D")
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hd)
            weights = F.softmax(scores, dim=-1)
            weights_store.append(weights[:, :, 2, 0].detach().cpu())
            from pluto.trm.models.layers import _resolve_attn_impl

            value = einops.rearrange(value, "B S H D -> B H S D")
            attn_impl = _resolve_attn_impl()
            attn_output = attn_impl(query, key, value, causal=attn_module.causal)
            attn_output = einops.rearrange(attn_output, "B H S D -> B S H D")
            attn_output = attn_output.reshape(batch_size, seq_len, attn_module.output_size)
            return attn_module.o_proj(attn_output)

        attn_module.forward = wrapped

    for block in inner.L_level.layers:
        patch_attention(block.self_attn)

    carry = model.initial_carry(batch)
    carry, _ = model.model(carry=carry, batch=batch)
    return weights_store[0]


@torch.no_grad()
def _capture_attn_to_b(model, cfg, device: torch.device) -> torch.Tensor:
    """Per-pair attention from '=' to token $b$ per head: [n_pairs, n_heads]."""
    inner = model.model.inner
    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    weights_store: List[torch.Tensor] = []

    def patch_attention(attn_module):
        def wrapped(cos_sin, hidden_states):
            batch_size, seq_len, _ = hidden_states.shape
            qkv = attn_module.qkv_proj(hidden_states)
            nh = attn_module.num_heads
            nkv = attn_module.num_key_value_heads
            hd = attn_module.head_dim
            qkv = qkv.view(batch_size, seq_len, nh + 2 * nkv, hd)
            query = qkv[:, :, :nh]
            key = qkv[:, :, nh : nh + nkv]
            value = qkv[:, :, nh + nkv :]
            if cos_sin is not None:
                from pluto.trm.models.layers import apply_rotary_pos_emb

                cos, sin = cos_sin
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            query = einops.rearrange(query, "B S H D -> B H S D")
            key = einops.rearrange(key, "B S H D -> B H S D")
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hd)
            weights = F.softmax(scores, dim=-1)
            weights_store.append(weights[:, :, 2, 1].detach().cpu())
            from pluto.trm.models.layers import _resolve_attn_impl

            value = einops.rearrange(value, "B S H D -> B H S D")
            attn_impl = _resolve_attn_impl()
            attn_output = attn_impl(query, key, value, causal=attn_module.causal)
            attn_output = einops.rearrange(attn_output, "B H S D -> B S H D")
            attn_output = attn_output.reshape(batch_size, seq_len, attn_module.output_size)
            return attn_module.o_proj(attn_output)

        attn_module.forward = wrapped

    for block in inner.L_level.layers:
        patch_attention(block.self_attn)

    carry = model.initial_carry(batch)
    carry, _ = model.model(carry=carry, batch=batch)
    return weights_store[0]


def run(checkpoint: str, model_type: str, output_dir: str) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _, _ = load_analysis_bundle(checkpoint, model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("head_trig_fit requires TRM checkpoints")

    p = cfg.p
    a = torch.arange(p, dtype=torch.float64).repeat_interleave(p)
    b = torch.arange(p, dtype=torch.float64).repeat(p)
    attn_a = _capture_attn_to_a(model, cfg, device)
    attn_b = _capture_attn_to_b(model, cfg, device)
    n_heads = attn_a.shape[1]

    rows: List[Dict[str, Any]] = []
    for hi in range(n_heads):
        ya = attn_a[:, hi]
        yb = attn_b[:, hi]
        fit_a = fit_head_trig_poly(ya, a, b, p)
        fit_b = fit_head_trig_poly(yb, a, b, p)
        fit = fit_a if fit_a["fve_pct"] >= fit_b["fve_pct"] else fit_b
        fit["target"] = "a" if fit_a["fve_pct"] >= fit_b["fve_pct"] else "b"
        rows.append({"layer": 0, "head": hi, **fit})

    out = ensure_dir(Path(output_dir))
    save_json(out / "head_trig_fits.json", {"rows": rows, "checkpoint": checkpoint})
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-type", default="trm_minimal")
    args = ap.parse_args()
    r = run(args.checkpoint, args.model_type, args.output_dir)
    print(f"head fits: {len(r['rows'])} rows")


if __name__ == "__main__":
    main()
