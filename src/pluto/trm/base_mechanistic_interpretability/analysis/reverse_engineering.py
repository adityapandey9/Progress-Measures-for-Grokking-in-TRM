#!/usr/bin/env python3
"""Full reverse-engineering analysis (Nanda §4): W_L, MLP neurons, attention heads."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pluto.trm.base_mechanistic_interpretability.analysis.common import (
    ensure_dir,
    eval_all_pairs_logits,
    load_analysis_bundle,
    save_json,
)
from pluto.trm.models.losses import ACTLossHead
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    embedding_fourier_norms,
    fft1d,
    fourier_basis,
    identify_key_frequencies_by_excluded,
    logits_grid,
    progress_measure_bundle,
)


def _mlp_inter_size(hidden: int, expansion: float) -> int:
    inter = int(round(expansion * hidden * 2 / 3))
    return ((inter + 255) // 256) * 256


@torch.no_grad()
def _collect_mlp_activations(model, cfg: ModAddGrokkingConfig, device: torch.device) -> Dict[str, Any]:
    """Capture post-SwiGLU activations at '=' for each L-layer."""
    inner = model.model.inner
    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    storage: Dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            x = inputs[0]
            storage[name] = x[:, 2, :].detach().cpu()

        return hook

    handles = []
    for li, block in enumerate(inner.L_level.layers):
        handles.append(block.mlp.down_proj.register_forward_hook(make_hook(f"layer{li}_mlp_act")))

    carry = model.initial_carry(batch)
    carry, _ = model.model(carry=carry, batch=batch)

    for h in handles:
        h.remove()

    attn_storage: Dict[str, torch.Tensor] = {}

    def attn_hook(name: str):
        def hook(_module, _inputs, output):
            attn_storage[name] = output[:, 2, :].detach().cpu()

        return hook

    attn_handles = []
    for li, block in enumerate(inner.L_level.layers):
        attn_handles.append(block.self_attn.register_forward_hook(attn_hook(f"layer{li}_attn_out")))

    carry = model.initial_carry(batch)
    carry, _ = model.model(carry=carry, batch=batch)

    for h in attn_handles:
        h.remove()

    return {"mlp": storage, "attn": attn_storage}


def _neuron_logit_map_wl(model, cfg: ModAddGrokkingConfig, device: torch.device) -> Dict[str, Any]:
    """W_L: MLP neuron -> logit map = down_proj @ lm_head^T (Nanda W_out @ W_U)."""
    inner = model.model.inner
    w_u = inner.lm_head.weight[: cfg.p].detach().to(device)
    maps: Dict[str, Any] = {}
    for li, block in enumerate(inner.L_level.layers):
        down = block.mlp.down_proj.weight.detach().to(device)  # [hidden, inter]
        wl = down.T @ w_u.T  # [inter, p]
        basis = fourier_basis(cfg.p, device)
        coeffs = fft1d(wl, basis)
        norms = coeffs.pow(2).sum(0).sqrt().cpu().tolist()
        key = identify_key_frequencies_by_excluded_from_norms(norms, top_k=5)
        maps[f"layer_{li}"] = {
            "shape": list(wl.shape),
            "fourier_norms": norms,
            "key_frequencies": key,
            "gini": _gini(torch.tensor(norms)),
            "top_neurons_by_norm": _top_neuron_indices(wl, k=10),
        }
    final = maps[f"layer_{len(inner.L_level.layers) - 1}"]
    return {"per_layer": maps, "W_L_final": final}


def identify_key_frequencies_by_excluded_from_norms(norms: List[float], top_k: int = 5) -> List[int]:
    pair_energy = []
    for k in range(len(norms) // 2):
        e = norms[2 * k] + norms[2 * k + 1]
        pair_energy.append((e, k))
    pair_energy.sort(reverse=True)
    return [k for _, k in pair_energy[:top_k]]


def _top_neuron_indices(wl: torch.Tensor, k: int = 10) -> List[int]:
    energy = wl.pow(2).sum(dim=1)
    top = torch.topk(energy, min(k, energy.numel()))
    return top.indices.cpu().tolist()


def _gini(x: torch.Tensor) -> float:
    v = x.flatten().float()
    v = v[v >= 0]
    if v.numel() == 0:
        return 0.0
    s, _ = torch.sort(v)
    n = s.numel()
    idx = torch.arange(1, n + 1, dtype=s.dtype)
    return float((2 * (idx * s).sum() / (n * s.sum())) - (n + 1) / n)


def _cluster_mlp_neurons(
    acts: torch.Tensor, cfg: ModAddGrokkingConfig, key_freqs: List[int]
) -> Dict[str, Any]:
    """Cluster neurons by dominant cos/sin(w(a+b)) correlation (Nanda §4.2)."""
    p = cfg.p
    n_pairs = acts.shape[0]
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    clusters: Dict[int, List[int]] = {k: [] for k in key_freqs}
    unassigned: List[int] = []
    neuron_freq: Dict[int, int] = {}
    fve_scores: List[float] = []

    neurons: List[Dict[str, Any]] = []
    for n in range(acts.shape[1]):
        y = acts[:, n].double()
        if y.std() < 1e-8:
            unassigned.append(n)
            continue
        best_k, best_r2, best_basis = -1, 0.0, "unknown"
        for k in key_freqs:
            w = 2.0 * math.pi * k / p
            for basis_name, feat_fn in (("cos", torch.cos), ("sin", torch.sin)):
                feat = feat_fn(w * (a + b).double())
                feat = feat - feat.mean()
                yc = y - y.mean()
                denom = (feat.pow(2).sum() * yc.pow(2).sum()).clamp_min(1e-12)
                r2 = float((feat @ yc).pow(2).item() / denom.item())
                if r2 > best_r2:
                    best_r2, best_k, best_basis = r2, k, basis_name
        neuron_freq[n] = best_k
        fve_scores.append(best_r2)
        neurons.append(
            {
                "neuron": int(n),
                "best_frequency": int(best_k),
                "best_r2": float(best_r2),
                "basis": best_basis,
            }
        )
        if best_k in clusters:
            clusters[best_k].append(n)
        else:
            unassigned.append(n)

    frac_assigned = sum(len(v) for v in clusters.values()) / max(acts.shape[1], 1)
    return {
        "key_frequencies": key_freqs,
        "neurons_per_freq": {str(k): len(v) for k, v in clusters.items()},
        "fraction_neurons_assigned": frac_assigned,
        "mean_neuron_fve": float(np.mean(fve_scores)) if fve_scores else 0.0,
        "neuron_dominant_freq_sample": {str(n): neuron_freq[n] for n in list(neuron_freq.keys())[:20]},
        "neurons": neurons,
    }


def _attention_decomposition(model, cfg: ModAddGrokkingConfig, device: torch.device) -> Dict[str, Any]:
    """Average attention from '=' to a,b tokens per head (Nanda §4.1)."""
    inner = model.model.inner
    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    attn_weights: List[torch.Tensor] = []

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
            import einops

            query = einops.rearrange(query, "B S H D -> B H S D")
            key = einops.rearrange(key, "B S H D -> B H S D")
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hd)
            weights = F.softmax(scores, dim=-1)
            attn_weights.append(weights[:, :, 2, :3].detach().cpu())
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

    results: Dict[str, Any] = {"layers": []}
    for li, w in enumerate(attn_weights):
        mean_w = w.mean(dim=0)
        results["layers"].append(
            {
                "layer": li,
                "mean_attn_eq_to_abc": mean_w.tolist(),
                "heads_attn_to_a": mean_w[:, 0].tolist(),
                "heads_attn_to_b": mean_w[:, 1].tolist(),
                "heads_attn_to_eq": mean_w[:, 2].tolist(),
            }
        )
    return results


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg, w_e, w_u = load_analysis_bundle(args.checkpoint, args.model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("Reverse engineering requires TRM checkpoints")

    from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset

    logits = eval_all_pairs_logits(model, cfg, device)
    ds = ModAddFullDataset(cfg)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)
    bundle = progress_measure_bundle(grid, labels, train_m, test_m, [], w_e, w_u)
    key_freqs = bundle["key_frequencies"]

    captures = _collect_mlp_activations(model, cfg, device)
    wl = _neuron_logit_map_wl(model, cfg, device)

    mlp_clusters: Dict[str, Any] = {}
    for name, acts in captures["mlp"].items():
        mlp_clusters[name] = _cluster_mlp_neurons(acts, cfg, key_freqs)

    attn = _attention_decomposition(model, cfg, device)

    emb_key = identify_key_frequencies_by_excluded_from_norms(
        embedding_fourier_norms(w_e, cfg.p).cpu().tolist(), top_k=5
    )

    results: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "key_frequencies_progress": key_freqs,
        "key_frequencies_embedding": emb_key,
        "W_L_analysis": wl,
        "mlp_neuron_clustering": mlp_clusters,
        "attention_decomposition": attn,
        "progress_measures_summary": {
            "full_loss_test": bundle["full_loss_test"],
            "trig_loss_test": bundle["trig_loss_test"],
            "excluded_loss_test": bundle["excluded_loss_test"],
            "logit_trig_fve": bundle["logit_trig_fve"],
            "embedding_gini": bundle["embedding_gini"],
            "unembed_gini": bundle["unembed_gini"],
        },
        "algorithm_summary": _algorithm_summary(key_freqs, wl, mlp_clusters, attn),
    }

    out = ensure_dir(Path(args.output_dir))
    save_json(out / "reverse_engineering.json", results)
    return results


def _algorithm_summary(
    key_freqs: List[int],
    wl: Dict[str, Any],
    mlp_clusters: Dict[str, Any],
    attn: Dict[str, Any],
) -> Dict[str, Any]:
    final_layer = mlp_clusters.get("layer1_mlp_act", mlp_clusters.get("layer0_mlp_act", {}))
    return {
        "steps": [
            "Embed a,b into sparse Fourier components (W_E)",
            "Attention routes '=' to a,b token positions",
            "MLP neurons compute cos/sin(w_k(a+b)) via SwiGLU layers",
            "Unembed maps to logits approximating cos(w_k(a+b-c))",
        ],
        "key_frequencies": key_freqs,
        "W_L_key_frequencies": wl.get("W_L_final", {}).get("key_frequencies", []),
        "mlp_neuron_fraction_assigned": final_layer.get("fraction_neurons_assigned", 0),
        "n_attention_layers": len(attn.get("layers", [])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/reverse_engineering")
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--frac-train", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-type", default="trm_full", choices=["trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run(args)
    print("W_L keys", r["W_L_analysis"]["W_L_final"]["key_frequencies"])
    print("algorithm", r["algorithm_summary"])


if __name__ == "__main__":
    main()
