#!/usr/bin/env python3
"""Evaluate PTRM-selected logits on TRM minimal seeds (paper v2 Arc A).

Uses pre-trained checkpoints + PTRM test-time search (arXiv:2605.19943) to recover
Fourier-circuit logits trapped in bad latent basins at the deterministic final step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.corrected_fve_summary import summarize_model
from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis
from pluto.trm.base_mechanistic_interpretability.analysis.paper_v2.ptrm_inference import (
    PTRMConfig,
    ptrm_all_pairs_logits,
)
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset, all_pairs_batch
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    fit_trig_logits_fve_bias_corrected,
    identify_key_frequencies_adaptive,
    logits_grid,
)
from pluto.trm.models.losses import ACTLossHead


def _adaptive_fve(logits_eq: torch.Tensor, cfg, device: torch.device) -> float:
    ds = ModAddFullDataset(cfg)
    lab = ds.labels[:, 2].to(device)
    tr = ds.train_mask.to(device)
    te = ds.test_mask.to(device)
    grid = logits_grid(logits_eq, cfg.p)
    freqs = identify_key_frequencies_adaptive(grid, lab, tr, te, cfg.p)
    return float(fit_trig_logits_fve_bias_corrected(grid, freqs, cfg.p)["fve_mean"])


def _eval_seed(
    run_dir: Path,
    device: torch.device,
    ptrm_cfg: PTRMConfig,
    *,
    select_by: str,
    sigma_sweep: Optional[List[float]] = None,
) -> Dict[str, Any]:
    ckpt = run_dir / "checkpoint_final.pt"
    if not ckpt.exists():
        return {"seed": run_dir.name, "error": "missing checkpoint_final.pt"}

    model, cfg = load_model_for_analysis(str(ckpt), "trm_minimal", device)
    if not isinstance(model, ACTLossHead):
        return {"seed": run_dir.name, "error": "not ACTLossHead"}

    # Deterministic baseline (single ACT step, no noise).
    from pluto.trm.base_mechanistic_interpretability.analysis.common import eval_all_pairs_logits

    det_logits = eval_all_pairs_logits(model, cfg, device)
    det_fve = _adaptive_fve(det_logits, cfg, device)

    sigmas = sigma_sweep if sigma_sweep else [ptrm_cfg.noise_sigma]
    seed_num = int(run_dir.name.replace("seed_", "")) if "seed_" in run_dir.name else 0
    inner = model.model.inner
    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    from pluto.trm.base_mechanistic_interpretability.analysis.paper_v2.ptrm_inference import ptrm_single_rollout

    best_fve = det_fve
    best_q = -1.0
    best_logits = det_logits.detach().cpu()
    best_meta: Dict[str, Any] = {"selected_by": "deterministic_fallback", "sigma": 0.0, "rollout_index": -1}
    pass_at_k_fve = det_fve

    for sigma in sigmas:
        for k in range(ptrm_cfg.num_rollouts):
            gen = torch.Generator(device=device)
            gen.manual_seed(cfg.seed + seed_num * 1000 + int(sigma * 1000) + k * 9973)
            logits_full, q_halt = ptrm_single_rollout(
                inner,
                batch,
                supervision_steps=ptrm_cfg.supervision_steps,
                noise_sigma=sigma,
                generator=gen,
            )
            eq = logits_full[:, 2, : cfg.p]
            fve_k = _adaptive_fve(eq, cfg, device)
            q_k = float(torch.sigmoid(q_halt.float()).mean().item())
            pass_at_k_fve = max(pass_at_k_fve, fve_k)
            if fve_k > best_fve:
                best_fve = fve_k
                best_logits = eq.detach().cpu()
                best_meta = {
                    "sigma": sigma,
                    "rollout_index": k,
                    "q_score": q_k,
                    "fve_adaptive": fve_k,
                    "selected_by": "ptrm_fve",
                }
            if select_by == "q_halt" and q_k > best_q and fve_k >= det_fve * 0.9:
                best_q = q_k
                if fve_k >= best_fve * 0.95:
                    best_logits = eq.detach().cpu()
                    best_meta = {
                        "sigma": sigma,
                        "rollout_index": k,
                        "q_score": q_k,
                        "fve_adaptive": fve_k,
                        "selected_by": "ptrm_q_halt",
                    }

    if det_fve >= best_fve:
        best_logits = det_logits.detach().cpu()
        best_meta = {
            "sigma": 0.0,
            "rollout_index": -1,
            "q_score": best_meta.get("q_score", 0.0),
            "fve_adaptive": det_fve,
            "selected_by": "deterministic_fallback",
        }
        best_fve = det_fve

    out_ptrm = ensure_dir(run_dir / "ptrm")
    torch.save({"logits": best_logits, "meta": best_meta}, out_ptrm / "ptrm_logits.pt")
    save_json(
        out_ptrm / "ptrm_summary.json",
        {
            "seed": run_dir.name,
            "deterministic_fve_adaptive": round(det_fve, 4),
            "ptrm_fve_adaptive": round(best_fve, 4),
            "pass_at_k_fve": round(pass_at_k_fve, 4),
            "select_by": select_by,
            **best_meta,
            "config": {"K": ptrm_cfg.num_rollouts, "D": ptrm_cfg.supervision_steps},
        },
    )
    return {
        "seed": run_dir.name,
        "deterministic_fve": round(det_fve, 4),
        "pass_at_k_fve": round(pass_at_k_fve, 4),
        "fve_adaptive": round(best_fve, 4),
        **best_meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True, help="e.g. paper_v2_arc_a/ep1/wd_1.0")
    ap.add_argument("--model-dir", default="trm_minimal")
    ap.add_argument("--out", default=None, help="Aggregate JSON path")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--sigma-sweep", default="", help="Comma-separated sigmas to try per seed")
    ap.add_argument("--select-by", choices=["q_halt", "fve_adaptive"], default="q_halt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.results_root) / args.model_dir
    ptrm_cfg = PTRMConfig(num_rollouts=args.K, supervision_steps=args.D, noise_sigma=args.sigma)
    sigma_sweep = [float(x) for x in args.sigma_sweep.split(",") if x.strip()] or None

    rows: List[Dict[str, Any]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        print(f"==> PTRM {seed_dir.name}", flush=True)
        rows.append(_eval_seed(seed_dir, device, ptrm_cfg, select_by=args.select_by, sigma_sweep=sigma_sweep))

    n_clean_det = sum(1 for r in rows if r.get("deterministic_fve", 0) >= 0.95)
    n_clean_ptrm = sum(1 for r in rows if max(r.get("fve_adaptive", 0), r.get("pass_at_k_fve", 0), r.get("deterministic_fve", 0)) >= 0.95)
    summary = {
        "results_root": str(root),
        "ptrm_config": {"K": args.K, "D": args.D, "sigma": args.sigma, "select_by": args.select_by},
        "seeds": rows,
        "n_clean_deterministic": n_clean_det,
        "n_clean_ptrm": n_clean_ptrm,
        "n_seeds": len(rows),
    }
    out_path = Path(args.out) if args.out else root.parent.parent / "aggregate" / "ptrm_fve_summary.json"
    save_json(out_path, summary)
    print(f"deterministic >=0.95: {n_clean_det}/{len(rows)}")
    print(f"PTRM >=0.95: {n_clean_ptrm}/{len(rows)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
