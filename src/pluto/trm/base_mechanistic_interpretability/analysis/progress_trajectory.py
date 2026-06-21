#!/usr/bin/env python3
"""Progress measures over training checkpoints (Nanda §5 dynamics)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import (
    ensure_dir,
    eval_all_pairs_logits_and_latent,
    eval_all_pairs_logits_from_checkpoint,
    save_json,
)
from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig, mod_add_dataset_config
from pluto.trm.base_mechanistic_interpretability.models.nanda_one_layer import NandaOneLayerTransformer
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    embedding_fourier_norms,
    identify_key_frequencies,
    logits_grid,
    progress_measure_bundle,
)


def _checkpoint_steps(run_dir: Path) -> List[int]:
    steps: List[int] = []
    for p in sorted(run_dir.glob("checkpoint_step*.pt")):
        m = re.search(r"checkpoint_step(\d+)\.pt", p.name)
        if m:
            steps.append(int(m.group(1)))
    if (run_dir / "checkpoint_final.pt").exists():
        steps.append(10**9)
    return sorted(set(steps))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dir = Path(args.run_dir)
    cfg = ModAddGrokkingConfig(p=args.p, frac_train=args.frac_train, seed=args.seed)
    ds = ModAddFullDataset(cfg)
    model_type = args.model_type
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)

    trajectory: List[Dict[str, Any]] = []
    ckpt_paths = sorted(run_dir.glob("checkpoint_step*.pt"))
    if (run_dir / "checkpoint_final.pt").exists():
        ckpt_paths.append(run_dir / "checkpoint_final.pt")

    for ckpt_path in ckpt_paths:
        if model_type == "nanda":
            logits_flat, ds_cfg, model = eval_all_pairs_logits_from_checkpoint(str(ckpt_path), model_type, device)
            cfg = ds_cfg
        else:
            model, raw_cfg = load_model_for_analysis(str(ckpt_path), model_type, device)
            ds_cfg = mod_add_dataset_config(raw_cfg) if not isinstance(raw_cfg, ModAddGrokkingConfig) else raw_cfg
            logits_flat, _ = eval_all_pairs_logits_and_latent(model, ds_cfg, device)  # type: ignore[arg-type]
            cfg = ds_cfg
        grid = logits_grid(logits_flat, cfg.p)
        if isinstance(model, NandaOneLayerTransformer):
            w_e = model.embed_tokens.embedding_weight.detach()
            w_u = model.lm_head.weight.detach()
        else:
            w_e = model.model.inner.embed_tokens.embedding_weight.detach()
            w_u = model.model.inner.lm_head.weight.detach()
        bundle = progress_measure_bundle(grid, labels, train_m, test_m, [], w_e, w_u)
        key_freqs = bundle["key_frequencies"]
        step = ckpt_path.stem.replace("checkpoint_step", "").replace("checkpoint_final", "final")
        row = {
            "checkpoint": ckpt_path.name,
            "step": step,
            "key_frequencies": key_freqs,
            **{k: bundle[k] for k in (
                "full_loss_train", "full_loss_test", "excluded_loss_train", "excluded_loss_test",
                "trig_loss_train", "trig_loss_test", "embedding_gini", "unembed_gini",
            )},
            "logit_trig_fve_mean": bundle["logit_trig_fve"]["fve_mean"],
            "logit_trig_fve_faithful_mean": bundle["logit_trig_fve_faithful"]["fve_mean"],
        }
        trajectory.append(row)
        print(f"{ckpt_path.name}: trig_test={row['trig_loss_test']:.4f} excluded_test={row['excluded_loss_test']:.4f}")

    results = {"trajectory": trajectory, "run_dir": str(run_dir)}
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "progress_trajectory.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/trajectory")
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--frac-train", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-type", default="trm_full", choices=["nanda", "trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
