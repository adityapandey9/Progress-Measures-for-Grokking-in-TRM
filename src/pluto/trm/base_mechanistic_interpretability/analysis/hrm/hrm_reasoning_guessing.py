#!/usr/bin/env python3
"""HRM reasoning-vs-guessing probe entry point (arXiv:2601.10679)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.hrm._core import run_hrm_probes
from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig, mod_add_dataset_config
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch
from pluto.trm.models.losses import ACTLossHead


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg = load_model_for_analysis(args.checkpoint, args.model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("HRM probes require TRM (trm_minimal or trm_full), not nanda baseline")

    ds_cfg = mod_add_dataset_config(cfg) if not isinstance(cfg, ModAddGrokkingConfig) else cfg
    test_batch = {k: v.to(device) for k, v in all_pairs_batch(ds_cfg, test_only=True).items()}
    results = run_hrm_probes(model, test_batch, max_probe_samples=args.batch_size)
    results.update(
        {
            "variant": args.model_type,
            "task": "modular_addition",
            "checkpoint": args.checkpoint,
            "model_type": args.model_type,
        }
    )
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "hrm_reasoning_guessing.json", results)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/hrm")
    p.add_argument("--model-type", default="trm_full", choices=["trm_minimal", "trm_full"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run(args)
    print(f"modes={r['reasoning_mode_counts']} fp_viol={r['fixed_point_violation_rate']:.4f}")


if __name__ == "__main__":
    main()
