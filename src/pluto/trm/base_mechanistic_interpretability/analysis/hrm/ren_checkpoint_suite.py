#!/usr/bin/env python3
"""Run Ren (2601.10679) probes at final / grokking / best-FVE checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.checkpoint_selection import select_checkpoints
from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.hrm._core import (
    depth_sensitivity,
    ren_mechanism_verdict,
    run_hrm_probes,
)
from pluto.trm.base_mechanistic_interpretability.analysis.hrm.act_depth_ablation import run as run_act_depth
from pluto.trm.base_mechanistic_interpretability.analysis.hrm.checkpoint_paths import selected_checkpoint_paths
from pluto.trm.base_mechanistic_interpretability.analysis.hybrid_aggregate import diagnose_seed_variance
from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis
from pluto.trm.base_mechanistic_interpretability.config import mod_add_dataset_config
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch
from pluto.trm.models.losses import ACTLossHead


def _probe_checkpoint(
    ckpt: Path,
    model_type: str,
    out_dir: Path,
    *,
    cpu: bool = False,
) -> Dict[str, Any]:
    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model_for_analysis(str(ckpt), model_type, device)
    if not isinstance(model, ACTLossHead):
        return {"checkpoint": str(ckpt), "skipped": "not_act_model"}
    halt_max = int(getattr(model.model.config, "halt_max_steps", 1))
    ds_cfg = mod_add_dataset_config(cfg)
    batch = {k: v.to(device) for k, v in all_pairs_batch(ds_cfg, test_only=True).items()}
    hrm = run_hrm_probes(model, batch, max_probe_samples=64)
    act_args = argparse.Namespace(
        checkpoint=str(ckpt),
        output_dir=str(out_dir / "_act_tmp"),
        model_type=model_type,
        depths=[1, 2, 4, 8],
        batch_size=512,
        cpu=cpu,
    )
    act = run_act_depth(act_args)
    return {
        "checkpoint": str(ckpt),
        "hrm": hrm,
        "act_depth_ablation": act,
    }


def run_suite(run_dir: Path, model_type: str, *, cpu: bool = False) -> Dict[str, Any]:
    paths = selected_checkpoint_paths(run_dir)
    out_root = ensure_dir(run_dir / "analysis" / "ren")
    checkpoint_payload: Dict[str, Any] = {}
    for name, ckpt in paths.items():
        if ckpt is None or not ckpt.exists():
            checkpoint_payload[name] = None
            continue
        ck_out = ensure_dir(out_root / name)
        checkpoint_payload[name] = _probe_checkpoint(ckpt, model_type, ck_out, cpu=cpu)
        save_json(ck_out / "ren_probes.json", checkpoint_payload[name])

    history = json.loads((run_dir / "training_history.json").read_text()) if (run_dir / "training_history.json").exists() else []
    nanda_diag = diagnose_seed_variance(history, model_name=run_dir.name)
    final_probe = checkpoint_payload.get("final") or {}
    hrm_final = (final_probe or {}).get("hrm") or {}
    act_final = (final_probe or {}).get("act_depth_ablation") or {}
    act_rows = act_final.get("rows") or []
    halt_max = 1
    final_ckpt = paths.get("final")
    if final_ckpt and final_ckpt.exists():
        try:
            m, _ = load_model_for_analysis(str(final_ckpt), model_type, torch.device("cpu"))
            if isinstance(m, ACTLossHead):
                halt_max = int(getattr(m.model.config, "halt_max_steps", 1))
        except Exception:
            pass
    verdict = ren_mechanism_verdict(
        model_type=model_type,
        nanda_causes=nanda_diag.get("likely_causes") or [],
        fixed_point_violation_rate=float(hrm_final.get("fixed_point_violation_rate", 0.0)),
        depth_acc_drop=depth_sensitivity(act_rows),
        critical_act_step=int(hrm_final.get("critical_act_grokking_step", -1)),
        halt_max_steps=halt_max,
    )
    payload = {
        "run_dir": str(run_dir),
        "model_type": model_type,
        "paper": "2601.10679",
        "checkpoints": checkpoint_payload,
        "nanda_seed_diagnosis": nanda_diag,
        "mechanism_verdict": verdict,
    }
    save_json(out_root / "ren_checkpoint_suite.json", payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-type", required=True, choices=["trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    r = run_suite(Path(args.run_dir), args.model_type, cpu=args.cpu)
    print(r["mechanism_verdict"])


if __name__ == "__main__":
    main()
