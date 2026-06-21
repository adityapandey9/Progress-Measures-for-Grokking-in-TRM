#!/usr/bin/env python3
"""Per-ACT-step latent progress measures for full TRM (Nanda stack on z_t readout).

At each ACT step the recursive model emits logits; we reshape them to the (a, b, c)
grid and run Nanda's progress-measure stack to test whether the Fourier circuit is
present in intermediate latent readouts even when the *final* logits collapse.
Also computes a causal-lite per-step key-frequency ablation (Delta test CE), and a
training-trajectory mode that tracks a latent progress measure across checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch

from pluto.trm.base_mechanistic_interpretability.analysis.common import (
    ensure_dir,
    load_analysis_bundle,
    rollout_act_steps,
    save_json,
)
from pluto.trm.models.losses import ACTLossHead
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch, ModAddFullDataset
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    calculate_excluded_loss,
    fourier_basis,
    identify_key_frequencies_adaptive,
    progress_measure_bundle,
)


def _grid(step_logits: torch.Tensor, p: int) -> torch.Tensor:
    """[p*p, vocab] logits at '=' -> [p, p, p] grid over (a, b, c)."""
    return step_logits[:, 2, :p].reshape(p, p, p)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg, w_e, w_u = load_analysis_bundle(args.checkpoint, args.model_type, device)
    if not isinstance(model, ACTLossHead):
        raise ValueError("latent progress measures require a TRM (ACT) checkpoint")

    ds = ModAddFullDataset(cfg)
    labels = ds.labels[:, 2].to(device)
    train_mask = ds.train_mask.to(device)
    test_mask = ds.test_mask.to(device)
    p = cfg.p

    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    steps = rollout_act_steps(model, batch, max_steps=int(model.model.config.halt_max_steps))

    # Fix key frequencies from the final-step readout for comparability across steps.
    final_grid = _grid(steps[-1]["logits"], p)
    key_freqs = identify_key_frequencies_adaptive(final_grid, labels, train_mask, test_mask, p)

    per_step: List[Dict[str, Any]] = []
    for t, s in enumerate(steps):
        grid = _grid(s["logits"], p)
        bundle = progress_measure_bundle(grid, labels, train_mask, test_mask, key_freqs, w_e, w_u)
        per_step.append(
            {
                "act_step": t,
                "logit_trig_fve_adaptive": float(bundle["logit_trig_fve_adaptive"]["fve_mean"]),
                "logit_trig_fve_faithful": float(bundle["logit_trig_fve_faithful"]["fve_mean"]),
                "trig_loss_test": float(bundle["trig_loss_test"]),
                "excluded_loss_test": float(bundle["excluded_loss_test"]),
                "full_loss_test": float(bundle["full_loss_test"]),
            }
        )

    # Causal-lite: per ACT step, ablate the key frequencies and measure the test-CE jump.
    basis = fourier_basis(p, device)
    for t, s in enumerate(steps):
        grid = _grid(s["logits"], p)
        excl, _ = calculate_excluded_loss(grid, key_freqs, labels, train_mask, test_mask, basis, mode="test")
        full = float(per_step[t]["full_loss_test"])
        per_step[t]["ablate_key_freqs_test_ce"] = float(excl)
        per_step[t]["causal_delta_ce"] = float(excl) - full

    fves = [r["logit_trig_fve_adaptive"] for r in per_step]
    results = {
        "checkpoint": str(args.checkpoint),
        "model_type": args.model_type,
        "key_frequencies": list(key_freqs),
        "n_act_steps": len(steps),
        "per_step": per_step,
        "final_step_fve_adaptive": fves[-1] if fves else 0.0,
        "peak_step_fve_adaptive": max(fves) if fves else 0.0,
        "peak_act_step": int(max(range(len(fves)), key=lambda i: fves[i])) if fves else -1,
    }
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "latent_progress_measures.json", results)
    return results


def run_trajectory(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir)
    numbered = sorted(
        run_dir.glob("checkpoint_step*.pt"),
        key=lambda q: int(re.search(r"step(\d+)", q.name).group(1)),
    )

    acc_by_step: Dict[int, float] = {}
    hist_path = run_dir / "training_history.json"
    if hist_path.exists():
        for row in json.loads(hist_path.read_text()):
            if str(row.get("step", "")).isdigit():
                acc_by_step[int(row["step"])] = float(row.get("test_acc", 0.0))

    # Resolve checkpoint_final.pt to the true final training step (so it is not
    # plotted at a sentinel x-position), and skip it when it merely duplicates
    # the last numbered checkpoint.
    max_numbered = int(re.search(r"step(\d+)", numbered[-1].name).group(1)) if numbered else 0
    final_step = max(acc_by_step) if acc_by_step else max_numbered
    ckpts: List[tuple[int, Path]] = [
        (int(re.search(r"step(\d+)", q.name).group(1)), q) for q in numbered
    ]
    final_ck = run_dir / "checkpoint_final.pt"
    if final_ck.exists() and final_step > max_numbered:
        ckpts.append((final_step, final_ck))

    traj: List[Dict[str, Any]] = []
    for step, ck in ckpts:
        sub = argparse.Namespace(
            checkpoint=str(ck),
            output_dir=str(Path(args.output_dir) / ck.stem),
            model_type=args.model_type,
            cpu=args.cpu,
        )
        r = run(sub)
        traj.append(
            {
                "step": step,
                "latent_progress": r["peak_step_fve_adaptive"],
                "final_step_fve": r["final_step_fve_adaptive"],
                "test_acc": acc_by_step.get(step, None),
            }
        )
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "latent_progress_trajectory.json", {"trajectory": traj})
    return {"trajectory": traj}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint")
    ap.add_argument("--run-dir")
    ap.add_argument("--trajectory", action="store_true")
    ap.add_argument("--output-dir", default="bmi_analysis/latent_progress")
    ap.add_argument("--model-type", default="trm_full", choices=["trm_minimal", "trm_full"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.trajectory:
        if not args.run_dir:
            ap.error("--trajectory requires --run-dir")
        r = run_trajectory(args)
        t = r["trajectory"]
        print(f"trajectory steps={len(t)} latent_progress={[round(x['latent_progress'], 3) for x in t]}")
        return
    if not args.checkpoint:
        ap.error("single-checkpoint mode requires --checkpoint")
    r = run(args)
    print(
        f"final_fve={r['final_step_fve_adaptive']:.4f} "
        f"peak_fve={r['peak_step_fve_adaptive']:.4f}@step{r['peak_act_step']} "
        f"key_freqs={r['key_frequencies']}"
    )


if __name__ == "__main__":
    main()
