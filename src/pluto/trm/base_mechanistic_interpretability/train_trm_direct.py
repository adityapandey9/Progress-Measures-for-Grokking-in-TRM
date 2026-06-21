#!/usr/bin/env python3
"""Train TRM minimal directly (no ACT / q_halt) with hi-prec float64 CE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from pluto.trm.base_mechanistic_interpretability.config import (
    ModAddGrokkingConfig,
    mod_add_dataset_config,
    trm_direct_flat_bidir_config,
    trm_direct_flat_causal_config,
    trm_direct_full_config,
    trm_direct_recursive_config,
)
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset, all_pairs_batch, save_dataset_artifacts
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    embedding_fourier_norms,
    identify_key_frequencies,
    logits_grid,
    progress_measure_bundle,
)
from pluto.trm.base_mechanistic_interpretability.models.trm_direct import TrmDirectModel
from pluto.trm.base_mechanistic_interpretability.train_nanda_baseline import _high_precision_ce
from pluto.trm.models.losses import IGNORE_LABEL_ID
from pluto.trm.models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Config


def build_trm_direct_model(cfg: ModAddGrokkingConfig, batch_size: int) -> TrmDirectModel:
    act_cfg = TinyRecursiveReasoningModel_ACTV1Config(**cfg.to_model_dict(batch_size=batch_size))
    return TrmDirectModel(act_cfg, mode=cfg.trm_direct_mode)


def _accuracy_at_equals(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != IGNORE_LABEL_ID
    preds = logits.argmax(-1)
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


@torch.no_grad()
def evaluate(model: TrmDirectModel, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    model.eval()
    logits = model(batch)
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    ce = F.cross_entropy(logits[mask], labels[mask], reduction="mean").item()
    return {"loss": ce, "accuracy": _accuracy_at_equals(logits, labels)}


@torch.no_grad()
def _progress_metrics(model: TrmDirectModel, cfg: ModAddGrokkingConfig, device: torch.device) -> Dict[str, float]:
    from pluto.trm.base_mechanistic_interpretability.analysis.mlp_activations import collect_mlp_neuron_acts_at_equals

    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    logits = model(batch)[:, 2, : cfg.p]
    mlp_acts = collect_mlp_neuron_acts_at_equals(model, batch)
    ds = ModAddFullDataset(cfg)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)
    w_e = model.embed_tokens.embedding_weight.detach()
    w_u = model.lm_head.weight.detach()
    key_freqs = identify_key_frequencies(embedding_fourier_norms(w_e, cfg.p), top_k=5)
    bundle = progress_measure_bundle(
        grid, labels, train_m, test_m, key_freqs, w_e, w_u, mlp_neuron_acts=mlp_acts
    )
    return {
        "trig_loss_test": bundle["trig_loss_test"],
        "excluded_loss_test": bundle["excluded_loss_test"],
        "logit_trig_fve_adaptive": bundle["logit_trig_fve_adaptive"]["fve_mean"],
        "n_key_frequencies_adaptive": bundle["n_key_frequencies_adaptive"],
    }


def train(cfg: ModAddGrokkingConfig, out_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_artifacts(cfg, str(out_dir / "data"))

    train_batch = all_pairs_batch(cfg, train_only=True)
    test_batch = all_pairs_batch(cfg, test_only=True)
    bs = train_batch["inputs"].shape[0]

    model = build_trm_direct_model(cfg, bs).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    warmup = max(1, cfg.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 1.0))

    train_batch_d = {k: v.to(device) for k, v in train_batch.items()}
    test_batch_d = {k: v.to(device) for k, v in test_batch.items()}
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(1, cfg.max_steps + 1):
        model.train()
        logits = model(train_batch_d)
        labels = train_batch_d["labels"]
        mask = labels != IGNORE_LABEL_ID
        loss = _high_precision_ce(logits[mask], labels[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if step % cfg.log_every == 0 or step == 1:
            train_eval = evaluate(model, train_batch_d)
            test_eval = evaluate(model, test_batch_d)
            row: Dict[str, Any] = {
                "step": step,
                "train_loss": train_eval["loss"],
                "train_acc": train_eval["accuracy"],
                "test_loss": test_eval["loss"],
                "test_acc": test_eval["accuracy"],
                "trm_direct_mode": cfg.trm_direct_mode,
                "elapsed_s": time.time() - t0,
            }
            if step % cfg.eval_every == 0:
                row.update(_progress_metrics(model, cfg, device))
            history.append(row)
            print(
                f"step={step} train_acc={row['train_acc']:.4f} test_acc={row['test_acc']:.4f} "
                f"train_loss={row['train_loss']:.4f} test_loss={row['test_loss']:.4f}",
                flush=True,
            )
            (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

        if step % cfg.save_every == 0:
            torch.save(
                {"step": step, "model": model.state_dict(), "config": cfg},
                out_dir / f"checkpoint_step{step}.pt",
            )

    torch.save(
        {"step": cfg.max_steps, "model": model.state_dict(), "config": cfg},
        out_dir / "checkpoint_final.pt",
    )
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Checkpoints and history in {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--preset", choices=["flat_causal", "flat_bidir", "recursive", "full"], required=True)
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--frac-train", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    common = dict(
        p=args.p,
        frac_train=args.frac_train,
        seed=args.seed,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        save_every=args.save_every,
    )
    if args.preset == "flat_causal":
        cfg = trm_direct_flat_causal_config(**common)
    elif args.preset == "flat_bidir":
        cfg = trm_direct_flat_bidir_config(**common)
    elif args.preset == "recursive":
        cfg = trm_direct_recursive_config(**common)
    else:
        cfg = trm_direct_full_config(**common)
    train(cfg, Path(args.output_dir))


if __name__ == "__main__":
    main()
