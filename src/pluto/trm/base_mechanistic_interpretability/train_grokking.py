#!/usr/bin/env python3
"""Train base TRM on modular addition for latent grokking (arXiv:2301.05217 setup)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from dataclasses import asdict

from pluto.trm.base_mechanistic_interpretability.config import (
    ModAddGrokkingConfig,
    trm_full_config,
    trm_minimal_config,
    trm_minimal_hiprec_act_config,
    trm_minimal_l2_config,
    trm_minimal_nanda_bypass_config,
    nanda_faithful_50k_config,
)
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch, save_dataset_artifacts
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    embedding_fourier_norms,
    identify_key_frequencies,
    logits_grid,
    progress_measure_bundle,
)
from pluto.trm.base_mechanistic_interpretability.models.trm_nanda_bypass import TrmNandaBypassACTV1
from pluto.trm.base_mechanistic_interpretability.trm import TinyRecursiveReasoningModel_BMI
from pluto.trm.models.losses import ACTLossHead, IGNORE_LABEL_ID


def build_grokking_inner(cfg: ModAddGrokkingConfig, batch_size: int):
    if cfg.nanda_bypass:
        return TrmNandaBypassACTV1(cfg, batch_size=batch_size)
    return TinyRecursiveReasoningModel_BMI(cfg.to_model_dict(batch_size=batch_size))


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(-1)
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


@torch.no_grad()
def evaluate(model: ACTLossHead, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, float]:
    model.eval()
    carry = model.initial_carry({k: v.to(device) for k, v in batch.items()})
    while True:
        carry, loss, metrics, _, done = model(carry=carry, batch={k: v.to(device) for k, v in batch.items()}, return_keys=[])
        if done:
            break
    labels = batch["labels"].to(device)
    mask = labels != IGNORE_LABEL_ID
    # Re-run once for logits at halt
    carry = model.initial_carry({k: v.to(device) for k, v in batch.items()})
    carry, outputs = model.model(carry=carry, batch={k: v.to(device) for k, v in batch.items()})
    logits = outputs["logits"]
    ce = F.cross_entropy(logits[mask], labels[mask], reduction="mean").item()
    acc = _accuracy(logits, labels, mask)
    return {"loss": ce, "accuracy": acc}


@torch.no_grad()
def _progress_metrics(model: ACTLossHead, cfg: ModAddGrokkingConfig, device: torch.device) -> Dict[str, float]:
    """Faithful Nanda progress measures on all pairs (eval_every only)."""
    from pluto.trm.base_mechanistic_interpretability.analysis.mlp_activations import (
        collect_mlp_neuron_acts_at_equals,
    )
    from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset

    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    carry = model.initial_carry(batch)
    carry, outputs = model.model(carry=carry, batch=batch)
    logits = outputs["logits"][:, 2, : cfg.p]
    mlp_acts = collect_mlp_neuron_acts_at_equals(model, batch)
    ds = ModAddFullDataset(cfg)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)
    w_e = model.model.inner.embed_tokens.embedding_weight.detach()
    if hasattr(model.model.inner, "nanda"):
        w_u = model.model.inner.nanda.W_U.detach().T
    else:
        w_u = model.model.inner.lm_head.weight.detach()
    key_freqs = identify_key_frequencies(embedding_fourier_norms(w_e, cfg.p), top_k=5)
    bundle = progress_measure_bundle(
        grid, labels, train_m, test_m, key_freqs, w_e, w_u, mlp_neuron_acts=mlp_acts
    )
    return {
        "trig_loss_test": bundle["trig_loss_test"],
        "excluded_loss_test": bundle["excluded_loss_test"],
        "embedding_gini": bundle["embedding_gini"],
        "unembed_gini": bundle["unembed_gini"],
        "logit_trig_fve": bundle["logit_trig_fve"]["fve_mean"],
        "logit_trig_fve_faithful": bundle["logit_trig_fve_faithful"]["fve_mean"],
        "logit_trig_fve_bias_corrected": bundle["logit_trig_fve_bias_corrected"]["fve_mean"],
        "logit_trig_fve_adaptive": bundle["logit_trig_fve_adaptive"]["fve_mean"],
        "logit_trig_fve_neuron_keys": bundle["logit_trig_fve_neuron_keys"]["fve_mean"],
        "n_key_frequencies_adaptive": bundle["n_key_frequencies_adaptive"],
        "n_key_frequencies_neuron": bundle["n_key_frequencies_neuron"],
    }


def train(cfg: ModAddGrokkingConfig, out_dir: Path, *, resume_from: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_artifacts(cfg, str(out_dir / "data"))

    train_batch = all_pairs_batch(cfg, train_only=True)
    test_batch = all_pairs_batch(cfg, test_only=True)
    bs = train_batch["inputs"].shape[0]

    inner = build_grokking_inner(cfg, bs)
    model = ACTLossHead(inner, loss_type=cfg.loss_type).to(device)

    start_step = 1
    history: List[Dict[str, Any]] = []
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        start_step = int(ckpt.get("step", 0)) + 1
        hist_path = out_dir / "training_history.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text())
        print(f"Resuming from step {start_step - 1} via {resume_from}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    scheduler = None
    if cfg.warmup_steps > 0 and (
        cfg.nanda_bypass or cfg.loss_type == "cross_entropy_high_precision"
    ):
        warmup = max(1, cfg.warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 1.0))

    train_batch_d = {k: v.to(device) for k, v in train_batch.items()}
    test_batch_d = {k: v.to(device) for k, v in test_batch.items()}

    carry = model.initial_carry(train_batch_d)
    t0 = time.time()
    for step in range(start_step, cfg.max_steps + 1):
        model.train()
        carry, loss, metrics, _, _ = model(carry=carry, batch=train_batch_d, return_keys=[])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % cfg.log_every == 0 or step == 1:
            train_eval = evaluate(model, train_batch, device)
            test_eval = evaluate(model, test_batch, device)
            row = {
                "step": step,
                "train_loss": train_eval["loss"],
                "train_acc": train_eval["accuracy"],
                "test_loss": test_eval["loss"],
                "test_acc": test_eval["accuracy"],
                "lm_loss": metrics.get("lm_loss", torch.tensor(0.0)).item() if metrics else 0.0,
                "fidelity": cfg.fidelity,
                "loss_type": cfg.loss_type,
                "elapsed_s": time.time() - t0,
            }
            if step % cfg.eval_every == 0:
                row.update(_progress_metrics(model, cfg, device))
            history.append(row)
            msg = (
                f"step={step} train_acc={row['train_acc']:.4f} test_acc={row['test_acc']:.4f} "
                f"train_loss={row['train_loss']:.4f} test_loss={row['test_loss']:.4f}"
            )
            print(msg, flush=True)
            (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

        if step % cfg.save_every == 0:
            ckpt = out_dir / f"checkpoint_step{step}.pt"
            torch.save({"step": step, "model": model.state_dict(), "config": cfg, "fidelity": cfg.fidelity}, ckpt)

    final_ckpt = out_dir / "checkpoint_final.pt"
    torch.save(
        {"step": cfg.max_steps, "model": model.state_dict(), "config": cfg, "fidelity": cfg.fidelity},
        final_ckpt,
    )
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Checkpoints and history in {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="bmi_grokking_runs/default")
    p.add_argument("--max-steps", type=int, default=20_000)
    p.add_argument("--preset", choices=["minimal", "minimal_l2", "minimal_nanda_bypass", "minimal_hiprec_act", "nanda_faithful_50k", "full", "default"], default="default")
    p.add_argument("--fidelity", default="B", choices=["A", "B"])
    p.add_argument("--resume-from", default=None, help="Checkpoint to continue training from")
    p.add_argument("--save-every", type=int, default=None)
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
        save_every=args.save_every if args.save_every is not None else 2000,
    )
    if args.preset == "minimal":
        cfg = trm_minimal_config(**common)
    elif args.preset == "minimal_l2":
        cfg = trm_minimal_l2_config(**common)
    elif args.preset == "minimal_nanda_bypass":
        cfg = trm_minimal_nanda_bypass_config(**common)
    elif args.preset == "minimal_hiprec_act":
        cfg = trm_minimal_hiprec_act_config(**common)
    elif args.preset == "nanda_faithful_50k":
        cfg = nanda_faithful_50k_config(**common)
    elif args.preset == "full":
        cfg = trm_full_config(**common)
    else:
        cfg = ModAddGrokkingConfig(**common)
    if args.fidelity == "A":
        cfg = ModAddGrokkingConfig(**{**asdict(cfg), "loss_type": "cross_entropy", "fidelity": "A"})
    train(cfg, Path(args.output_dir), resume_from=args.resume_from)


if __name__ == "__main__":
    main()
