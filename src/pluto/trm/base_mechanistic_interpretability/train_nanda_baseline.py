#!/usr/bin/env python3
"""Train Nanda one-layer baseline on modular addition (arXiv:2301.05217)."""

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
    NandaBaselineConfig,
    mod_add_dataset_config,
)
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import ModAddFullDataset, all_pairs_batch, save_dataset_artifacts
from pluto.trm.base_mechanistic_interpretability.fourier.progress_measures import (
    embedding_fourier_norms,
    identify_key_frequencies,
    logits_grid,
    progress_measure_bundle,
)
from pluto.trm.base_mechanistic_interpretability.models.nanda_one_layer import (
    NandaFaithfulTransformer,
    NandaOneLayerTransformer,
)
from pluto.trm.models.losses import IGNORE_LABEL_ID


def build_nanda_model(cfg: NandaBaselineConfig):
    """Faithful Nanda transformer when cfg.faithful else the modernized baseline."""
    return NandaFaithfulTransformer(cfg) if cfg.faithful else NandaOneLayerTransformer(cfg)


def _model_w_e_w_u(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Embedding [vocab,hidden] and unembedding [vocab,hidden] for either model."""
    w_e = model.embed_tokens.embedding_weight.detach()
    if hasattr(model, "lm_head"):
        w_u = model.lm_head.weight.detach()
    else:  # faithful model stores W_U as [hidden, vocab]
        w_u = model.W_U.detach().T
    return w_e, w_u


def _high_precision_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """log_softmax in float64 to avoid Nanda's documented float32 underflow spikes."""
    logprobs = F.log_softmax(logits.double(), dim=-1)
    return -logprobs.gather(-1, labels[:, None]).mean()


def _accuracy_at_equals(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != IGNORE_LABEL_ID
    preds = logits.argmax(-1)
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


@torch.no_grad()
def evaluate(model: NandaOneLayerTransformer, batch: Dict[str, torch.Tensor], cfg: NandaBaselineConfig) -> Dict[str, float]:
    model.eval()
    logits = model(batch["inputs"])
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    ce = F.cross_entropy(logits[mask], labels[mask], reduction="mean").item()
    acc = _accuracy_at_equals(logits, labels)
    return {"loss": ce, "accuracy": acc}


@torch.no_grad()
def _progress_metrics(model: NandaOneLayerTransformer, cfg: NandaBaselineConfig, device: torch.device) -> Dict[str, float]:
    ds_cfg = mod_add_dataset_config(cfg)
    batch = {k: v.to(device) for k, v in all_pairs_batch(ds_cfg).items()}
    logits = model(batch["inputs"])[:, 2, : cfg.p]
    from pluto.trm.base_mechanistic_interpretability.analysis.mlp_activations import (
        collect_mlp_neuron_acts_at_equals,
    )

    mlp_acts = collect_mlp_neuron_acts_at_equals(model, batch)
    ds = ModAddFullDataset(ds_cfg)
    labels = ds.labels[:, 2].to(device)
    train_m = ds.train_mask.to(device)
    test_m = ds.test_mask.to(device)
    grid = logits_grid(logits, cfg.p)
    w_e, w_u = _model_w_e_w_u(model)
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


def train(cfg: NandaBaselineConfig, out_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_cfg = mod_add_dataset_config(cfg)
    save_dataset_artifacts(ds_cfg, str(out_dir / "data"))

    train_batch = all_pairs_batch(ds_cfg, train_only=True)
    test_batch = all_pairs_batch(ds_cfg, test_only=True)

    model = build_nanda_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    scheduler = None
    if cfg.faithful or getattr(cfg, "grokking_ce", False):
        warmup = max(1, cfg.warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 1.0))

    train_batch_d = {k: v.to(device) for k, v in train_batch.items()}
    test_batch_d = {k: v.to(device) for k, v in test_batch.items()}
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(1, cfg.max_steps + 1):
        model.train()
        logits = model(train_batch_d["inputs"])
        labels = train_batch_d["labels"]
        mask = labels != IGNORE_LABEL_ID
        if cfg.faithful and not getattr(cfg, "grokking_ce", False):
            loss = _high_precision_ce(logits[mask], labels[mask])
        else:
            loss = F.cross_entropy(logits[mask], labels[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % cfg.log_every == 0 or step == 1:
            train_eval = evaluate(model, train_batch_d, cfg)
            test_eval = evaluate(model, test_batch_d, cfg)
            row: Dict[str, Any] = {
                "step": step,
                "train_loss": train_eval["loss"],
                "train_acc": train_eval["accuracy"],
                "test_loss": test_eval["loss"],
                "test_acc": test_eval["accuracy"],
                "fidelity": cfg.fidelity,
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
                {"step": step, "model": model.state_dict(), "config": cfg, "fidelity": cfg.fidelity},
                out_dir / f"checkpoint_step{step}.pt",
            )

    torch.save(
        {"step": cfg.max_steps, "model": model.state_dict(), "config": cfg, "fidelity": cfg.fidelity},
        out_dir / "checkpoint_final.pt",
    )
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Checkpoints and history in {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="bmi_grokking_runs/nanda_baseline")
    p.add_argument("--max-steps", type=int, default=20_000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--frac-train", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fidelity", default="B", choices=["A", "B"])
    p.add_argument("--attn-only", action="store_true")
    p.add_argument("--faithful", action="store_true", help="Faithful Nanda repro (no-norm, ReLU MLP, warmup, hi-prec CE)")
    p.add_argument(
        "--grokking-ce",
        action="store_true",
        help="Float32 CE (matches train_grokking ACT lm loss) instead of hi-prec CE; use with --faithful",
    )
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    args = p.parse_args()

    cfg = NandaBaselineConfig(
        p=args.p,
        frac_train=args.frac_train,
        seed=args.seed,
        max_steps=args.max_steps,
        save_every=args.save_every,
        fidelity=args.fidelity,
        attn_only=args.attn_only,
        faithful=args.faithful or args.grokking_ce,
        grokking_ce=args.grokking_ce,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
    )
    train(cfg, Path(args.output_dir))


if __name__ == "__main__":
    main()
