"""Shared utilities for BMI research analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# torch and torch-dependent imports are deferred so that lightweight figure
# scripts (figstyle, plot_v2_figures, plot_latent_figures) can be imported in
# environments without torch installed.  Only the checkpoint-loading /
# evaluation helpers actually need torch.


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: Any) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(x: Any) -> Any:
    try:
        import torch  # noqa: PLC0415
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


def load_analysis_bundle(
    checkpoint: str,
    model_type: str,
    device: "torch.device",
) -> "tuple[object, Any, Any, Any]":
    """Load checkpoint; return (model, dataset_cfg, w_E, w_U)."""
    import torch  # noqa: PLC0415
    from pluto.trm.base_mechanistic_interpretability.config import mod_add_dataset_config  # noqa: PLC0415
    from pluto.trm.base_mechanistic_interpretability.models.nanda_one_layer import NandaOneLayerTransformer  # noqa: PLC0415
    from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis  # noqa: PLC0415

    model, raw_cfg = load_model_for_analysis(checkpoint, model_type, device)  # type: ignore[arg-type]
    cfg = mod_add_dataset_config(raw_cfg)
    if isinstance(model, NandaOneLayerTransformer):
        w_e = model.embed_tokens.embedding_weight.detach()
        w_u = model.lm_head.weight.detach()
    else:
        w_e = model.model.inner.embed_tokens.embedding_weight.detach()
        inner = model.model.inner
        if hasattr(inner, "nanda"):
            w_u = inner.nanda.W_U.detach().T
        else:
            w_u = model.model.inner.lm_head.weight.detach()
    return model, cfg, w_e, w_u


def load_bmi_model(
    checkpoint: str,
    cfg: Any,
    *,
    device: "torch.device",
) -> Any:
    import torch  # noqa: PLC0415
    from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch  # noqa: PLC0415
    from pluto.trm.base_mechanistic_interpretability.trm import TinyRecursiveReasoningModel_BMI  # noqa: PLC0415
    from pluto.trm.models.losses import ACTLossHead  # noqa: PLC0415

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    bs = all_pairs_batch(cfg, train_only=True)["inputs"].shape[0]
    if getattr(cfg, "nanda_bypass", False):
        from pluto.trm.base_mechanistic_interpretability.models.trm_nanda_bypass import TrmNandaBypassACTV1  # noqa: PLC0415

        inner = TrmNandaBypassACTV1(cfg, batch_size=bs)
    else:
        inner = TinyRecursiveReasoningModel_BMI(cfg.to_model_dict(batch_size=bs))
    model = ACTLossHead(inner, loss_type=cfg.loss_type)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def token_ce(logits: Any, labels: Any) -> Any:
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from pluto.trm.models.losses import IGNORE_LABEL_ID  # noqa: PLC0415

    mask = labels != IGNORE_LABEL_ID
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_LABEL_ID, reduction="none")
    ce = ce.view(labels.shape)
    return torch.where(mask, ce, torch.zeros_like(ce))


def seq_exact(logits: Any, labels: Any) -> Any:
    from pluto.trm.models.losses import IGNORE_LABEL_ID  # noqa: PLC0415

    mask = labels != IGNORE_LABEL_ID
    correct = mask & (logits.argmax(dim=-1) == labels)
    counts = mask.sum(-1).clamp_min(1)
    return correct.sum(-1) == counts


def eval_all_pairs_logits_from_checkpoint(
    checkpoint: str,
    model_type: str,
    device: Any,
    *,
    prefer_ptrm: bool = True,
) -> Any:
    """Load checkpoint and return logits [p*p, p] at '='.

    When ``prefer_ptrm`` and ``ptrm/ptrm_logits.pt`` exists beside the checkpoint,
    returns PTRM-selected logits (arXiv:2605.19943 best-Q@K rollouts).
    """
    import torch  # noqa: PLC0415
    from pluto.trm.base_mechanistic_interpretability.config import mod_add_dataset_config  # noqa: PLC0415
    from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch  # noqa: PLC0415
    from pluto.trm.models.losses import ACTLossHead, IGNORE_LABEL_ID  # noqa: PLC0415
    from pluto.trm.base_mechanistic_interpretability.analysis.model_factory import load_model_for_analysis  # noqa: PLC0415

    ckpt_path = Path(checkpoint)
    ptrm_path = ckpt_path.parent / "ptrm" / "ptrm_logits.pt"
    ptrm_summary_path = ckpt_path.parent / "ptrm" / "ptrm_summary.json"
    if prefer_ptrm and ptrm_path.exists() and ptrm_summary_path.exists():
        summary = json.loads(ptrm_summary_path.read_text())
        det_fve = float(summary.get("deterministic_fve_adaptive", 0))
        ptrm_fve = float(summary.get("ptrm_fve_adaptive", 0))
        if ptrm_fve >= det_fve:
            model, cfg = load_model_for_analysis(checkpoint, model_type, device)  # type: ignore[arg-type]
            ds_cfg = mod_add_dataset_config(cfg)
            payload = torch.load(ptrm_path, map_location=device, weights_only=False)
            logits = payload["logits"].to(device)
            if logits.dim() == 2:
                return logits, ds_cfg, model

    model, cfg = load_model_for_analysis(checkpoint, model_type, device)  # type: ignore[arg-type]
    ds_cfg = mod_add_dataset_config(cfg)
    batch = {k: v.to(device) for k, v in all_pairs_batch(ds_cfg).items()}
    if not isinstance(model, ACTLossHead):  # NandaOneLayer or NandaFaithful
        logits = model(batch["inputs"])[:, 2, : model.cfg.p]
        return logits, ds_cfg, model
    assert isinstance(model, ACTLossHead)
    carry = model.initial_carry(batch)
    carry, outputs = model.model(carry=carry, batch=batch)
    logits = outputs["logits"][:, 2, : ds_cfg.p]
    return logits, ds_cfg, model


def eval_all_pairs_logits(model: Any, cfg: Any, device: Any) -> Any:
    """Return logits [p*p, p] for all (a,b) at '=' position."""
    logits, _ = eval_all_pairs_logits_and_latent(model, cfg, device)
    return logits


def eval_all_pairs_logits_and_latent(
    model: Any, cfg: Any, device: Any
) -> Any:
    """Return logits [p*p, p] and latent z_H at '=' [p*p, hidden]."""
    from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch  # noqa: PLC0415

    batch = {k: v.to(device) for k, v in all_pairs_batch(cfg).items()}
    carry = model.initial_carry(batch)
    carry, outputs = model.model(carry=carry, batch=batch)
    logits = outputs["logits"][:, 2, : cfg.p]
    z_h = carry.inner_carry.z_H[:, 2, :].detach()
    return logits, z_h


def rollout_act_steps(
    model: Any,
    batch: Dict[str, Any],
    *,
    max_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Collect per-ACT-step logits and metrics (HRM §4 segment analysis)."""
    device = batch["inputs"].device
    carry = model.initial_carry(batch)
    limit = max_steps or int(model.model.config.halt_max_steps)
    steps: List[Dict[str, Any]] = []
    for _ in range(limit):
        carry, outputs = model.model(carry=carry, batch=batch)
        labels = carry.current_data["labels"]
        logits = outputs["logits"]
        z_h = carry.inner_carry.z_H.detach()
        steps.append(
            {
                "logits": logits,
                "labels": labels,
                "z_H": z_h,
                "ce": token_ce(logits, labels).mean(-1),
                "exact": seq_exact(logits, labels),
                "halted": carry.halted.clone(),
                "steps": carry.steps.clone(),
            }
        )
        if carry.halted.all():
            break
    return steps
