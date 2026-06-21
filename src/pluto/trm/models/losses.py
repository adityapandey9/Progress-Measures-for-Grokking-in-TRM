from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0)


def cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    """Standard CE (Nanda fidelity A); same signature as stablemax_cross_entropy."""
    if valid_mask is None:
        valid_mask = labels != ignore_index
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="none")
    return ce.view(labels.shape) * valid_mask.to(ce.dtype)


def cross_entropy_high_precision(logits, labels, ignore_index: int = -100, valid_mask=None):
    """Float64 log-softmax CE (Nanda faithful); same tensor shape as ``cross_entropy``."""
    if valid_mask is None:
        valid_mask = labels != ignore_index
    logprobs = F.log_softmax(logits.double(), dim=-1)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, torch.zeros_like(prediction_logprobs))


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum"
        )
        metrics.update({"lm_loss": lm_loss.detach(), "q_halt_loss": q_halt_loss.detach()})
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()
        inner_q_halt_loss = torch.zeros((), device=lm_loss.device, dtype=lm_loss.dtype)
        if "inner_q_halt_logits" in outputs:
            iq = outputs["inner_q_halt_logits"]
            it = outputs["inner_seq_correct"]
            inner_q_halt_loss = F.binary_cross_entropy_with_logits(
                iq.reshape(-1), it.reshape(-1).to(iq.dtype), reduction="sum"
            )
            metrics["inner_q_halt_loss"] = inner_q_halt_loss.detach()
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        total_q = 0.5 * (q_halt_loss + q_continue_loss + inner_q_halt_loss)
        total = lm_loss + total_q
        if "ras_aux_loss" in outputs:
            total = total + outputs["ras_aux_loss"]
            metrics["ras_aux_loss"] = outputs["ras_aux_loss"].detach()
        if "candidate_recall" in outputs:
            metrics["candidate_recall"] = outputs["candidate_recall"].detach()
        if "natural_candidate_recall" in outputs:
            metrics["natural_candidate_recall"] = outputs["natural_candidate_recall"].detach()
        return new_carry, total, metrics, detached_outputs, new_carry.halted.all()


def _dis_flow_total_loss(
    *,
    lm_loss: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if "flow_velocity_pred" not in outputs:
        return lm_loss, {}
    # Lazy import: only flow-matching losses need this; the grokking path returns above.
    from pluto.trm.flow_paths import flow_velocity_mse

    v_loss = flow_velocity_mse(
        outputs["flow_velocity_pred"],
        outputs["flow_velocity_target"],
        outputs["flow_velocity_mask"],
    )
    cfg = getattr(model, "config", None)
    weight = float(getattr(cfg, "dis_flow_loss_weight", 1.0))
    flow_metrics = {"flow_velocity_loss": v_loss.detach()}
    if bool(getattr(cfg, "dis_flow_velocity_only", False)):
        return weight * v_loss, flow_metrics
    return lm_loss + weight * v_loss, flow_metrics


class ACTLossHeadIntermediate(nn.Module):
    """DIS-style head: LM loss on corrupt-path targets only (ACTLossHeadV4); ARC eval slice like ACTLossHeadV5."""

    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds

            # DIS ACTLossHeadV5: one-shot ARC layout [train1 | train2 | test] — log metrics on test third only.
            B, T = labels.shape[0], labels.shape[1]
            metric_slice = slice(0, T)
            if T % 3 == 0 and T >= 900:
                third = T // 3
                metric_slice = slice(2 * third, 3 * third)
            labels_m = labels[:, metric_slice]
            preds_m = preds[:, metric_slice]

            mask = labels_m != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (preds_m == labels_m)
            seq_is_correct_final = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct_final).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, torch.zeros_like(new_carry.steps)).sum(),
            }

        logits = outputs["logits"]
        B, T, V = logits.size(0), logits.size(1), logits.size(-1)
        intermediate = new_carry.intermediate_labels_path
        num_steps = intermediate.size(1)
        if "dis_flow_path_idx" in outputs:
            step_idx = outputs["dis_flow_path_idx"].clamp(0, num_steps - 1).to(torch.long)
        else:
            step_idx = new_carry.steps.clamp(0, num_steps - 1).to(torch.long)
        step_idx = step_idx.view(-1, 1, 1).expand(-1, 1, T)
        current_step_labels = torch.gather(intermediate, 1, step_idx).squeeze(1)

        invalid = (current_step_labels < 0) | (current_step_labels >= V)
        safe_step_labels = current_step_labels.masked_fill(invalid, IGNORE_LABEL_ID)
        step_mask = safe_step_labels != IGNORE_LABEL_ID
        step_loss_counts = step_mask.sum(-1).clamp_min(1)
        step_loss_divisor = step_loss_counts.unsqueeze(-1)

        try:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=step_mask,
            )
        except TypeError:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
            )
        per_token_loss = per_token_loss * step_mask
        lm_loss = (per_token_loss / step_loss_divisor).sum()

        total_loss, flow_metrics = _dis_flow_total_loss(lm_loss=lm_loss, outputs=outputs, model=self.model)
        metrics["lm_loss"] = lm_loss.detach()
        metrics.update(flow_metrics)
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class ACTLossHeadMG(ACTLossHeadIntermediate):
    """DIS intermediate loss + optional Diffusion-wo-CFG EMA logit target shift on the MG batch slice."""

    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        *,
        mg_ema_target_shift: bool = False,
        mg_ema_shift_high: float = 0.75,
        ema_module: Optional[nn.Module] = None,
        train_step_fn: Optional[Callable[[], int]] = None,
        mg_ema_shift_start_step: int = 0,
    ):
        super().__init__(model, loss_type)
        self.mg_ema_target_shift = mg_ema_target_shift
        self.mg_ema_shift_high = mg_ema_shift_high
        self._train_step_fn = train_step_fn
        self.mg_ema_shift_start_step = mg_ema_shift_start_step
        # Not an nn.Module child: attaching the EMA copy via normal setattr would register
        # ``_ema_module.*`` parameters and break EMAHelper (KeyError on update).
        object.__setattr__(self, "_ema_module", ema_module)

    def set_ema_module(self, ema_module: Optional[nn.Module]) -> None:
        object.__setattr__(self, "_ema_module", ema_module)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        logits = outputs["logits"]
        step = int(self._train_step_fn()) if self._train_step_fn is not None else 0
        if (
            self.mg_ema_target_shift
            and self.training
            and self._ema_module is not None
            and step >= self.mg_ema_shift_start_step
            and "mg_num_mg" in outputs
        ):
            from pluto.trm.mg_trm.ema_target_shift import ema_mg_logit_shift

            num_mg = int(outputs["mg_num_mg"].item())
            mg_scale = outputs["mg_scale"]
            batch = model_kwargs.get("batch")
            carry = model_kwargs.get("carry")
            if batch is not None and carry is not None and num_mg > 0:
                delta, w = ema_mg_logit_shift(
                    self._ema_module,
                    carry=carry,
                    batch=batch,
                    num_mg=num_mg,
                    mg_scale=mg_scale,
                    mg_high=self.mg_ema_shift_high,
                )
                if delta.numel() > 0:
                    shift = (w.view(-1, 1, 1) * delta).detach()
                    logits = logits.clone()
                    logits[:num_mg] = logits[:num_mg] + shift
                    outputs["logits"] = logits

        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds

            B, T = labels.shape[0], labels.shape[1]
            metric_slice = slice(0, T)
            if T % 3 == 0 and T >= 900:
                third = T // 3
                metric_slice = slice(2 * third, 3 * third)
            labels_m = labels[:, metric_slice]
            preds_m = preds[:, metric_slice]

            mask = labels_m != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (preds_m == labels_m)
            seq_is_correct_final = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct_final).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, torch.zeros_like(new_carry.steps)).sum(),
            }

        logits = outputs["logits"]
        B, T, V = logits.size(0), logits.size(1), logits.size(-1)
        intermediate = new_carry.intermediate_labels_path
        num_steps = intermediate.size(1)
        if "dis_flow_path_idx" in outputs:
            step_idx = outputs["dis_flow_path_idx"].clamp(0, num_steps - 1).to(torch.long)
        else:
            step_idx = new_carry.steps.clamp(0, num_steps - 1).to(torch.long)
        step_idx = step_idx.view(-1, 1, 1).expand(-1, 1, T)
        current_step_labels = torch.gather(intermediate, 1, step_idx).squeeze(1)

        invalid = (current_step_labels < 0) | (current_step_labels >= V)
        safe_step_labels = current_step_labels.masked_fill(invalid, IGNORE_LABEL_ID)
        step_mask = safe_step_labels != IGNORE_LABEL_ID
        step_loss_divisor = step_mask.sum(-1).clamp_min(1).unsqueeze(-1)

        try:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=step_mask,
            )
        except TypeError:
            per_token_loss = self.loss_fn(
                logits,
                safe_step_labels,
                ignore_index=IGNORE_LABEL_ID,
            )
        per_token_loss = per_token_loss * step_mask
        lm_loss = (per_token_loss / step_loss_divisor).sum()

        total_loss, flow_metrics = _dis_flow_total_loss(lm_loss=lm_loss, outputs=outputs, model=self.model)
        metrics["lm_loss"] = lm_loss.detach()
        metrics.update(flow_metrics)
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
