"""Collect MLP neuron activations at the ``='' token (Nanda ``calculate_key_freqs`` input)."""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig, NandaBaselineConfig
from pluto.trm.base_mechanistic_interpretability.models.nanda_one_layer import (
    NandaFaithfulTransformer,
    NandaOneLayerTransformer,
)
from pluto.trm.base_mechanistic_interpretability.models.trm_direct import TrmDirectModel
from pluto.trm.base_mechanistic_interpretability.models.trm_nanda_bypass import TrmNandaBypassInner
from pluto.trm.models.losses import ACTLossHead


@torch.no_grad()
def collect_mlp_neuron_acts_at_equals(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    eq_token_idx: int = 2,
) -> torch.Tensor:
    """Return post-activation MLP hidden states ``[n_pairs, n_neurons]`` at ``=''.

    Matches Nanda ``blocks.0.mlp.hook_post[:, -1]`` up to SwiGLU vs ReLU differences.
    """
    storage: List[torch.Tensor] = []
    handles: list[Any] = []

    def _save(x: torch.Tensor) -> None:
        storage.append(x[:, eq_token_idx, :].detach())

    if isinstance(model, ACTLossHead):
        inner = model.model.inner
        if isinstance(inner, TrmNandaBypassInner):
            nanda = inner.nanda

            def faithful_hook(mod: NandaFaithfulTransformer, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
                x = inp[0]
                h = torch.relu(torch.einsum("md,bpd->bpm", mod.W_in, x) + mod.b_in)
                storage.append(h[:, eq_token_idx, :].detach())

            handles.append(nanda.mlp.register_forward_hook(faithful_hook))
            carry = model.initial_carry(batch)
            model.model(carry=carry, batch=batch)
        else:
            block = inner.L_level.layers[-1]
            handles.append(
                block.mlp.down_proj.register_forward_pre_hook(
                    lambda _m, inp: _save(inp[0])
                )
            )
            carry = model.initial_carry(batch)
            model.model(carry=carry, batch=batch)
    elif isinstance(model, TrmDirectModel):
        if model.mode.startswith("flat"):
            block = model.encoder[0]
        else:
            block = model.L_level.layers[0]
        handles.append(
            block.mlp.down_proj.register_forward_pre_hook(
                lambda _m, inp: _save(inp[0])
            )
        )
        model(batch)
    elif isinstance(model, NandaFaithfulTransformer):

        def faithful_hook(mod: NandaFaithfulTransformer, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            x = inp[0]
            h = torch.relu(torch.einsum("md,bpd->bpm", mod.W_in, x) + mod.b_in)
            storage.append(h[:, :, eq_token_idx].detach())

        handles.append(model.mlp.register_forward_hook(faithful_hook))
        model(batch["inputs"])
    elif isinstance(model, NandaOneLayerTransformer):
        handles.append(
            model.block.mlp.down_proj.register_forward_pre_hook(
                lambda _m, inp: _save(inp[0])
            )
        )
        model(batch["inputs"])
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    for h in handles:
        h.remove()

    if not storage:
        raise RuntimeError("MLP activation hook did not fire")
    return storage[0]
