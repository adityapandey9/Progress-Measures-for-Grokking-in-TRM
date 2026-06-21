"""Unified model loading for Nanda baseline and TRM variants."""

from __future__ import annotations

from typing import Literal, Tuple, Union

import torch

from pluto.trm.base_mechanistic_interpretability.config import (
    ModAddGrokkingConfig,
    NandaBaselineConfig,
    mod_add_dataset_config,
    trm_full_config,
    trm_minimal_config,
)
from pluto.trm.base_mechanistic_interpretability.dataset.mod_add import all_pairs_batch
from pluto.trm.base_mechanistic_interpretability.models.nanda_one_layer import (
    NandaFaithfulTransformer,
    NandaOneLayerTransformer,
)
from pluto.trm.base_mechanistic_interpretability.models.trm_direct import TrmDirectModel
from pluto.trm.base_mechanistic_interpretability.models.trm_nanda_bypass import TrmNandaBypassACTV1
from pluto.trm.base_mechanistic_interpretability.train_trm_direct import build_trm_direct_model
from pluto.trm.base_mechanistic_interpretability.trm import TinyRecursiveReasoningModel_BMI

from pluto.trm.models.losses import ACTLossHead

ModelType = Literal["nanda", "trm_minimal", "trm_full"]
LoadedModel = Union[NandaOneLayerTransformer, NandaFaithfulTransformer, ACTLossHead, TrmDirectModel]
LoadedConfig = Union[NandaBaselineConfig, ModAddGrokkingConfig]


def config_for_model_type(model_type: ModelType, **kwargs: object) -> LoadedConfig:
    if model_type == "nanda":
        return NandaBaselineConfig(**kwargs)  # type: ignore[arg-type]
    if model_type == "trm_minimal":
        return trm_minimal_config(**kwargs)  # type: ignore[arg-type]
    return trm_full_config(**kwargs)  # type: ignore[arg-type]


def load_model_for_analysis(
    checkpoint: str,
    model_type: ModelType,
    device: torch.device,
) -> Tuple[LoadedModel, LoadedConfig]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if model_type == "nanda":
        raw = ckpt.get("config")
        cfg = raw if isinstance(raw, NandaBaselineConfig) else NandaBaselineConfig()
        model = (NandaFaithfulTransformer(cfg) if getattr(cfg, "faithful", False) else NandaOneLayerTransformer(cfg)).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return model, cfg

    raw = ckpt.get("config")
    if isinstance(raw, ModAddGrokkingConfig):
        cfg = raw
    elif isinstance(raw, dict):
        cfg = ModAddGrokkingConfig(**raw)
    else:
        cfg = config_for_model_type(model_type)  # type: ignore[assignment]

    assert isinstance(cfg, ModAddGrokkingConfig)
    bs = all_pairs_batch(cfg, train_only=True)["inputs"].shape[0]
    if getattr(cfg, "trm_direct_mode", ""):
        model = build_trm_direct_model(cfg, bs)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device).eval()
        return model, cfg
    if getattr(cfg, "nanda_bypass", False):
        inner = TrmNandaBypassACTV1(cfg, batch_size=bs)
    else:
        inner = TinyRecursiveReasoningModel_BMI(cfg.to_model_dict(batch_size=bs))
    model = ACTLossHead(inner, loss_type=cfg.loss_type)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, cfg
