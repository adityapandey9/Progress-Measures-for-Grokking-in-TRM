"""Thin wrapper: upstream TinyRecursiveModels TRM only (no variant bloat)."""

from __future__ import annotations

from pluto.trm.models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1Config,
)

__all__ = [
    "TinyRecursiveReasoningModel_BMI",
    "TinyRecursiveReasoningModel_BMIConfig",
    "TinyRecursiveReasoningModel_BMICarry",
]

TinyRecursiveReasoningModel_BMI = TinyRecursiveReasoningModel_ACTV1
TinyRecursiveReasoningModel_BMIConfig = TinyRecursiveReasoningModel_ACTV1Config
TinyRecursiveReasoningModel_BMICarry = TinyRecursiveReasoningModel_ACTV1Carry
