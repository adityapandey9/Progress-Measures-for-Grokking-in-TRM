"""Base TRM for latent grokking mechanistic interpretability (modular addition).

Display name: ``base-mechanistic-interpretability`` (BMI).
Python package: ``pluto.trm.base_mechanistic_interpretability``.

Note: the top-level model class (TinyRecursiveReasoningModel_BMI) is imported
lazily so that analysis/figure scripts can be used without torch installed.
"""

__all__ = ["TinyRecursiveReasoningModel_BMI"]


def __getattr__(name: str):
    if name == "TinyRecursiveReasoningModel_BMI":
        from pluto.trm.base_mechanistic_interpretability.trm import (  # noqa: PLC0415
            TinyRecursiveReasoningModel_BMI,
        )
        return TinyRecursiveReasoningModel_BMI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
