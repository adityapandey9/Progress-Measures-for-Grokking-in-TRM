"""Checkpoint selection rules for hybrid latent-grokking paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _fve(row: Dict[str, Any]) -> float:
    value = row.get("logit_trig_fve_faithful", row.get("logit_trig_fve", 0.0))
    if isinstance(value, dict):
        return float(value.get("fve_mean", value.get("fve_faithful", 0.0)))
    return float(value or 0.0)


def _eligible_grokking(row: Dict[str, Any], *, acc_threshold: float, loss_threshold: float) -> bool:
    return float(row.get("test_acc", row.get("test_accuracy", 0.0))) >= acc_threshold and float(
        row.get("test_loss", row.get("full_loss_test", float("inf")))
    ) <= loss_threshold


def select_checkpoints(
    history: Iterable[Dict[str, Any]],
    *,
    acc_threshold: float = 0.99,
    loss_threshold: float = 0.05,
) -> Dict[str, Optional[Dict[str, Any]]]:
    rows = sorted(list(history), key=lambda r: int(r["step"]) if str(r["step"]).isdigit() else 10**12)
    if not rows:
        return {"final_checkpoint": None, "grokking_checkpoint": None, "best_fve_checkpoint": None}

    final = rows[-1]
    strict = [r for r in rows if _eligible_grokking(r, acc_threshold=acc_threshold, loss_threshold=loss_threshold)]
    loose = [r for r in rows if float(r.get("test_acc", r.get("test_accuracy", 0.0))) >= acc_threshold]
    grokking = strict[0] if strict else (loose[0] if loose else None)
    best_fve = max(loose, key=_fve) if loose else max(rows, key=_fve)
    return {
        "final_checkpoint": final,
        "grokking_checkpoint": grokking,
        "best_fve_checkpoint": best_fve,
    }


def select_checkpoints_from_file(path: str | Path) -> Dict[str, Optional[Dict[str, Any]]]:
    return select_checkpoints(json.loads(Path(path).read_text()))
