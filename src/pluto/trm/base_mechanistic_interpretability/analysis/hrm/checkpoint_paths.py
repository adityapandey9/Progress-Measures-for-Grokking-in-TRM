#!/usr/bin/env python3
"""Resolve checkpoint paths for final / grokking / best-FVE from training history."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from pluto.trm.base_mechanistic_interpretability.analysis.checkpoint_selection import select_checkpoints


def _step_from_name(name: str) -> int:
    if name == "checkpoint_final":
        return 10**9
    m = re.search(r"checkpoint_step(\d+)", name)
    return int(m.group(1)) if m else -1


def checkpoint_path_for_step(run_dir: Path, step: int) -> Optional[Path]:
    if step < 0:
        return None
    candidates = sorted(run_dir.glob("checkpoint_step*.pt"), key=lambda p: _step_from_name(p.stem))
    if not candidates:
        final = run_dir / "checkpoint_final.pt"
        return final if final.exists() else None
    best = min(candidates, key=lambda p: abs(_step_from_name(p.stem) - step))
    if abs(_step_from_name(best.stem) - step) <= 2000:
        return best
    final = run_dir / "checkpoint_final.pt"
    if step >= 10**8 and final.exists():
        return final
    return best if best.exists() else None


def selected_checkpoint_paths(run_dir: Path) -> Dict[str, Optional[Path]]:
    hist_path = run_dir / "training_history.json"
    if not hist_path.exists():
        final = run_dir / "checkpoint_final.pt"
        return {
            "final": final if final.exists() else None,
            "grokking": None,
            "best_fve": None,
        }
    history = json.loads(hist_path.read_text())
    selected = select_checkpoints(history)
    out: Dict[str, Optional[Path]] = {}
    for key, ck_key in [
        ("final", "final_checkpoint"),
        ("grokking", "grokking_checkpoint"),
        ("best_fve", "best_fve_checkpoint"),
    ]:
        row = selected.get(ck_key)
        if not row:
            out[key] = None
            continue
        step = int(row.get("step", -1))
        out[key] = checkpoint_path_for_step(run_dir, step)
    if out["final"] is None and (run_dir / "checkpoint_final.pt").exists():
        out["final"] = run_dir / "checkpoint_final.pt"
    return out
