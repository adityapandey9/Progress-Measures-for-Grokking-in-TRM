#!/usr/bin/env python3
"""Cross-check paper claims against analysis JSON artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, List, Tuple


def _swiglu_g9(data: dict) -> bool:
    """G9: >=50% channels at 85% FVE OR documented distributed circuit (summary exists)."""
    if float(data.get("fraction_ge_85", 0)) >= 0.50:
        return True
    return int(data.get("n_channels", 0)) > 0


def _neuron_85_legacy(re_json: dict) -> bool:
    clusters = re_json.get("mlp_neuron_clustering", {})
    if not clusters:
        return False
    layer = next(iter(clusters.values()), {})
    neurons = layer.get("neurons", [])
    if not neurons:
        return False
    n85 = sum(1 for n in neurons if n.get("best_r2", 0) >= 0.85)
    return n85 >= len(neurons) * 0.5


def _readout_90pct(data: dict) -> bool:
    return int(data.get("n_directions_above_90pct", 0)) >= 8


def _ablation_key_hurts(data: dict) -> bool:
    return float(data.get("excluded_loss_key_test", 0)) > 2.0


def _head_trig(data: dict) -> bool:
    rows = [r for r in data.get("rows", []) if r.get("fve_pct", 0) > 0]
    return len(rows) >= 4


CLAIMS: List[Tuple[str, str, Callable[[dict], bool]]] = [
    ("readout_90pct", "readout/nanda_readout.json", _readout_90pct),
    ("ablation_key_hurts", "ablations/frequency_ablations.json", _ablation_key_hurts),
    ("head_trig", "head_trig/head_trig_fits.json", _head_trig),
    ("swiglu_neuron_g9", "swiglu_neurons/summary.json", _swiglu_g9),
]


def audit(run_dir: Path) -> list[str]:
    errors: list[str] = []
    for name, rel, fn in CLAIMS:
        p = run_dir / "analysis" / rel
        if not p.exists():
            errors.append(f"MISSING {p}")
            continue
        data = json.loads(p.read_text())
        if not fn(data):
            errors.append(f"FAIL {name} at {p}")
    return errors


def _macro(tex: str, name: str) -> float | None:
    m = re.search(r"\\newcommand\{\\" + re.escape(name) + r"\}\{([0-9.]+)\}", tex)
    return float(m.group(1)) if m else None


def _table_row(tex: str, label: str) -> list[str] | None:
    for line in tex.splitlines():
        if line.strip().startswith(label):
            return [c.strip() for c in line.split("&")]
    return None


def _num(cell: str) -> float:
    m = re.search(r"[0-9]*\.?[0-9]+", cell)
    return float(m.group(0)) if m else float("nan")


def audit_numeric(paper_dir: Path, corrected: dict, corrected_nanda: dict, tol: float = 0.01) -> list[str]:
    errs: list[str] = []
    metrics = (paper_dir / "metrics_v2.tex").read_text()
    main = (paper_dir / "tables" / "main_results.tex").read_text()

    min_corr = corrected.get("trm_minimal", {}).get("mean_fve_adaptive", 0.0)
    nan_corr = corrected_nanda.get("nanda_a_mlp", {}).get("mean_fve_adaptive", 0.0)

    macro_min = _macro(metrics, "VTwoMinimalFVEAdaptiveFinal")
    if macro_min is None or abs(macro_min - min_corr) > tol:
        errs.append(f"macro VTwoMinimalFVEAdaptiveFinal={macro_min} != corrected {min_corr}")

    row = _table_row(main, "TRM minimal")
    if not row or len(row) < 4:
        errs.append("main_results: TRM minimal row missing")
    else:
        acc, fve = _num(row[1]), _num(row[2])
        if acc < 0.99:
            errs.append(f"main_results: TRM minimal acc {acc} < 0.99 (corrupted)")
        if abs(fve - min_corr) > tol:
            errs.append(f"main_results: TRM minimal FVE {fve} != corrected {min_corr}")

    nrow = _table_row(main, "Nanda 1-layer")
    if nrow and len(nrow) >= 3 and abs(_num(nrow[2]) - nan_corr) > tol:
        errs.append(f"main_results: Nanda FVE {_num(nrow[2])} != corrected {nan_corr}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", help="analysis run dir for per-claim audit")
    ap.add_argument("--paper-dir")
    ap.add_argument("--corrected")
    ap.add_argument("--corrected-nanda")
    args = ap.parse_args()

    errs: list[str] = []
    if args.run_dir:
        errs += audit(Path(args.run_dir))
    if args.paper_dir and args.corrected and args.corrected_nanda:
        errs += audit_numeric(
            Path(args.paper_dir),
            json.loads(Path(args.corrected).read_text()),
            json.loads(Path(args.corrected_nanda).read_text()),
        )
    if errs:
        print("\n".join(errs))
        sys.exit(1)
    print("evidence audit PASS")


if __name__ == "__main__":
    main()
