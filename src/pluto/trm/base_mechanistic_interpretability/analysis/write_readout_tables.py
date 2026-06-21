#!/usr/bin/env python3
"""Emit the W_L projection readout, single-neuron readout, and attention-head
trig tables (paper v2 Tables 2, A1, 3) from the mainline-seed analysis JSONs.

These three table bodies were previously placeholder stubs, which rendered as
caption-only floats in the compiled PDF. This script regenerates them from the
ground-truth analysis artifacts so the tables carry real numbers.

Usage:
    python -m pluto.trm.base_mechanistic_interpretability.analysis.write_readout_tables \
        --seed-dir .bmi-remote-results/paper_v2_arc_a/ep1/wd_1.0/trm_minimal/seed_0 \
        --tables-dir pluto/trm/base_mechanistic_interpretability/paper_v2/tables
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _component_math(name: str) -> str:
    """``cos(w_14c)`` -> ``$\\cos(w_{14}c)$`` for LaTeX."""
    m = re.match(r"(cos|sin)\(w_(\d+)c\)", name)
    if not m:
        return name
    fn, freq = m.group(1), m.group(2)
    return f"$\\{fn}(w_{{{freq}}}c)$"


def _write(path: Path, body: str) -> None:
    path.write_text(body.rstrip() + "\n")
    print("wrote", path)


def write_projection_readout(rows: List[Dict[str, Any]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Component & Freq $k$ & $W_L$-proj.\ FVE (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_component_math(r['component'])} & {int(r['frequency'])} & {float(r['fve']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines))


def write_single_neuron_readout(rows: List[Dict[str, Any]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Component & Freq $k$ & Best single-neuron FVE (\%) & $W_L$-proj.\ FVE (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_component_math(r['component'])} & {int(r['frequency'])} & "
            f"{float(r.get('neuron_fve', 0.0)):.1f} & {float(r['fve']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines))


def write_attention_heads(rows: List[Dict[str, Any]], out: Path) -> None:
    lines = [
        r"\begin{tabular}{rrrl}",
        r"\toprule",
        r"Head & Dom.\ freq $k$ & Trig FVE (\%) & Target \\",
        r"\midrule",
    ]
    for r in rows:
        head = f"L{int(r['layer'])}H{int(r['head'])}"
        lines.append(
            f"{head} & {int(r['dominant_frequency'])} & {float(r['fve_pct']):.2f} & {r.get('target', '--')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-dir", required=True, help="mainline seed dir holding analysis/")
    ap.add_argument("--tables-dir", required=True, help="paper_v2/tables output dir")
    args = ap.parse_args()

    seed = Path(args.seed_dir)
    tables = Path(args.tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    readout = json.loads((seed / "analysis/readout/nanda_readout.json").read_text())
    rows = readout.get("readout_rows", [])
    write_projection_readout(rows, tables / "neuron_readout.tex")
    write_single_neuron_readout(rows, tables / "neuron_readout_appendix.tex")

    head_path = seed / "analysis/head_trig/head_trig_fits.json"
    if head_path.exists():
        heads = json.loads(head_path.read_text()).get("rows", [])
        write_attention_heads(heads, tables / "attention_heads.tex")


if __name__ == "__main__":
    main()
