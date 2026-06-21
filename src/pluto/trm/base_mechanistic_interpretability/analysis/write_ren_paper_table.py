#!/usr/bin/env python3
"""Write paper/tables/ren_mechanism_summary.tex from ren_diagnosis_report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="bmi_hybrid/aggregate/ren_diagnosis_report.json")
    p.add_argument("--output", default="pluto/trm/base_mechanistic_interpretability/paper/tables/ren_mechanism_summary.tex")
    args = p.parse_args()
    report = json.loads(Path(args.report).read_text())
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Post-grokking mechanism classification (Ren + Nanda probes).}",
        r"  \label{tab:ren_mechanism}",
        r"  \begin{tabular}{llll}",
        r"    \toprule",
        r"    Model & Seed & Primary mechanism & Ren applies? \\",
        r"    \midrule",
    ]
    for model, payload in sorted(report.get("models", {}).items()):
        for seed in payload.get("seeds", []):
            seed_name = Path(seed.get("run_dir", "")).name
            primary = _tex_escape(str(seed.get("primary_mechanism", "unknown")))
            ren = "Yes" if seed.get("ren_applies") else ("Null" if seed.get("ren_null_on_minimal") else "No")
            lines.append(f"    {_tex_escape(model)} & {_tex_escape(seed_name)} & {primary} & {ren} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
