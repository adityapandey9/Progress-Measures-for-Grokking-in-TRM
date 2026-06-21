#!/usr/bin/env python3
"""Emit paper/metrics.tex from analysis JSON (keeps paper numbers in sync)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", default="bmi_analysis/full50k")
    p.add_argument("--output", default="pluto/trm/base_mechanistic_interpretability/paper/metrics.tex")
    args = p.parse_args()
    ad = Path(args.analysis_dir)
    pm = json.loads((ad / "progress/progress_measures_grokking.json").read_text())
    re_ = json.loads((ad / "reverse_engineering/reverse_engineering.json").read_text())
    ms_path = ad / "multi_seed/multi_seed_aggregate.json"
    ms = json.loads(ms_path.read_text()) if ms_path.exists() else {"n_seeds": 0}

    kf = pm["key_frequencies"]
    wl_kf = re_["W_L_analysis"]["W_L_final"]["key_frequencies"]
    tex = f"""% Auto-generated — do not edit by hand
\\newcommand{{\\BMIKeyFreqs}}{{{','.join(map(str, kf))}}}
\\newcommand{{\\BMIWLKeyFreqs}}{{{','.join(map(str, wl_kf))}}}
\\newcommand{{\\BMITestAcc}}{{{pm['test_accuracy']:.4f}}}
\\newcommand{{\\BMIFullTestCE}}{{{pm['test_loss']:.2e}}}
\\newcommand{{\\BMITrigTestCE}}{{{pm['trig_loss_test']:.4f}}}
\\newcommand{{\\BMIExcludedTestCE}}{{{pm['excluded_loss_test']:.4f}}}
\\newcommand{{\\BMIFVEMean}}{{{pm['logit_trig_fve']['fve_mean']:.3f}}}
\\newcommand{{\\BMIEmGini}}{{{pm['embedding_gini']:.3f}}}
\\newcommand{{\\BMINSeeds}}{{{ms['n_seeds']}}}
"""
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
