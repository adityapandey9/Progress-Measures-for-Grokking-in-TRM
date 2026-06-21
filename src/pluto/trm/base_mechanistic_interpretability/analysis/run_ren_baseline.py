#!/usr/bin/env python3
"""Phase 0: Ren baseline on existing TRM checkpoints before any 50k training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json
from pluto.trm.base_mechanistic_interpretability.analysis.post_grokking_diagnosis import run_results_root


TRM_MODELS = {
    "trm_minimal": "trm_minimal",
    "trm_full_b": "trm_full",
    "trm_full_a": "trm_full",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="bmi_hybrid")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-suite", action="store_true", help="Only aggregate existing ren JSON")
    args = p.parse_args()
    root = Path(args.results_root)
    py = sys.executable
    suite_script = Path(__file__).resolve().parent / "hrm" / "ren_checkpoint_suite.py"

    if not args.skip_suite:
        for model_dir_name, model_type in TRM_MODELS.items():
            model_dir = root / model_dir_name
            if not model_dir.is_dir():
                continue
            for seed_dir in sorted(model_dir.glob("seed_*")):
                if not (seed_dir / "checkpoint_final.pt").exists():
                    print(f"skip {seed_dir} (no checkpoint)")
                    continue
                cmd = [
                    py,
                    str(suite_script),
                    "--run-dir",
                    str(seed_dir),
                    "--model-type",
                    model_type,
                ]
                if args.cpu:
                    cmd.append("--cpu")
                print(">>", " ".join(cmd))
                subprocess.check_call(cmd)

    report = run_results_root(root)
    out = ensure_dir(root / "aggregate")
    save_json(out / "ren_diagnosis_report.json", report)
    table_script = Path(__file__).resolve().parent / "write_ren_paper_table.py"
    if table_script.exists():
        subprocess.run(
            [py, str(table_script), "--report", str(out / "ren_diagnosis_report.json")],
            check=False,
        )

    # Gate: require all TRM seeds to have ren suite before 50k TRM training
    missing = []
    for model_dir_name in TRM_MODELS:
        for seed_dir in sorted((root / model_dir_name).glob("seed_*")):
            ren_json = seed_dir / "analysis" / "ren" / "ren_checkpoint_suite.json"
            if not ren_json.exists():
                missing.append(str(seed_dir))
    if missing:
        print("REN_GATE: INCOMPLETE — missing suites:", len(missing))
        for m in missing[:10]:
            print(" ", m)
        sys.exit(2)
    print("REN_GATE: PASS — all TRM seeds have Ren checkpoint suites")
    print("report ->", out / "ren_diagnosis_report.json")


if __name__ == "__main__":
    main()
