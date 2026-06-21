#!/usr/bin/env python3
"""Run all HRM probes for a TRM checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/hrm")
    p.add_argument("--model-type", default="trm_full", choices=["trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    cpu = ["--cpu"] if args.cpu else []
    out = Path(args.output_dir)

    cmds = [
        [
            py,
            str(root / "hrm_reasoning_guessing.py"),
            "--checkpoint",
            args.checkpoint,
            "--output-dir",
            str(out),
            "--model-type",
            args.model_type,
        ]
        + cpu,
        [
            py,
            str(root / "act_depth_ablation.py"),
            "--checkpoint",
            args.checkpoint,
            "--output-dir",
            str(out),
            "--model-type",
            args.model_type,
        ]
        + cpu,
    ]
    for cmd in cmds:
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)
    print("HRM analysis complete.")


if __name__ == "__main__":
    main()
