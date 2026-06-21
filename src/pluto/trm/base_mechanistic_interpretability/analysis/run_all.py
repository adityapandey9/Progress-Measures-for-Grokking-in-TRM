#!/usr/bin/env python3
"""Run full BMI analysis pipeline (Nanda + HRM probes)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-dir", default="bmi_grokking_runs/default")
    p.add_argument("--output-dir", default="bmi_analysis")
    p.add_argument("--training-history", default="bmi_grokking_runs/default/training_history.json")
    p.add_argument("--skip-trajectory", action="store_true")
    p.add_argument("--model-type", default="trm_full", choices=["nanda", "trm_minimal", "trm_full"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    cpu = ["--cpu"] if args.cpu else []
    out = Path(args.output_dir)

    mtype = ["--model-type", args.model_type]
    skip_traj = args.skip_trajectory or args.model_type == "nanda"
    cmds = [
        [
            py,
            str(root / "progress_measures_grokking.py"),
            "--checkpoint",
            args.checkpoint,
            "--output-dir",
            str(out / "progress"),
            "--training-history",
            args.training_history,
        ]
        + mtype
        + cpu,
    ]
    if args.model_type != "nanda":
        cmds.extend(
            [
                [
                    py,
                    str(root / "mechanistic_circuit.py"),
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    str(out / "mechanistic"),
                ]
                + mtype
                + cpu,
                [
                    py,
                    str(root / "frequency_ablations.py"),
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    str(out / "ablations"),
                ]
                + mtype
                + cpu,
                [
                    py,
                    str(root / "all_frequency_ablations.py"),
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    str(out / "all_frequency_ablations"),
                    "--model-type",
                    args.model_type,
                ]
                + cpu,
                [
                    py,
                    str(root / "reverse_engineering.py"),
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    str(out / "reverse_engineering"),
                ]
                + mtype
                + cpu,
                [
                    py,
                    str(root / "latent_reasoning_probes.py"),
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    str(out / "reasoning"),
                ]
                + mtype
                + cpu,
                [
                    py,
                    str(root / "neuron_tables.py"),
                    "--reverse-engineering",
                    str(out / "reverse_engineering" / "reverse_engineering.json"),
                    "--output-dir",
                    str(out / "neuron_tables"),
                ],
            ]
        )
    if not skip_traj:
        cmds.insert(
            1,
            [
                py,
                str(root / "progress_trajectory.py"),
                "--run-dir",
                args.run_dir,
                "--output-dir",
                str(out / "trajectory"),
            ]
            + mtype
            + cpu,
        )
        cmds.append(
            [
                py,
                str(root / "weight_norms.py"),
                "--run-dir",
                args.run_dir,
                "--output-dir",
                str(out / "weight_norms"),
            ]
        )

    cmds.append(
        [
            py,
            str(root / "plot_figures.py"),
            "--analysis-dir",
            str(out),
            "--figures-dir",
            str(out / "figures"),
            "--training-history",
            args.training_history,
        ]
    )

    for cmd in cmds:
        print(">>", " ".join(cmd))
        subprocess.check_call(cmd)
    print("BMI analysis complete.")


if __name__ == "__main__":
    main()
