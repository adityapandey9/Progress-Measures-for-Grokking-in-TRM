#!/usr/bin/env python3
"""Export per-neuron frequency/R2 tables from reverse-engineering artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from pluto.trm.base_mechanistic_interpretability.analysis.common import ensure_dir, save_json


def summarize_neuron_assignments(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(int(r["best_frequency"])) for r in rows)
    strong = [r for r in rows if float(r.get("best_r2", 0.0)) >= 0.5]
    return {
        "n_neurons": len(rows),
        "n_strong_neurons_r2_ge_0_5": len(strong),
        "fraction_strong": len(strong) / max(1, len(rows)),
        "frequency_counts": dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
    }


def rows_from_reverse_engineering(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text())
    rows: List[Dict[str, Any]] = []
    clusters = obj.get("mlp_neuron_clustering", {})
    for layer, payload in clusters.items():
        neurons = payload.get("neurons", [])
        if neurons:
            for item in neurons:
                rows.append(
                    {
                        "layer": layer,
                        "neuron": int(item.get("neuron", -1)),
                        "best_frequency": int(item.get("best_frequency", item.get("frequency", -1))),
                        "best_r2": float(item.get("best_r2", item.get("r2", 0.0))),
                        "basis": item.get("basis", item.get("best_basis", "unknown")),
                    }
                )
        else:
            sample = payload.get("neuron_dominant_freq_sample", {})
            for neuron_str, freq in sample.items():
                rows.append(
                    {
                        "layer": layer,
                        "neuron": int(neuron_str),
                        "best_frequency": int(freq),
                        "best_r2": 0.0,
                        "basis": "unknown",
                    }
                )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["layer", "neuron", "best_frequency", "best_r2", "basis"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = rows_from_reverse_engineering(Path(args.reverse_engineering))
    results = {"rows": rows, "summary": summarize_neuron_assignments(rows)}
    out = ensure_dir(Path(args.output_dir))
    save_json(out / "neuron_tables.json", results)
    write_csv(out / "neuron_tables.csv", rows)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reverse-engineering", required=True)
    p.add_argument("--output-dir", default="bmi_analysis/neuron_tables")
    args = p.parse_args()
    r = run(args)
    print(f"neuron rows={len(r['rows'])} strong={r['summary']['fraction_strong']:.3f}")


if __name__ == "__main__":
    main()
