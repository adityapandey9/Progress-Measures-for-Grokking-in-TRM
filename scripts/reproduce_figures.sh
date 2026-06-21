#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIG="${ROOT}/results/figures"
mkdir -p "${FIG}"
echo "Repro bundle ships precomputed paper figures under results/figures/"
echo "Core figures (Nanda Fig 2/6/7 analogues):"
for f in fig_grokking_curves.pdf fig_frequency_ablation_grid.pdf fig_progress_measures.pdf; do
  if [[ -f "${FIG}/${f}" ]]; then
    echo "  OK ${f}"
  else
    echo "  MISSING ${f}"
    exit 1
  fi
done
echo "All core figures present."
