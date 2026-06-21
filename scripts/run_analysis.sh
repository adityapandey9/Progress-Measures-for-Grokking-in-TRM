#!/usr/bin/env bash
# run_analysis.sh — regenerate the metrics-driven summary figures from the
# precomputed aggregated metrics shipped in this repo.
#
# What this regenerates (from results/metrics/paper_v2_metrics.json, no GPU):
#   fig_algorithm_schematic.pdf, fig_multiseed_summary.pdf,
#   fig_weight_decay_sweep.pdf, fig_data_fraction_sweep.pdf
#
# The reverse-engineering, progress-measure, calibration, and latent figures are
# derived from per-checkpoint mechanistic analysis. They are shipped precomputed
# under results/figures/; regenerating them requires running the analysis modules
# in src/.../analysis/ on the checkpoints under checkpoints/ (GPU recommended).
#
# Usage:
#   pip install -r requirements.txt
#   bash scripts/run_analysis.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

METRICS="${ROOT}/results/metrics/paper_v2_metrics.json"
FIGURES="${ROOT}/results/figures"
mkdir -p "${FIGURES}"

if [[ ! -f "${METRICS}" ]]; then
  echo "ERROR: ${METRICS} not found. Clone the full repo and retry." >&2
  exit 1
fi

echo "==> Repo root:        ${ROOT}"
echo "==> Metrics:          ${METRICS}"
echo "==> Figures output:   ${FIGURES}"
echo "==> Regenerating metrics-driven summary figures..."

python3 - "${METRICS}" "${FIGURES}" <<'PY'
import json
import sys
from pathlib import Path

from pluto.trm.base_mechanistic_interpretability.analysis import figstyle
from pluto.trm.base_mechanistic_interpretability.analysis import plot_v2_figures as p

metrics = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])

figstyle.apply_style()
p.plot_v2_algorithm_schematic(out)
p.plot_v2_multiseed(metrics, out)
p.plot_v2_weight_decay(metrics, out)
p.plot_v2_data_fraction(metrics, out)
print("  regenerated: fig_algorithm_schematic, fig_multiseed_summary, "
      "fig_weight_decay_sweep, fig_data_fraction_sweep")
PY

echo ""
echo "==> Done. The mechanistic figures (Fourier, progress, ablation, latent) are"
echo "    shipped precomputed under results/figures/ and require running the"
echo "    analysis modules on the checkpoints to regenerate."
ls "${FIGURES}/"
