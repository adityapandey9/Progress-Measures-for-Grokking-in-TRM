# Progress Measures for Grokking in Minimal Recursive Transformers

Reproducibility bundle for *Progress Measures for Grokking in Minimal Recursive Transformers*. This repository contains trained checkpoints, precomputed results, paper figures, and the full Python source needed to reproduce all experiments and regenerate figures from scratch. The paper studies grokking — delayed generalisation after near-perfect memorisation — in a Tiny Recursive Reasoning Model (TRM) and a one-layer transformer baseline, identifying Fourier-basis progress measures and a latent-grokking regime where the model continues to reorganise its internal representations after the test-loss plateau.

## Paper

[`paper/progress_measures_grokking_trm.pdf`](paper/progress_measures_grokking_trm.pdf)

## Contents

```
src/                     Python source (install with PYTHONPATH=src)
  pluto/trm/
    base_mechanistic_interpretability/   main BMI package
      config.py                          experiment config dataclasses
      train_grokking.py                  training entry point
      train_nanda_baseline.py            Nanda baseline training
      models/                            TRM and Nanda model definitions
      dataset/                           modular-arithmetic dataset
      fourier/                           Fourier progress measures
      analysis/                          figure/metric generation scripts
    models/
      losses.py                          ACT loss head (cross-package dep)
      layers.py                          attention/embedding layers
checkpoints/
  nanda_1layer/seed_{0..4}/   Nanda one-layer calibration checkpoints
  trm_minimal/seed_{0..4}/    TRM minimal hero runs (nanda_faithful_50k preset)
results/
  figures/                 precomputed paper figures (PDFs)
  metrics/
    paper_v2_metrics.json  aggregated metrics across all seeds and sweeps
paper/
  progress_measures_grokking_trm.pdf   compiled paper PDF
scripts/
  reproduce_figures.sh     verifies precomputed figures are present
  run_analysis.sh          regenerates figures from precomputed metrics
notebooks/
  nanda_figure_walkthrough.ipynb   Colab-compatible step-by-step walkthrough
configs/                 YAML experiment configs
requirements.txt         Python dependencies
```

## Reproduce

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

### 1. Verify precomputed figures are present

```bash
bash scripts/reproduce_figures.sh
```

Expected output: `All core figures present.`

### 2. Regenerate the summary figures from precomputed metrics

```bash
bash scripts/run_analysis.sh
```

This regenerates the metrics-driven summary figures (algorithm schematic, multi-seed
summary, weight-decay and data-fraction sweeps) from
`results/metrics/paper_v2_metrics.json` (no GPU required). The reverse-engineering,
progress-measure, and latent figures are shipped precomputed under `results/figures/`;
regenerating those requires running the analysis modules in
`src/pluto/trm/base_mechanistic_interpretability/analysis/` on the checkpoints.

### 3. Retrain from scratch (requires GPU + ~6 h per seed)

```bash
export PYTHONPATH=src
python -m pluto.trm.base_mechanistic_interpretability.train_grokking \
  --preset nanda_faithful_50k \
  --output-dir my_runs/seed_0 \
  --seed 0
```

For multiple seeds:

```bash
python -m pluto.trm.base_mechanistic_interpretability.train_multi_seed \
  --preset nanda_faithful_50k \
  --output-dir my_runs
```

## Protocol note

All hero models use the Nanda-faithful float64 cross-entropy loss
(`cross_entropy_high_precision`), applied to the language-modelling objective.
This prevents float32 precision artefacts from masking the grokking transition.

## Citation

```bibtex
@article{pandey2026progress,
  title   = {Progress Measures for Grokking in Minimal Recursive Transformers},
  author  = {Pandey, Aditya Kumar and Pandey, Anuj Kumar},
  year    = {2026},
  url     = {https://github.com/adityapandey9/Progress-Measures-for-Grokking-in-TRM}
}
```
