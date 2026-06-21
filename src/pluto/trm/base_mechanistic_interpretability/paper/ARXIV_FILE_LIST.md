# arXiv upload file list — Progress Measures for Latent Grokking Mechanistic Interpretability

Based on completed 20k-step run (RTX 3060, test acc = 100%).

## Required for PDF-only submission

| File | Description |
|------|-------------|
| `latent_grokking_mechanistic_interp.pdf` | Compiled paper |

## LaTeX source bundle (`bmi_arxiv.tar.gz`)

| File | Required |
|------|----------|
| `latent_grokking_mechanistic_interp.tex` | Yes |
| `references.bib` | Yes |
| `00README.txt` | Yes |
| `figures/fig_grokking_curves.pdf` | Yes |
| `figures/fig_progress_measures.pdf` | Yes |
| `figures/fig_latent_reasoning_modes.pdf` | Yes |
| `figures/fig_latent_pca_trajectory.pdf` | Yes |

## Experimental artifacts (supplementary / reproducibility)

| Path | Description |
|------|-------------|
| `../train_grokking.py` | Training script |
| `../analysis/run_all.py` | Full analysis pipeline |
| `../analysis/progress_measures_grokking.py` | arXiv:2301.05217 metrics |
| `../analysis/latent_reasoning_probes.py` | arXiv:2601.10679 metrics |
| `../analysis/plot_figures.py` | Figure generation |
| `../dataset/mod_add.py` | Modular addition dataset (P=113, 30% train) |
| `../MAPPING_TRM.md` | 1:1 mapping vs TinyRecursiveModels |
| `.bmi-remote-results/full/training_history.json` | 201 logged checkpoints |
| `.bmi-remote-results/full/progress/progress_measures_grokking.json` | Fourier + phase metrics |
| `.bmi-remote-results/full/reasoning/latent_reasoning_probes.json` | ACT probe metrics |
| `.bmi-remote-results/full/checkpoint_final.pt` | Final model weights |

## Key results (for abstract/metadata)

- Train acc = 100% by step 600; test acc < 1% until step ~5300
- Final test acc = 100%, test loss = 2.9e-5
- Key Fourier frequencies: {22, 44, 45, 34, 0}
- Fixed-point violation rate: 78%
- Reasoning modes (post-grokking): 100% trivial success

## Create tarball

```bash
cd pluto/trm/base_mechanistic_interpretability/paper
tar czf bmi_arxiv.tar.gz \
  latent_grokking_mechanistic_interp.tex \
  references.bib \
  00README.txt \
  figures/*.pdf
```
