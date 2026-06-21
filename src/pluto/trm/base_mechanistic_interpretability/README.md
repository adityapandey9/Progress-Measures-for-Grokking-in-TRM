# Base Mechanistic Interpretability (BMI)

Display name: **base-mechanistic-interpretability**

Stripped **base TRM** variant for studying **latent grokking** on modular addition (arXiv:2301.05217), merged with recursive **reasoning vs guessing** probes (arXiv:2601.10679). No Sudoku, no ARC-AGI.

Python package: `pluto.trm.base_mechanistic_interpretability`

## Dataset

Modular addition mod `P=113`, input format `a b =`, predict `(a+b) mod P` at `=`. **30% train / 70% test** split (seed 0), matching the Progress Measures paper mainline experiment.

```bash
python -c "
from pluto.trm.base_mechanistic_interpretability.dataset import save_dataset_artifacts
from pluto.trm.base_mechanistic_interpretability.config import ModAddGrokkingConfig
save_dataset_artifacts(ModAddGrokkingConfig(), 'bmi_grokking_runs/default/data')
"
```

## Train (~20k steps, grokking)

```bash
python pluto/trm/base_mechanistic_interpretability/train_grokking.py \
  --output-dir bmi_grokking_runs/default \
  --max-steps 20000 \
  --log-every 100
```

Hyperparameters: AdamW `lr=1e-3`, `weight_decay=1.0`, full-batch train pairs (Nanda et al. §3).

## Analysis

```bash
python pluto/trm/base_mechanistic_interpretability/analysis/run_all.py \
  --checkpoint bmi_grokking_runs/default/checkpoint_final.pt \
  --training-history bmi_grokking_runs/default/training_history.json
```

Scripts:

| Script | Paper | Metrics |
|--------|-------|---------|
| `progress_measures_grokking.py` | arXiv:2301.05217 | train/test loss & acc, embedding Fourier norms, key frequencies, restricted/excluded loss, 3 training phases |
| `latent_reasoning_probes.py` | arXiv:2601.10679 | ACT-step mean-field CE, fixed-point violation, 4 reasoning modes, latent PCA trajectory |
| `plot_figures.py` | both | PDF figures for arXiv |

## Remote CUDA

```bash
bash pluto/trm/base_mechanistic_interpretability/remote/run_full_training.sh
```

## Paper

`paper/latent_grokking_mechanistic_interp.tex` — **Progress measures for latent grokking mechanistic interpretability**

See `MAPPING_TRM.md` for 1:1 mapping vs [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
