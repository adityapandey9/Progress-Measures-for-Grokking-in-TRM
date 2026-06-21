# How to publish on arXiv

**Paper:** Progress Measures for Latent Grokking Mechanistic Interpretability

Follow the structure of [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) and [arXiv:2601.10679](https://arxiv.org/abs/2601.10679).

## Step 1 — Reproduce figures (already done on remote GPU)

```bash
# Train 20k steps (CUDA)
python pluto/trm/base_mechanistic_interpretability/train_grokking.py \
  --output-dir bmi_grokking_runs/default --max-steps 20000

# Analysis + figures
python pluto/trm/base_mechanistic_interpretability/analysis/run_all.py \
  --checkpoint bmi_grokking_runs/default/checkpoint_final.pt \
  --training-history bmi_grokking_runs/default/training_history.json

# Copy figures into paper/
cp bmi_analysis/default/figures/*.pdf \
   pluto/trm/base_mechanistic_interpretability/paper/figures/
```

Remote one-liner: `bash pluto/trm/base_mechanistic_interpretability/remote/run_full_training.sh`

## Step 2 — Compile PDF

```bash
cd pluto/trm/base_mechanistic_interpretability/paper
pdflatex latent_grokking_mechanistic_interp.tex
bibtex latent_grokking_mechanistic_interp
pdflatex latent_grokking_mechanistic_interp.tex
pdflatex latent_grokking_mechanistic_interp.tex
```

## Step 3 — arXiv account

Register at [https://arxiv.org/user/register](https://arxiv.org/user/register). cs.LG requires an endorser if you are a first-time submitter.

## Step 4 — New submission

1. Login → **Submit** → **Start New Submission**
2. Primary: **cs.LG**; secondary: **cs.AI** (optional)
3. Metadata:
   - **Title:** Progress Measures for Latent Grokking Mechanistic Interpretability
   - **Abstract:** paste from `latent_grokking_mechanistic_interp.tex`

## Step 5 — Upload

**Option A — PDF only:** upload `latent_grokking_mechanistic_interp.pdf`

**Option B — LaTeX source (recommended):**
```bash
tar czf bmi_arxiv.tar.gz \
  latent_grokking_mechanistic_interp.tex references.bib 00README.txt figures/*.pdf
```

## Step 6 — License and comments

- License: arXiv.org perpetual non-exclusive license
- Comments: `Code: pluto/trm/base_mechanistic_interpretability`

## Step 7 — Preview and submit

Verify equations, figures, and bibliography render correctly.

## Step 8 — Post-submission

Add arXiv ID to `README.md`; optionally register on Google Scholar after announcement.

## Checklist

- [x] 20k-step training complete (test acc = 100%)
- [x] Four PDF figures generated from real run data
- [x] Paper written with empirical numbers
- [ ] Compile final PDF locally (requires pdflatex + bibtex)
- [ ] Upload to arXiv
