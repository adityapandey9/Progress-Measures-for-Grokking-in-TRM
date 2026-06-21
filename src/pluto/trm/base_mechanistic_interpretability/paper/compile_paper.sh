#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:${PATH}"

DOC=latent_grokking_mechanistic_interp

# Deterministic sequence so bibtex always runs and the .bbl is regenerated.
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex"
bibtex "$DOC"
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex"
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex"

echo "=== build checks ==="
if grep -q "Citation .* undefined" "$DOC.log"; then
  echo "FAIL: undefined citations remain"; grep "Citation .* undefined" "$DOC.log" | head; exit 1
fi
if [ ! -f "$DOC.bbl" ]; then echo "FAIL: no .bbl produced"; exit 1; fi
if command -v pdfinfo >/dev/null 2>&1; then pdfinfo "$DOC.pdf" | grep '^Pages:'; fi
echo "OK: build clean, bibliography present"
