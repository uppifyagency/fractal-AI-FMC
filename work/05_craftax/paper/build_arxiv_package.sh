#!/usr/bin/env bash
# build_arxiv_package.sh — package paper/ as arxiv-ready tar.gz.
#
# Output: paper/arxiv_submission_TIMESTAMP.tar.gz containing:
#   main.tex
#   references.bib
#   figures/*.pdf
#   README_arxiv.md (instructions for user)
#
# arXiv tar.gz must contain only the source files needed to compile
# the PDF. Subdirectories are flattened to root (with figures/ kept
# as relative path).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

STAMP=$(date +"%Y%m%d_%H%M%S")
PKG_DIR="$HERE/arxiv_pkg_$STAMP"
mkdir -p "$PKG_DIR/figures"

# Copy main artifacts
cp main.tex "$PKG_DIR/"
cp references.bib "$PKG_DIR/"
cp figures/fig1_trajectory.pdf "$PKG_DIR/figures/"
cp figures/fig2_ablation.pdf "$PKG_DIR/figures/"
cp figures/fig2bis_leave_one_out.pdf "$PKG_DIR/figures/"
cp figures/fig3_pareto.pdf "$PKG_DIR/figures/"
cp figures/fig4_schematic.pdf "$PKG_DIR/figures/"
cp figures/fig5_blockers.pdf "$PKG_DIR/figures/"

# Generate README for the arxiv submitter
cat > "$PKG_DIR/README_arxiv.md" <<'EOF'
# arXiv submission package — Chain-tier Compounding Amplification

## What this is

This tarball is the LaTeX source for a workshop-track paper on
zero-training Crafter / Craftax-Classic FMC results. Built from the
markdown master in `paper/draft.md` via `paper/md2tex.py` +
`paper/build_main_tex.py`.

## Files

- `main.tex` — the only LaTeX file you compile
- `references.bib` — bibliography (~25 entries, all verified)
- `figures/*.pdf` — 6 publication figures (vector PDF, 300 DPI)

## Compile

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

If pdflatex is not installed: install MacTeX (~4 GB) or BasicTeX (~80 MB):

```bash
brew install --cask mactex            # full
# or
brew install --cask basictex          # minimal; you'll need to install
                                       # extra packages on first compile
```

Or compile in the cloud:
- Overleaf: upload all files, choose "pdflatex" engine
- arXiv's own preview compiles automatically before submission

## Before submitting to arXiv

1. **Replace the author block** at the top of `main.tex`:
   - Currently: `\author[1]{Anonymous Author}` and
                `\affil[1]{Anonymous Affiliation \\\texttt{anonymous@example.org}}`
   - Replace with your real name, affiliation, ORCID, email.

2. **Verify abstract is at most 1920 characters** (arXiv limit).

3. **Submit at https://arxiv.org/submit**:
   - Category: cs.LG (primary), cs.AI (secondary)
   - License: arXiv default (non-exclusive distribution)
   - Comments field: "Workshop submission, 9 pages + 4 appendices,
     5 figures."

4. **Endorser**: if first arXiv submission, you need an endorser in
   cs.LG. Sergio Hernández-Cerezo or any prior arXiv-published author
   in this area can endorse.

5. **Cross-listing**: consider adding stat.ML or cs.NE depending on
   reviewer audience.

## Limitations to flag in the comments field

The peer review (`paper/peer_review_self.md`) lists open weaknesses:
- 18/30 seed completion (target 30, hit budget cap)
- L1 ablation single-seed (others n=30)
- Cross-benchmark Crafter-original is preliminary (3-seed exp17 vs
  3-seed v4 smoke; full study is companion work)
- Figure 4 is a programmatic schematic, not Nano-Banana-rendered

These are honestly stated in the paper itself. Address them as the
work progresses if you upgrade to a conference-track v2.

## Data + code release

The full repository (including the FMC implementation, autoresearch
loop, ablation runner, statistical tests) is at:
- (your repo URL here)

Specifically:
- `work/05_craftax/autoresearch/fmc_mutable.py` (12 KB single-file FMC)
- `work/05_craftax/autoresearch/results.tsv` (24-experiment log)
- `work/05_craftax/autoresearch/results/*.json` (per-experiment raw_runs)
- `work/05_crafter_original_port/fmc_crafter.py` (Crafter port)
EOF

# Make the tar.gz (arxiv prefers tar.gz, no leading directory)
cd "$PKG_DIR"
tar czf "$HERE/arxiv_submission_$STAMP.tar.gz" *.tex *.bib figures/ README_arxiv.md
cd "$HERE"

# Cleanup workdir
rm -rf "$PKG_DIR"

echo "=== arXiv package ready ==="
ls -la "$HERE/arxiv_submission_$STAMP.tar.gz"
echo ""
echo "Contents:"
tar tzf "$HERE/arxiv_submission_$STAMP.tar.gz" | head -20
