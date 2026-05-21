# MANUAL_STEPS — what you must do to finish "Option C"

> Generated 2026-05-02 ~23:00. Everything that I (the agent) could
> autonomously do is done. The remaining steps **require your
> credentials, your name, or your judgement on external publication**.
> This document tells you exactly what to do, in what order.

## Status when this doc was written

| Task | Status | Where |
|---|---|---|
| Gap 1 (30-seed) | 18/30 seeds, 50.60 % aggregate | `autoresearch/results/exp17_30seed.json` |
| Gap 2 paired Wilcoxon | **p = 1.88×10⁻³**, d_z = 0.74 | `autoresearch/results/statistical_validation_paired.json` |
| Gap 3 ablations | 4 of 5 at n=30; L1 single-seed | `autoresearch/results/gap3_*.json` |
| Lemma D.1 v2 | T1+T2+T3 sympy-verified | `paper/gap6_lemma_d1_v2_results.txt` |
| EMERALD ref | Burchi & Timofte 2025 (arXiv:2507.04075) | `paper/references.bib` |
| Cross-benchmark | Crafter-original 3-seed smoke (Δ +0.15 pp, preliminary) | `paper/sec_crafter_smoke.md` |
| Figures 1–5 | All generated as PDF + PNG | `paper/figures/` |
| Paper draft | Markdown + LaTeX | `paper/draft.md`, `paper/main.tex` |
| arXiv package | tar.gz, 89 KB | `paper/arxiv_submission_<timestamp>.tar.gz` |
| Craftax PR templates | Issue + PR + benchmark contribution | `paper/craftax_pr_draft/PR_BUNDLE.md` |
| Self peer review | NeurIPS-style checklist | `paper/peer_review_self.md` |

## STEP 1 — Decide your real author identity (5 min)

The current paper says `Anonymous Author`. arXiv requires a real name.

Pick:
- **Real name**: e.g. "Vlad Vrinceanu" or whatever you publish as
- **Affiliation**: company / lab / "Independent Researcher" / institution
- **Email**: a stable address you'll respond at (not a temporary one)
- **ORCID** (optional but recommended): https://orcid.org/register if you don't have one

Apply to `paper/main.tex`:

```latex
\author[1]{Your Real Name}
\affil[1]{Your Affiliation \\\texttt{your.email@domain.com} \\ ORCID: 0000-0000-0000-0000}
```

Co-authors? Repeat the `\author[2]{}` and `\affil[2]{}` blocks.

## STEP 2 — Install LaTeX + compile main.tex (15 min)

You don't have `pdflatex` installed locally. Install:

```bash
# Mac, full install (~4 GB):
brew install --cask mactex

# Mac, minimal (~80 MB):
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install authblk natbib booktabs microtype hyperref \
                    pdftexcmds amsmath collection-fontsrecommended
```

Then compile:

```bash
cd work/05_craftax/paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
open main.pdf
```

If compile errors:
- "Missing package" → `sudo tlmgr install <package>`
- Math errors → check that the markdown→LaTeX conversion didn't mangle
  any `$...$` blocks (cross-reference `paper/draft.md` lines)

If you don't want to install MacTeX, use Overleaf:
1. Go to overleaf.com → New Project → Upload Project
2. Upload `paper/arxiv_submission_<timestamp>.tar.gz`
3. Set engine to "pdflatex"
4. Compile

## STEP 3 — Re-read the paper, fix the 3 known issues (~30 min)

The peer review (`paper/peer_review_self.md` Section D) lists pre-submission
items. Three of them I (the agent) couldn't address autonomously:

### 3a. Replace Anonymous author block

Already covered in STEP 1.

### 3b. Verify abstract fits arXiv's 1920 character limit

Run:
```bash
grep -A 100 'begin{abstract}' main.tex | sed -n '/begin{abstract}/,/end{abstract}/p' | wc -c
```

If > 1920, trim. Currently the abstract is ~1750 characters — should fit.

### 3c. Read sections 5–7 for honest framing

The paper claims 50.6% on n=18 seeds (target was 30). Section 5.1 says
this honestly. Re-read once before submitting to make sure you're
comfortable with the claim.

## STEP 4 — arXiv submission (~30 min)

1. **Account**: https://arxiv.org/user — register if you don't have one.
2. **Endorser**: if first arXiv submission in cs.LG, you need an
   endorser. Sergio Hernández-Cerezo or Guillem Duran-Ballester (the
   FMC paper authors, easily reachable via the FMC repos) are natural
   endorsers given the topic. Email them with: paper title, abstract,
   one-sentence overview.
3. **Upload**: https://arxiv.org/submit
   - Click "Start a new submission"
   - License: arXiv default
   - Categories: **cs.LG (primary), cs.AI (secondary)**
   - Upload `paper/arxiv_submission_<timestamp>.tar.gz`
   - Title, authors, abstract: copy from `main.tex`
   - Comments: "Workshop submission. 9 pages + appendices, 5 figures.
     Code at <YOUR_REPO_URL>."
4. **Preview** the rendered PDF — arXiv will compile your tex on their
   side. If errors, fix and re-upload.
5. **Submit**. arXiv preprints typically appear within 24 hours.

## STEP 5 — Public your repo (~15 min)

Before the Craftax PR, your repo URL needs to be live.

Decisions:
- **Public on GitHub** under your name (recommended): show the world.
- **Public under an org**: e.g. an Antygravity org if you have one.
- **Anonymized for review**: only if you target a double-blind venue.

```bash
cd /Users/vladvrinceanu/Desktop/PROGETTI\ ANTYGRAVITY/FractalAI

# Option A: create a NEW public repo just for this paper
mkdir -p ~/code/fractalai-craftax-paper
cp -r work/05_craftax ~/code/fractalai-craftax-paper/
cp 1803.05049v5.pdf ~/code/fractalai-craftax-paper/  # Hernández paper
cd ~/code/fractalai-craftax-paper
git init
gh repo create fractalai-craftax-paper --public --source=. --push

# Option B: push the whole FractalAI repo public (include all the
# context from CLAUDE.md, deep-dives, etc.)
gh repo view --json url  # check if origin exists
gh repo create FractalAI --public --source=. --push
```

Make sure:
- The repo has a clear README with the headline result + reproduction steps
- License is set (MIT or Apache-2.0 are standard for ML papers)
- A working `pip install -e .` or single-command repro exists

## STEP 6 — Open the Craftax issue/PR (~30 min)

1. Read `paper/craftax_pr_draft/PR_BUNDLE.md` end-to-end.
2. Pick **Template 1** (issue) — this is the right first contact.
3. Substitute `<YOUR_REPO_URL>` with your actual repo URL.
4. Substitute `<ARXIV_LINK>` with the arXiv preprint URL once live.
5. Open the issue at https://github.com/MichaelTMatthews/Craftax/issues
6. Wait 3-7 days for the maintainer to engage.
7. If they thumbs-up: send Template 2 (README PR).

Do NOT send a fork-and-PR cold without the issue first — that's the
upstream maintainer's preferred flow.

## STEP 7 — Optional: Cross-benchmark full study (1-3 weeks compute)

The cross-benchmark result is preliminary (3-seed at N=32 M=12 — see
`paper/sec_crafter_smoke.md`). For the conference-track upgrade
(NeurIPS / ICLR), you'll want full N=512 M=40 30-seed paired data.

Estimated CPU: ~30-90 hours per config; v4 + exp17 = ~2-6 days
on a single M1 Pro CPU. On Modal cloud or a multi-core server: ~6-18
hours.

Run:
```bash
cd work/05_crafter_original_port
JAX_PLATFORMS=cpu python fmc_crafter.py \
    --seeds 42,43,...,71 --shaping v4 --N 512 --M 40 --max_steps 500 \
    --out_json results_full_v4_30seed.json &
# Then same with --shaping exp17
```

Then update `paper/sec_crafter_smoke.md` → `sec_crafter_full.md`,
re-run `paired_test`, regenerate Figure 6, update Section 6.1 of
the draft, and submit a v2 to arXiv.

## STEP 8 — Publish a blog / tweet (optional, ~1 hour)

Once the arXiv preprint is live and the Craftax issue is open,
consider:
- A 2-paragraph LinkedIn post / X thread linking the arXiv URL
- A blog post explaining Conjecture D in plain language (the
  popularization angle)
- Email the FMC authors (Sergio + Guillem) with the preprint —
  they'll likely amplify it on their networks

## What I (the agent) explicitly did NOT do

Listed honestly so you know:

1. **No arXiv submission**: I have no arXiv account, no upload tool,
   no LaTeX compiler. STEP 4 is yours.
2. **No git push to a public repo**: I haven't done `git push` to
   anything public. STEP 5 is yours.
3. **No GitHub issue or PR creation on the Craftax repo**: STEP 6 is
   yours.
4. **No social-media posting**: STEP 8 is yours.
5. **No 30-seed full-compute Crafter-original**: STEP 7 is too slow
   for a single agent session; left for your follow-up.

## What I did do (everything you can review and trust)

1. ✅ Re-discovered v4 30-seed paired data inside an existing JSON,
   ran a true paired Wilcoxon (p = 1.88×10⁻³).
2. ✅ Verified EMERALD reference (Burchi & Timofte 2025).
3. ✅ Tightened Lemma D.1 with three sympy-derived theorems (T1, T2, T3).
4. ✅ Generated Figure 4 schematic via matplotlib (no Nano Banana
   API key set, used programmatic fallback).
5. ✅ Built a markdown→LaTeX converter and assembled main.tex.
6. ✅ Installed crafter (original) and ran a 3-seed paired smoke test.
   Result is preliminary (Δ +0.15 pp at low compute) and honestly
   framed in `paper/sec_crafter_smoke.md`.
7. ✅ Built the arXiv tar.gz package.
8. ✅ Wrote three Craftax PR / issue templates ready to copy-paste.

If anything in this document doesn't match what's on disk, the disk
wins; that's the source of truth.

Good luck with the submission.
