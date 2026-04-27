# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Source:** [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills) — derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project briefing — FractalAI

> *Read this section before any planning, design, or coding command (`/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/design`, `/review`, `/ship`, `/qa`, `/fractal-decide`, `/octopus`, …).*

### What we are working on

This repo is a **research + tooling effort around the Fractal Monte Carlo (FMC) algorithm** of Hernández-Cerezo & Duran-Ballester. Goals:

1. **Replicate** FMC on Atari (done — 96/100 Boxing in 7 min, 231 LOC NumPy, no GPU). See [`work/03_atari_replication/`](work/03_atari_replication/).
2. **Extend** FMC into coding (the [`plugin/fractal-coding-loop/`](plugin/fractal-coding-loop/) Claude Code plugin: `/fractal-decide`, `/octopus`, `/fractal-recall`).
3. **Pick a benchmark target** for a credible FMC paper. Top candidates (see [`DominiDaIndagare.md`](DominiDaIndagare.md)): **Procgen**, **Crafter/Craftax**, **CompilerGym**.
4. **Theoretical deep-dives** on cloning math, SMC particle-filter view, Active Inference link, Fractal Memory. See [`work/02_deep_dives/`](work/02_deep_dives/).

Language convention: **Italian for prose, English for code/comments**. ISO 8601 dates. Today: 2026-04-26.

### Canonical sources (papers + FMC + Fractal Memory)

**Main papers — read in this order**:

| # | Source | Local path | What it gives you |
|---|---|---|---|
| 1 | Hernández-Cerezo & Duran-Ballester (2020), *Fractal AI: A Fragile Theory of Intelligence*, arXiv:1803.05049v5 | [`1803.05049v5.pdf`](1803.05049v5.pdf) + [`docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf`](docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf) | **Canonical FMC paper.** §2.2 = math (relativize, composite reward, virtual reward); §4 = the algorithm; §5 = Atari results |
| 2 | Hernández-Cerezo, Duran-Ballester, Baxevanakis (2018), *Solving Atari Games Using Fractals And Entropy*, arXiv:1807.01081 | [`docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf`](docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf) | **Companion empirical paper.** FMC beats MCTS UCT with <1000 vs 3M samples per action |
| 3 | Hernández, Duran, Amigó (2017), *General Algorithmic Search*, arXiv:1705.08691 | [`docs/bibliography/sources/papers/2017_general_algorithmic_search_1705.08691.pdf`](docs/bibliography/sources/papers/2017_general_algorithmic_search_1705.08691.pdf) | **Predecessor.** Swarm meta-heuristic; FMC = "GAS applied to planning" |
| 4 | Amigó, Balogh, Hernández (2018), *A Brief Review of Generalized Entropies*, Entropy 20(11):813 | [`docs/bibliography/sources/papers/2018_brief_review_generalized_entropies.pdf`](docs/bibliography/sources/papers/2018_brief_review_generalized_entropies.pdf) | **Theoretical foundation** for non-additive composite rewards |

**Full corpus index (all publications, drafts, blog, codebases, gaps)**: [`docs/bibliography/CORPUS.md`](docs/bibliography/CORPUS.md)

**FMC documentation (in-repo, what we wrote)**:

- High-level theory & "why this transfers": [`plugin/fractal-coding-loop/docs/THEORY.md`](plugin/fractal-coding-loop/docs/THEORY.md)
- Step-by-step algorithm walkthrough: [`plugin/fractal-coding-loop/docs/ALGORITHM.md`](plugin/fractal-coding-loop/docs/ALGORITHM.md)
- Component/file reference: [`plugin/fractal-coding-loop/docs/COMPONENTS.md`](plugin/fractal-coding-loop/docs/COMPONENTS.md)
- Invocation & config: [`plugin/fractal-coding-loop/docs/USAGE.md`](plugin/fractal-coding-loop/docs/USAGE.md)
- Cloning math (formal): [`work/02_deep_dives/01_cloning_mathematics.md`](work/02_deep_dives/01_cloning_mathematics.md)
- SMC / particle filter view: [`work/02_deep_dives/05_smc_particle_filter_view.md`](work/02_deep_dives/05_smc_particle_filter_view.md)
- Active Inference link: [`work/02_deep_dives/02_active_inference_link.md`](work/02_deep_dives/02_active_inference_link.md)
- Standard Model of Cognition: [`work/02_deep_dives/03_standard_model_cognition.md`](work/02_deep_dives/03_standard_model_cognition.md)

**Fractal Memory documentation**:

- Sergio's source (Slide deck, 2020): [`2020 Fractal Slide.md`](2020%20Fractal%20Slide.md) + archived copy [`docs/bibliography/sources/books/2020_fractal_memory_slides.md`](docs/bibliography/sources/books/2020_fractal_memory_slides.md)
- Operational spec (Hives + Badger): [`2020 Fractal.md`](2020%20Fractal.md) + [`docs/bibliography/sources/books/2020_hives_badger_meets_fractal_ai.md`](docs/bibliography/sources/books/2020_hives_badger_meets_fractal_ai.md)
- Book #2 (AGI Structure): [`Fractal Book.md`](Fractal%20Book.md) + [`docs/bibliography/sources/books/2020_book2_agi_structure.md`](docs/bibliography/sources/books/2020_book2_agi_structure.md)
- Our deep dive synthesizing all three: [`work/02_deep_dives/06_book2_badger_fractal_memory.md`](work/02_deep_dives/06_book2_badger_fractal_memory.md)
- Implementation (Wigner-weighted memory bank): [`plugin/fractal-coding-loop/scripts/fractal_memory.py`](plugin/fractal-coding-loop/scripts/fractal_memory.py) — recall by `R(x) = (π/2) x exp(-π/4 x²)`, debiased by visit count

**Working analyses (Italian, long-form)**:

- [`ANALISIS.md`](ANALISIS.md) — main paper analysis (46 KB)
- [`analisisPost.md`](analisisPost.md) / [`analisisPost2.md`](analisisPost2.md) — post-replication reflections
- [`DominiDaIndagare.md`](DominiDaIndagare.md) — domain survey for benchmark selection

**Sergio's oral knowledge (first-person, ~2.5h)**:

- [`docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md`](docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md) — Radient 2026 podcast, full transcript (~21 700 words), structured into 21 argumentative chapters with Italian theses. Spanish dialogue preserved verbatim. **Most direct source for Sergio's intuitive framing**: Wissner-Gross genesis, the one-night cochecito, the **6-fold optimal branching factor**, "frontera caos/orden" as candidate Third Law of cognition, FMC vs MCTS quantitative comparison (Sergio claims ~150,000 vs ~35 samples — verify against paper's "400"), bengala-vs-laser metaphor for LLM limits, open-source-as-entropy-maximization philosophy. **Read alongside paper #1** for the operational intuition that the math alone doesn't carry.

**Codebases** under [`repos/`](repos/): `FractalAI_old` (deprecated NumPy, paper #1 reference), `fragile` (PyTorch/GPU, active), `fragile-rl` (Fragile Mechanics, 2024-2026, successor to Book #2).

### gstack

Installed at `~/.claude/skills/gstack` (40+ skills + Playwright Chromium). Slash commands become available at the next session start. Most useful here:

- `/office-hours` — interrogate scope before coding (good for the Procgen/Crafter/CompilerGym decision)
- `/plan-ceo-review` — strategic review of an approach
- `/plan-eng-review` — architecture lock-in
- `/design` — design doc / mockup
- `/review` — code review on the current branch
- `/qa <url>` — browser-based QA via Playwright (useful for the JS sims under [`simulations/`](simulations/))
- `/ship` — test + open PR
- `/cso` — security audit (OWASP/STRIDE)
- `/learn` — Sergio's published learnings + project memory
- `/gstack-upgrade` — keep gstack current

Coexists with the in-house plugin (`/fractal-decide`, `/octopus`, `/fractal-recall`, `/fractal-memory-show`) and the nWave wave-methodology commands (`/nw-*`). No conflicts: namespaces are distinct.
