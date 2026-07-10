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
3. **Chain-tier reward shaping doubles FMC on Craftax-Classic** (2026-05-01; claim corretto 2026-07-10). **Claim difendibile**: exp17 vs baseline v4 = **+22.1pp appaiato** su 18 seed (Wilcoxon $p=1.9\times10^{-3}$, Cohen $d_z=0.74$), sblocca la catena iron→diamond in zero-training. ⚠️ **Il vecchio titolo "50.95% = matches human-expert (50.5%)" è RITRATTATO** (audit night_2026-07-09 / W3-3): 50.95% è l'aggregato cross-episodio (run n=11; ri-validato n=18 = 50.60%), **non** la media per-episodio (30%, CI95 [36.85, 59.46]); e **non è like-for-like** (exp17 su Craftax-Classic-Symbolic vs umano su Crafter-original a pixel — sull'ambiente umano FMC fa **3.77%**). 23-experiment autoresearch session, meccanismo chain-tier compounding (Cong. D). See [`work/14_night_2026-07-09/wave3_validation/W33_restatement_onesto.md`](work/14_night_2026-07-09/wave3_validation/W33_restatement_onesto.md) + [`work/05_craftax/autoresearch/HANDOFF.md`](work/05_craftax/autoresearch/HANDOFF.md). Per "human-expert" servono like-for-like su Crafter-original a compute pieno + replica Procgen. Procgen/CompilerGym restano candidati.
4. **Theoretical deep-dives** on cloning math, SMC particle-filter view, Active Inference link, Fractal Memory. See [`work/02_deep_dives/`](work/02_deep_dives/).

Language convention: **Italian for prose, English for code/comments**. ISO 8601 dates. Today: 2026-05-01.

### Canonical sources (papers + FMC + Fractal Memory)

**Main papers — read in this order**:

| # | Source | Local path | What it gives you |
|---|---|---|---|
| 1 | Hernández-Cerezo & Duran-Ballester (2020), *Fractal AI: A Fragile Theory of Intelligence*, arXiv:1803.05049v5 | [`1803.05049v5.pdf`](1803.05049v5.pdf) + [`docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf`](docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf) | **Canonical FMC paper.** §2.2 = math (relativize, composite reward, virtual reward); §4 = the algorithm; §5 = Atari results |
| 2 | Hernández-Cerezo, Duran-Ballester, Baxevanakis (2018), *Solving Atari Games Using Fractals And Entropy*, arXiv:1807.01081 | [`docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf`](docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf) | **Companion empirical paper.** FMC beats MCTS UCT with <1000 vs 3M samples per action |
| 3 | Hernández, Duran, Amigó (2017), *General Algorithmic Search*, arXiv:1705.08691 | [`docs/bibliography/sources/papers/2017_general_algorithmic_search_1705.08691.pdf`](docs/bibliography/sources/papers/2017_general_algorithmic_search_1705.08691.pdf) | **Predecessor.** Swarm meta-heuristic; FMC = "GAS applied to planning" |
| 4 | Amigó, Balogh, Hernández (2018), *A Brief Review of Generalized Entropies*, Entropy 20(11):813 | [`docs/bibliography/sources/papers/2018_brief_review_generalized_entropies.pdf`](docs/bibliography/sources/papers/2018_brief_review_generalized_entropies.pdf) | **Theoretical foundation** for non-additive composite rewards |
| 5 | Wissner-Gross & Freer (2013), *Causal Entropic Forces*, Phys. Rev. Lett. 110:168702 | [`docs/bibliography/sources/papers/2013_wissner_gross_causal_entropic_forces.pdf`](docs/bibliography/sources/papers/2013_wissner_gross_causal_entropic_forces.pdf) + [`supplemental`](docs/bibliography/sources/papers/2013_wissner_gross_causal_entropic_forces_supplemental.pdf) | **Antecedente fisico canonico** citato da Sergio. Eq. 4: $F = T_c \nabla_X S_c$. Eq. 11 = limite continuo di FMC con α=0 |

**Mathematical canon** (single source of truth per formule, definizioni, teoremi): [`docs/MATH_CANON.md`](docs/MATH_CANON.md)

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
- Relativize axiomatics: [`work/02_deep_dives/04_relativize_axiomatics.md`](work/02_deep_dives/04_relativize_axiomatics.md)
- Wright-Fisher mapping (falsifica empirica del "magic 6"): [`work/02_deep_dives/07_wright_fisher_mapping.md`](work/02_deep_dives/07_wright_fisher_mapping.md) + sweep numerico in [`work/07_sergio_branching_sweep/`](work/07_sergio_branching_sweep/)
- Estrazione video-seminario Sergio (formule F1-F15 verificate): [`work/02_deep_dives/08_video_seminar_extracted_insights.md`](work/02_deep_dives/08_video_seminar_extracted_insights.md) — sintesi di [`VideoTranscriptSergio.md`](VideoTranscriptSergio.md) (raw transcript ~75 KB, da leggere insieme)

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

**Sergio's oral knowledge (first-person)**:

- **Radient 2026 podcast** (~2.5h, ~21 700 words) → [`docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md`](docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md) — full transcript structured into 21 argumentative chapters with Italian theses. Spanish dialogue preserved verbatim. **Most direct source for Sergio's intuitive framing**: Wissner-Gross genesis, one-night cochecito, **6-fold optimal branching** (claim contestato — vedi sotto), "frontera caos/orden" as candidate Third Law, FMC vs MCTS quantitative comparison (claim contestato), bengala-vs-laser metaphor for LLM limits, open-source-as-entropy-maximization philosophy.
- **Video seminario su slide** (~2019-2021, raw ~75 KB) → [`VideoTranscriptSergio.md`](VideoTranscriptSergio.md) — trascrizione automatica grezza, senza punteggiatura. Versione pedagogica con slide: cone-entropy → ladder of simplification → algoritmo finale. Aggiunge cross-entropy collapse come framing centrale, metafora minatore (JTBD), coscienza emergente tripla, robustezza al rumore, demo razzo-uncino caotico. **Sintesi strutturata + verifica numerica delle 15 formule estratte** in [`work/02_deep_dives/08_video_seminar_extracted_insights.md`](work/02_deep_dives/08_video_seminar_extracted_insights.md).

**Read both alongside paper #1** for the operational intuition that the math alone doesn't carry.

**⚠️ Discrepanze note tra fonti** (registrate, non chiuse):

| # | Claim | Fonte A | Fonte B | Stato |
|---|---|---|---|---|
| D1 | Branching factor ottimo | Radient 2026 cap.16: "**~6**" | [`docs/MATH_CANON.md`](docs/MATH_CANON.md) Cong. A v0.4.0: $b_{\text{eff}}^*(K, N, M, \alpha) \approx 1 + (K-1) \cdot \mathcal{F}(M/N) \cdot \mathcal{G}(\alpha, K)$ — superficie 4D Wright-Fisher. Power-law $K^{0.6}$ è transiente a $M=15$, non un fixed point. | **FALSIFICATO come universale**. Sergio's "6" è snapshot di $(K=9, M=15, N\sim32{-}64, \alpha=0.1)$ — *triplamente contingente*. Asintoticamente $b_{\text{eff}} \to 1$ (Teorema 2). |
| D2 | FMC sample efficiency vs MCTS | Paper 1803.05049v5 §5.1.2: "**359× fewer**"; §6.2.1: "0.01-0.1%" (1000-10000×); §7: "2-3 OoM"; CLAUDE.md prev: "<1000 vs 3M" | Radient 2026 cap.10: "**~150 000 vs ~35**" (≈4286×) | **🟡 PARZIALMENTE CONFERMATO direzionalmente in-session** (Boxing, n=3, B∈{80,240}, RAM, CPU-only): FMC mean +91 (B=80) +100 (B=240) vs MCTS −5 entrambi → Δ ≈ 100 raw points; MCTS non migliora con budget × 3. Vedi [`work/09_fmc_vs_mcts_replication/REPORT.md`](work/09_fmc_vs_mcts_replication/REPORT.md). **Numero singolo definitivo** richiede full P0 sweep — costo revised dal cluster GPU al **single CPU overnight** (~7 h). Audit completo in [`docs/bibliography/paper_fmc_dhdna_audit.md`](docs/bibliography/paper_fmc_dhdna_audit.md). |
| D3 | Frontera caos/orden come "Third Law" | Radient 2026 cap.16: ipotesi articolata | Mai formalizzata come hypothesis testabile in alcun paper o deep-dive | Aperta — candidata per deep-dive 09 |
| D4 | Reward shaping ricetta universale per FMC su task chain-strutturati | n/a (insight nuovo, originale) | Empiricamente verificata su Craftax: exp03 → exp17 +10pp Crafter via inv-tier stacking + tier-weighted achievement-fire bonus. Pattern monotonico, falsifiche rigorose dei multipli > 1.4×. | **NUOVA Cong. D in MATH_CANON v0.5.0**, P9-P11 add. Replicate su Procgen necessaria. Vedi [`work/05_craftax/autoresearch/HANDOFF.md`](work/05_craftax/autoresearch/HANDOFF.md). |

**Codebases** under [`repos/`](repos/):

| Repo | Ruolo | Status |
|---|---|---|
| `FractalAI_old` | Deprecated NumPy reference (paper #1) | read-only |
| `fragile` | PyTorch/GPU FMC swarm — `core.py:716` chiama `env.step_batch`, `core.py:839` chiama `env.set_state` | active |
| `fragile-rl` | Fragile Mechanics, 2024-2026, successor to Book #2 | active |
| `plangym` | **Hard dep di `fragile`**: estende Gymnasium con `get_state()/set_state()` atomico → fa esistere FMC. 8 backend (Atari/MuJoCo/Box2D/DM/Mario/Retro/Balloon/ClassicControl) + Parallel/Ray vectorization | active 2026-04 |
| `shaolin` | **Dep di `fragile`**: dashboard `holoviews+panel+bokeh` per swarm live (graph viewer per albero walker, RGB streaming) | dep |
| `hydraclick` | **Dep di `fragile`**: Hydra config + Click CLI per sweep di iperparametri | dep |
| `flogging` | **Dep di `fragile`**: structured JSON logging + human-readable colorato | dep |

**Teardown architetturale completo** (cosa hanno implementato, come comporre lo stack, come usarlo come impalcatura per nuovi simulatori al posto dell'HTML): [`docs/architecture/tier1_repos_teardown.md`](docs/architecture/tier1_repos_teardown.md).

**Reference implementation** in-repo: [`fmc-core/`](fmc-core/) — NumPy core (~400 LOC) + JS port bit-identical (1e-12 tolerance), 66 test green, K/M/N benchmark sweeps in [`fmc-core/bench/REPORT.md`](fmc-core/bench/REPORT.md). Adapters disponibili: CartPole, GridWorld, Navigation2D, Pendulum, Rocket, **Atari (RAM + RGB via plangym)** [`fmc-core/src/fmc/envs/atari.py`](fmc-core/src/fmc/envs/atari.py).

**Empirical replication harnesses** in-repo (output del loop 2026-04-28):
- [`work/09_fmc_vs_mcts_replication/`](work/09_fmc_vs_mcts_replication/) — **MCTS-UCT baseline** + budget sweep, Boxing micro-result mostra FMC +100 vs MCTS −5 a B=240 (n=3, CPU)
- [`work/10_atari_replication/`](work/10_atari_replication/) — Atari multi-seed con bootstrap CI95, Boxing slice 5/5 knockout
- [`work/11_ram_vs_img_ablation/`](work/11_ram_vs_img_ablation/) — RAM vs IMG sweep parametrico

**Autoresearch sessions** (Karpathy-style /loop iteration con auto-status gate):
- [`work/05_craftax/autoresearch/`](work/05_craftax/autoresearch/) — branch `autoresearch/exp02-ach-bonus`, **23 esperimenti 2026-04-30 → 2026-05-01**, aggregato Crafter zero-training da 28.46% (baseline v4) a 50.60% (exp17), **+22.1pp appaiato p<0.01** (⚠️ il "50.95% ≈ human-expert" è ritrattato — vedi punto 3 sopra e W33_restatement_onesto.md). HANDOFF, results.tsv, fmc_mutable.py = exp17 final.

**Custom plangym simulators** ([`work/08_simulators/`](work/08_simulators/)): env Python custom che ereditano `PlanEnv` per girare FMC su scenari nostri (rocket-uncino F23, ecc.). Sostituiscono progressivamente le sims HTML statiche.

**Live simulations** ([`simulations/`](simulations/)): rocket, kart, pong, octopus, game-of-life (WebGPU multi-agent), highway, SUMO intersection, rocket validated, $b_{\text{eff}}$ surface 3D. Index in [`simulations/index.html`](simulations/index.html). **In deprecazione**: vedi `work/08_simulators/` per i porting Python+plangym.

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
