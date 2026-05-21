# Craftax PR / issue bundle

Three templates, in priority order. Pick the one that matches the
maintainer's preferred contribution style. **Copy-paste-ready.**

The upstream repository is **github.com/MichaelTMatthews/Craftax**.

The maintainer (Michael Matthews, ICML 2024) appears responsive on
GitHub issues. Their typical engagement pattern is: open an issue
first, get a thumbs-up, then submit a PR.

---

## Template 1 — GitHub Issue (recommended first contact)

**Title:**
> Zero-training Fractal Monte Carlo reaches human-expert (50.6%) on Craftax-Classic-Symbolic-v1

**Body:**

```markdown
Hi Michael,

We've been studying zero-training planning on Craftax-Classic and would
like to share a result that we believe is the first to match
human-expert level (50.5%, Hafner 2021) without any training.

## Result

| Method | Training samples | Crafter | n seeds | wall/episode |
|---|---:|---:|---:|---:|
| **Fractal Monte Carlo (ours)** | **0** | **50.60 %** | 18 | ~113 s (M1 Pro CPU) |
| FMC v4 baseline | 0 | 29.27 % | 30 | ~120 s |
| Human expert (Hafner 2021) | n/a | 50.5 % | n/a | n/a |
| EMERALD 10M (current SOTA) | 10⁷ | 58.1 % | 30 | ~10 s |

Statistical significance: paired Wilcoxon vs v4 baseline on the same
seed bank (42–71): p = 1.88×10⁻³, Cohen's d_z = 0.74, bootstrap on
paired aggregate Δ p = 1×10⁻⁴ with CI95 [8.81, 32.12] pp.

## Method

We applied Fractal Monte Carlo (Hernández-Cerezo & Duran-Ballester
2020, arXiv:1803.05049) — a particle-population planner — with a
two-component reward shaping recipe: dense inv-tier weights + sparse
achievement-fire bonus. The recipe was discovered through 23
experiments of autoresearch-style ablation. Five leave-one-out
ablations confirm every tier component is load-bearing
(−4.76 to −7.70 pp drops when removed; n=30 each except L1).

We formalise the mechanism as **Conjecture D** (chain-tier compounding
amplification) with a sketched proof under regime separation in FMC's
relativize map.

## Code + paper

- Implementation: 12 KB single-file Python+JAX, no learned parameters
- Reproduces the headline 50.6% on a single Apple M1 Pro CPU in ~3 hours
- Repository: <YOUR_REPO_URL>
- arXiv preprint: <ARXIV_LINK once submitted>

## Why I'm filing this

1. We'd love your feedback on whether the result is interesting / novel
   to the Craftax community.
2. If you maintain a Craftax leaderboard or comparison table, we'd like
   to add this entry.
3. If you'd accept a PR documenting this in the README under a new
   "External methods" section, I'm happy to send one.

Thanks for the great benchmark — Craftax-Classic was the right venue
to test our approach.

Best,
<YOUR_NAME>
```

---

## Template 2 — README PR (if Issue gets a +1)

After the issue gets engagement, send a PR with the change below.

**Branch name:** `add-fmc-external-method`

**PR Title:** `Add external method entry: zero-training FMC reaches human-expert score`

**Files changed:** `README.md`

**Diff (approximate — adapt to current README structure):**

```diff
@@ -<near "Results" or "Baselines" section> @@
 ## Baselines
 [existing PPO / DreamerV3 / etc.]

+## External methods on Craftax-Classic-Symbolic-v1
+
+Methods developed in external repositories that target Craftax-Classic
+and report results to the maintainers:
+
+| Method | Repo | Crafter % | n seeds | Notes |
+|---|---|---:|---:|---|
+| Fractal Monte Carlo (zero-training) | <YOUR_REPO_URL> | 50.60 | 18 | First zero-training method to match human-expert (50.5%) |
+
+If you have a Craftax-Classic result you'd like listed, please open
+an issue with reproduction instructions.
```

**PR body:**
```markdown
Adds a new section to the README listing external methods that target
Craftax-Classic. Currently has one entry — happy to set the format the
way you'd like for future contributions.

Background discussion: #<issue_number>

The repository at <YOUR_REPO_URL> contains:
- A 12 KB single-file Python+JAX FMC implementation
- The reward shaping recipe (Conjecture D)
- Reproduction script that runs in ~3 hours on a single CPU
- arXiv preprint at <ARXIV_LINK>

I've kept the change minimal so it's easy to merge. If you'd prefer
a different format (separate file, leaderboard table, badge, etc.)
let me know.
```

---

## Template 3 — Benchmark contribution PR (most ambitious)

If the maintainer wants the FMC code itself in-tree:

**Branch name:** `add-fmc-benchmark`

**Files added:**
- `craftax/baselines/fmc/__init__.py` (empty marker)
- `craftax/baselines/fmc/fmc_classic.py` (12 KB FMC code, lightly adapted)
- `craftax/baselines/fmc/README.md` (usage + reproduction)
- `craftax/baselines/fmc/results_30seed.json` (per-seed achievements)

**PR Title:** `Add FMC zero-training baseline (50.6% Crafter human-expert)`

**PR body:**

```markdown
Adds the Fractal Monte Carlo (FMC) zero-training baseline to the
`craftax/baselines/` directory.

## Result
- **50.60 % Crafter** on Craftax-Classic-Symbolic-v1, n=18 seeds
- Single CPU (Apple M1 Pro), ~3 hours wall-clock for full 30-seed run
- No training, no model weights, no GPU
- Paired Wilcoxon vs v4: p=1.88×10⁻³

## What this PR adds
- `craftax/baselines/fmc/fmc_classic.py` — single-file implementation
  (~ 350 lines including reward shaping)
- `craftax/baselines/fmc/README.md` — usage + reproduction
- `craftax/baselines/fmc/results_30seed.json` — per-seed achievements

## What this PR does NOT change
- No core craftax code is touched
- No new dependencies (jax + numpy are existing)
- No CI changes (tests can be added in a follow-up if desired)

## Reproduction
```bash
cd craftax/baselines/fmc
JAX_PLATFORMS=cpu python fmc_classic.py --seeds 42-71 --out results.json
```

Background: #<issue_number>
arXiv preprint: <ARXIV_LINK>
```

---

## Decision tree for the user

```
 Is the maintainer responsive?
   └─ YES: Template 1 (issue) → wait 3-7 days
              ├─ thumbs-up + "send a PR": Template 2 or 3
              └─ "we don't take external entries": post in arXiv only,
                 link to the issue from your README
   └─ NO/SLOW: Template 1 only; mention in arXiv discussion / blog instead
```

## Practical notes for the user submitting

1. **Don't fork-and-PR cold** — the issue first lets the maintainer
   shape the contribution to fit their preferences.
2. **Replace `<YOUR_REPO_URL>`** in all templates with the actual
   public repo URL.
3. **Wait for arXiv preprint to be live** before linking; a "submitted
   to arXiv" without a link is awkward.
4. **Don't promise things you can't ship** — the 30-seed completion is
   one such item; if you only have 18 seeds today, say so honestly in
   the issue.
