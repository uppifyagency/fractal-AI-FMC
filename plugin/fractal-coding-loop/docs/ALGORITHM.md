# ALGORITHM — the math, the pseudocode, the paper references

> *This document walks through the FMC algorithm as implemented in the plugin, with explicit references to paper sections, code line numbers, and worked numerical examples.*

For higher-level rationale ("why these mappings"), see [`THEORY.md`](THEORY.md). For component-level documentation see [`COMPONENTS.md`](COMPONENTS.md). For invocation see [`USAGE.md`](USAGE.md).

---

## 1. The core algorithm at one page

```
GOAL: produce ONE commit toward goal G, given current state x_0.

INPUT:  current git HEAD (x_0), goal G, hyperparameters (N, M, α, β, ess_threshold)
OUTPUT: init_commit_sha to cherry-pick onto main

# ─── Phase 0: spawn N walkers in distinct strategies ─────────────
for i in 1..N:
    walker[i].worktree = git worktree add fresh from x_0
    walker[i].init_action_label = strategy_i  (e.g. "extract-module")
    walker[i].init_action_desc = description_i

# ─── Phase 1: tick 0 (init) ──────────────────────────────────────
for i in 1..N parallel:
    sub_agent(walker[i], MODE=init)
    → walker[i] commits ONE commit implementing strategy_i
    → walker[i].init_commit_sha = HEAD of walker's branch

record(walkers, tick=0)

# ─── Phase 2: M-1 ticks of step + clone + perturbation ───────────
for t in 1..M-1:
    # 2a. Compute virtual reward and clone plan
    R_i      = composite_reward(walker[i])              # paper §2.2.2
    R_norm   = relativize(R)                             # paper §2.2.3
    j(i)     = random partner of i, j != i
    D_i      = jaccard_distance(walker[i], walker[j(i)]) # paper §4.5
    D_norm   = relativize(D)
    VR_i     = R_norm[i]^α × D_norm[i]^β                 # paper §4.4

    ESS      = (Σ VR_i)² / Σ VR_i²                       # Doucet 2001
    if ESS > ess_threshold × N:
        skip cloning
    else:
        for i in 1..N:
            k = random partner != i
            p_clone = clone_probability(VR_i, VR_k)      # paper §4.4
            if random() < p_clone:
                walker[i].worktree ← git reset --hard walker[k].HEAD
                walker[i].init_action_label  ← walker[k].init_action_label
                walker[i].init_commit_sha    ← walker[k].init_commit_sha

    # 2b. Continuation perturbation
    for i in 1..N parallel:
        sub_agent(walker[i], MODE=continuation)
        → walker[i] makes one small step toward G

    record(walkers, tick=t)

# ─── Phase 3: final decision ──────────────────────────────────────
counts = Counter(w.init_action_label for w in walkers if w.alive)
winner_label = argmax(counts)                            # paper §4.6
representative = max(walkers with label==winner_label, key=latest_R)

return {
    winner_label: winner_label,
    winner_init_commit_sha: representative.init_commit_sha,
    confidence: counts[winner_label] / Σ counts,
}
```

The rest of this document explains each of those mathematical pieces in detail.

---

## 2. `relativize` — paper §2.2.3, the unique reward reshape

### 2.1 The formula

```
R_N = (R - μ) / σ                     z-score the input
R̂(R) = exp(R_N)            if R_N ≤ 0
R̂(R) = 1 + log(1 + R_N)    if R_N >  0
```

**Implementation** ([`scripts/fractal_reward.py`](../scripts/fractal_reward.py) lines 115-134):

```python
def relativize(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return values
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    if std == 0 or not math.isfinite(std):
        return [1.0] * n
    out = []
    for v in values:
        z = (v - mean) / std
        if z <= 0:
            out.append(math.exp(max(z, -50)))
        else:
            out.append(1.0 + math.log1p(z))
    return out
```

### 2.2 What it does and why

Relativize maps any real vector to **strictly positive values** while preserving order. Specifically:

| Property | Effect |
|---|---|
| Strictly positive | Required for VR = R^α to be well-defined (paper §2.2) |
| Order-preserving | Better walkers stay better |
| Affine-invariant | Output unchanged under R → aR + b (z-score absorbs scale and shift) |
| Compresses high outliers | `1 + log(1+z)` grows slowly — one walker with 100× the reward of others doesn't dominate |
| Expands low outliers | `exp(z)` near zero for very negative z — dead walkers get tiny but nonzero VR |

### 2.3 Numerical example

For input `[-100, 0, 100]` (extreme outliers in both directions):

- mean = 0
- std ≈ 81.65
- z-scores: -1.225, 0, 1.225
- output: `[exp(-1.225), 1.0, 1 + log(1+1.225)] = [0.294, 1.0, 1.799]`

The 100x range in input becomes a 6x range in output. Tractable in subsequent multiplications.

For uniform input `[5, 5, 5]`: std=0 → returns `[1, 1, 1]`. No information, equal weighting.

### 2.4 Why this specific formula

Deep dive [`04_relativize_axiomatics.md`](../../../work/02_deep_dives/04_relativize_axiomatics.md) outlines a proposed uniqueness theorem: under five axioms (positivity, order preservation, affine invariance, logarithmic compression at +∞, sub-exponential decay at -∞), the formula is unique up to constants. This explains why Sergio's choice is not ad hoc.

### 2.5 Test that certifies correctness

[`tests/test_fractal_math.py:test_relativize_properties`](../tests/test_fractal_math.py) lines 76-92:

- Verifies strict positivity on `[1, 2, 3]`
- Verifies order preservation
- Verifies constant input returns ones
- Verifies extreme range `[-100, 0, 100]` → `[0.294, 1.0, 1.799]` (printed as part of test output)

---

## 3. Composite reward — paper §2.2.2 + post-Pareto

### 3.1 The formula

For each walker, given walker_json output:

```
R_alive = 1 if compile_ok else 0                   HARD constraint
R_tests = passed / total           ∈ [0, 1]        HARD constraint (multiplicative)
R_lint  = 1 / (1 + log(1 + warnings))   ∈ [0, 1]   SOFT constraint
R_diff  = max(0, 1 - lines/200)         ∈ [0, 1]   SOFT constraint
R_goal  = judge_score (default 0.7)     ∈ [0, 1]   SOFT constraint

R = R_alive × R_tests × (1 + R_lint) × (1 + R_diff) × (1 + R_goal)
```

**Implementation** ([`scripts/fractal_reward.py`](../scripts/fractal_reward.py) lines 86-108):

```python
def composite_reward(walker: dict) -> dict:
    alive = r_alive(walker)
    tests = r_tests(walker)
    lint = r_lint(walker)
    diff = r_diff(walker)
    goal = r_goal(walker)
    R = alive * tests * (1.0 + lint) * (1.0 + diff) * (1.0 + goal)
    return {"R": R, "R_alive": alive, ...}
```

### 3.2 Why multiplicative not additive

Sergio's [April 2016 blog post](../../../docs/bibliography/sources/blog_posts/2016-04_pareto_frontiers.md) explicitly rejects Pareto:

> *"Real-world problems typically have single underlying objectives. We only have one goal in life — maximizing long-term well-being."*

Pareto-style additive `R = w₀·R_alive + w₁·R_tests + w₂·R_lint + ...` allows a walker with broken tests but pretty lint to score nonzero. This is wrong: broken tests should kill the walker.

Multiplicative `R = R_alive × R_tests × (1+R_lint)` zeroes out the entire reward when any hard constraint fails. **Death in one component = death overall.** Aligned with paper §2.2.2:

> R(s) = R_0(s) × R_1(s) × ... × R_n(s)

### 3.3 Why hard vs soft distinction (multiplication vs `(1 + x)`)

Hard constraints (`R_alive`, `R_tests`) **must** be allowed to zero the reward. They use direct multiplication.

Soft contributions (lint, diff, goal) **must not** zero the reward when they're 0 (a walker with 0 lint warnings should be **rewarded**, not zeroed). They use `(1 + x)` — adds at minimum 1× to the product.

### 3.4 Numerical example

Walker with: `compile_ok=True`, `tests_passed=10/10`, `lint_warnings=0`, `lines_added=10, lines_deleted=0` (10 line diff), `goal_score=0.7` (default):

- R_alive = 1
- R_tests = 10/10 = 1
- R_lint = 1 / (1 + log(1+0)) = 1 / 1 = 1
- R_diff = max(0, 1 - 10/200) = 0.95
- R_goal = 0.7
- R = 1 × 1 × (1+1) × (1+0.95) × (1+0.7) = 1 × 1 × 2 × 1.95 × 1.7 = **6.63**

For a walker with broken tests (`tests_passed=1/10`):

- R_tests = 0.1
- R = 1 × 0.1 × 2 × 1.95 × 1.7 = 0.663 (10× lower)

For a walker that fails to compile (`compile_ok=False`):

- R_alive = 0
- R = 0 (regardless of other components)

This was verified in the smoke test — see the values printed during `step` output.

---

## 4. Virtual reward — paper §4.4

### 4.1 The formula

```
VR_i = relativize(R)[i]^α × relativize(D)[i]^β
```

where D_i is the distance from walker_i to a randomly chosen partner.

**Implementation** ([`scripts/fractal_reward.py:virtual_reward`](../scripts/fractal_reward.py) lines 160-194):

```python
def virtual_reward(walkers, rewards, alpha=1.0, beta=1.0):
    n = len(walkers)
    distances = []
    for i in range(n):
        partner = random.choice([j for j in range(n) if j != i])
        d = file_overlap_distance(walkers[i], walkers[partner]) + ...
        distances.append(d)
    R_norm = relativize(rewards)
    D_norm = relativize(distances)
    return [(R_norm[i] ** alpha) * (D_norm[i] ** beta) for i in range(n)]
```

### 4.2 What α and β mean

| Setting | Meaning | Behavior |
|---|---|---|
| α = β = 1 (default) | Equal exploration/exploitation | Standard FMC |
| α > β | Reward dominates | Greedy, less exploration |
| α < β | Distance dominates | Exploratory |
| α = 0 | "Common Sense" / Empowerment | No reward weighting; pure diversity. Drone autopilot mode of paper §6.3 |
| β = 0 | No exploration | Walkers collapse to high-R region (paper §4.5 lemma 4) |

### 4.3 Why distance is computed against a random partner (O(N) stochastic)

Paper §4.5: instead of computing all O(N²) pairwise distances, compute one distance per walker against a random partner. This is an unbiased Monte Carlo estimator of the local density inverse.

**Practical reason**: O(N²) is wasteful when N is large. O(N) keeps things tractable.

**Theoretical reason**: in expectation, a random partner samples the empirical distribution. So `E[D_i]` over runs converges to `1 / ρ(W_i)` where ρ is local walker density. This gives the algorithm its anti-collapse property (deep dive [`01_cloning_mathematics.md`](../../../work/02_deep_dives/01_cloning_mathematics.md) Lemma 4).

### 4.4 What "distance" means in the coding context

Atari uses L2 distance on RAM (128 bytes). For coding, the plugin uses Jaccard distance on the set of files changed by each walker:

```
d(W_i, W_j) = 1 - |files_i ∩ files_j| / |files_i ∪ files_j|
            + 0.01 × |lines_changed_i - lines_changed_j|
```

**Implementation** ([`scripts/fractal_reward.py:file_overlap_distance`](../scripts/fractal_reward.py) lines 137-150):

```python
def file_overlap_distance(w1, w2):
    f1 = set(w1.get("files_changed", []))
    f2 = set(w2.get("files_changed", []))
    if not f1 and not f2:
        return 0.0
    union = len(f1 | f2)
    if union == 0:
        return 0.0
    jaccard = len(f1 & f2) / union
    return 1.0 - jaccard
```

Walkers touching the same files are CLOSE (Jaccard high → distance low). Walkers touching disjoint sets are FAR (Jaccard 0 → distance 1).

The `+ 0.01 × lines_diff` adds a tiebreaker for walkers that hit the same files but with different intensity. Small coefficient — files-changed dominates.

### 4.5 Numerical example from the smoke test

Three walkers with R=(6.63, 2.31, 0.371) and disjoint file sets `{a.py}, {b.py}, {c.py}`:

- Jaccard distances pairwise = 1.0 (all disjoint)
- Random partner pairing produced (in seed=42): walker_2 paired with someone disjoint
- relativize(R) ≈ (1.80, 1.00, 0.29)
- relativize(D) ≈ (1.0, 1.0, 1.0) (all distances are 1, std=0, returns ones)

So in this case VR ≈ R_norm (because D normalizes to uniform). The smoke test showed VR=(0.91, 0.36, 0.66) — different because the actual run had different distance values from the partner sampling.

---

## 5. Pairwise stochastic clone — paper §4.4

### 5.1 The formula

For each walker i, choose random partner k. Compute:

```
P_clone(i → k) = 1                                  if VR_i = 0
P_clone(i → k) = 0                                  if VR_k ≤ VR_i
P_clone(i → k) = (VR_k - VR_i) / VR_i              otherwise (capped at 1)
```

If random() < P_clone, walker_i adopts walker_k's state and label.

**Implementation** ([`scripts/fractal_loop.py:cmd_step`](../scripts/fractal_loop.py) lines 200-220):

```python
vr_i, vr_p = vrs[i], vrs[partner]
if vr_i <= 1e-8:
    prob = 1.0
elif vr_p <= vr_i:
    prob = 0.0
else:
    prob = min(1.0, (vr_p - vr_i) / vr_i)
if random.random() < prob:
    clone_plan.append(_make_clone_entry(...))
```

### 5.2 Why this specific formula (and not something simpler)

Why not just "always copy from a better partner"? Two reasons:

1. **Variance preservation**: stochastic cloning maintains diversity in the swarm. Greedy copying would collapse.
2. **Detailed balance**: the probability `(VR_k - VR_i) / VR_i` produces the correct stationary distribution `π* ∝ R^α` (deep dive [`01_cloning_mathematics.md`](../../../work/02_deep_dives/01_cloning_mathematics.md) Theorem 3, detailed balance argument). With greedy or other formulas, the convergence guarantees fail.

In information-theoretic terms, this is the **Metropolis-Hastings acceptance probability** for the target distribution `R^α`. Specifically, `(VR_k - VR_i) / VR_i` is the ratio that produces detailed balance for a Gibbs-like distribution.

### 5.3 The dead-walker case

If `VR_i = 0` (walker died: `compile_ok=False` → R_alive=0 → R=0 → VR=0), `P_clone = 1`. Dead walkers **always** clone. They are revived from any alive partner.

This is handled in [`scripts/fractal_loop.py:cmd_step`](../scripts/fractal_loop.py) lines 195-209:

```python
if not walkers_data[i]["alive"]:
    alive_partners = [j for j in range(n) if j != i and walkers_data[j]["alive"]]
    if not alive_partners:
        continue
    partner = random.choice(alive_partners)
    clone_plan.append(_make_clone_entry(state, i, partner, "dead_revive", 1.0))
    continue
```

Verified by [`tests/test_fractal_math.py:test_dead_walker_rare_in_winner`](../tests/test_fractal_math.py): with R=(0.9, 0.5, 0.0) and M=20, walker_2 (R=0) ends up in 0.3% of bincount winners. The death-revive cycle squeezes it out almost completely.

### 5.4 What the clone propagates

When walker_i clones from walker_k, three things change ([`scripts/fractal_loop.py:cmd_apply_clones`](../scripts/fractal_loop.py) lines 248-285):

1. **`current_head`** = walker_k's HEAD (worktree state, applied via `git reset --hard` by the orchestrator)
2. **`init_action_label`** = walker_k's label (the strategy name — what's being voted on)
3. **`init_commit_sha`** = walker_k's `init_commit_sha` (the SHA of the FIRST commit on walker_k's lineage — this is what gets cherry-picked at decision time)

The label propagation is the **branching marker** of the auxiliary particle filter view (deep dive [`05_smc_particle_filter_view.md`](../../../work/02_deep_dives/05_smc_particle_filter_view.md) §2.3.3). Without it, FMC would just be a particle filter, not a decision-maker.

---

## 6. ESS-adaptive cloning — Doucet et al. 2001

### 6.1 The formula

```
ESS_t = (Σ VR_i)² / Σ VR_i²        range [1, N]
```

If `ESS_t > threshold × N` (default threshold = 0.7), skip cloning that tick.

**Implementation** ([`scripts/fractal_loop.py:cmd_step`](../scripts/fractal_loop.py) lines 175-188):

```python
sum_vr = sum(vrs)
sum_vr_sq = sum(v * v for v in vrs)
ess = (sum_vr ** 2) / sum_vr_sq if sum_vr_sq > 1e-12 else 0.0
ess_threshold_abs = state["ess_threshold"] * n
skip_cloning = ess > ess_threshold_abs
```

### 6.2 What ESS measures

Effective Sample Size measures how many "effectively independent" walkers exist in the swarm:

- ESS = N: all walkers equal weight, maximum diversity
- ESS = 1: one walker dominates, others are redundant copies

When ESS is high, the swarm already represents the distribution well. Resampling adds noise without selection benefit. When ESS is low, resampling concentrates mass on high-VR walkers — which is the whole point.

The 0.7 threshold is conventional (Doucet 2001). With N=3, threshold = 2.1 — if ESS > 2.1 (i.e., >70% of walkers contribute meaningfully), skip cloning.

### 6.3 Why this matters for cost

In Atari, skipping cloning saves microseconds per tick. Negligible.

For the plugin, skipping cloning saves **the next tick of N sub-agent calls**. Why? Because cloning forces walker_i to abandon its trajectory and continue from walker_k's state. Without cloning, walker_i continues from its own state, no work redone.

At N=3, M=3, skipping one tick of cloning saves ~3 sub-agent calls = ~$1.50 + ~90s wall time. Per `/fractal-decide`, this is 30-50% savings if ESS stays high throughout.

### 6.4 The smoke test result

In the CLI smoke test of [`scripts/fractal_loop.py`](../scripts/fractal_loop.py):

```
=== step (compute VR + ESS) ===
  ess: 2.677 / threshold: 2.100
  cloning_skipped: True
```

ESS=2.68 > 2.10 → cloning skipped, zero git resets, zero forced re-perturbation. Verified behavior on real CLI invocation.

---

## 6bis. Wigner reward for memory recall — Slide doc 2020

The plugin includes a **memory bank** that stores past `/fractal-decide` episodes and exposes Wigner-weighted recall. This implements the **Dataset as Fractal Memory** concept from the [2020 Fractal Slide doc](../../../2020%20Fractal%20Slide.md).

### 6bis.1 The formula

```
R'(x) = (π/2) · x · exp(-π/4 · x²)        x = loss / avg_loss
weight(memory) = R'(x) / (1 + log(1 + visits))
```

**Implementation** ([`scripts/fractal_memory.py`](../scripts/fractal_memory.py) lines 197-210):

```python
def wigner_reward(loss: float, avg_loss: float) -> float:
    if avg_loss <= 0:
        return 0.0
    x = loss / avg_loss
    return (math.pi / 2.0) * x * math.exp(-(math.pi / 4.0) * x * x)

def memory_weight(loss: float, avg_loss: float, visits: int) -> float:
    R = wigner_reward(loss, avg_loss)
    return R / (1.0 + math.log1p(visits))
```

### 6bis.2 What this distribution looks like

The Wigner semicircle from random matrix theory peaks near `x = 1` (= average loss):

```
weight ↑
       |       Wigner curve
       |        ╱╲
   max |       ╱  ╲          ← peak around x=1 (loss == avg)
       |     ╱      ╲
       |   ╱          ╲___
       | ╱                ‾─
       └────┬────┬────┬────┬────→  x = loss/avg
            0    1    2    3
```

### 6bis.3 Why this specific shape for memory

- **Low-loss memories** (`x → 0`): already mastered. `R'(x) → 0` linearly. Deprioritize.
- **Medium-loss memories** (`x ≈ 1`): active learning zone. **Maximum weight.** Surface these.
- **High-loss memories** (`x → ∞`): too difficult for current state. `R'(x)` decays exponentially. Postpone until simpler.

This is exactly the **automatic curriculum learning** described in the slide doc: examples are processed in waves from easiest to hardest, with the current "frontier" getting most attention.

### 6bis.4 What `loss` means for a coding decision

The plugin uses `loss = 1 - confidence` of the past decision. Confidence is computed at decision time (see §7.1: `confidence = bincount[winner_label] / sum(bincount)`).

- High-confidence decisions (`conf=0.9`) → low loss (0.1) → low Wigner weight → recall less often
- Medium-confidence decisions (`conf=0.5`) → high loss (0.5) → high Wigner weight → recall often
- Low-confidence decisions (`conf=0.1`) → very high loss (0.9) → high but past-peak weight

The interpretation: medium-confidence past decisions are the most informative — they're the ones where the swarm was uncertain and the resolution probably reveals interesting structure. High-confidence past decisions are settled; low-confidence ones may have been noise.

### 6bis.5 Visit-debiasing

Every recall increments the `visits` counter on the surfaced memories. The `1 / (1 + log(1+visits))` factor reduces their weight over time, ensuring the system doesn't get stuck recalling the same memories.

After `visits = 1`: factor = `1 / log(2)` ≈ 1.443
After `visits = 10`: factor = `1 / log(11)` ≈ 0.417
After `visits = 100`: factor = `1 / log(101)` ≈ 0.217

### 6bis.6 Verification

`e2e_test.sh` Test 5 verifies the shape:

```python
values = [wigner_reward(x, 1.0) for x in [0.5, 1.0, 1.5, 3.0]]
assert values[1] >= max(values[0], values[2], values[3]) * 0.95
```

The peak at `x=1.0` is greater than 95% of any other tested point. Verified over `e2e_test.sh` 17/17.

### 6bis.7 Where this is used in the pipeline

Currently, the memory subsystem is **invoked manually** via `/fractal-recall` and `/fractal-memory-show`. It is **NOT** auto-called from `/fractal-decide`. The plumbing for auto-call is documented as Phase 8b in [`commands/fractal-decide.md`](../commands/fractal-decide.md) — a one-shell-call addition planned for a near-future iteration.

Once wired, the integration would be:

1. After `/fractal-decide` Phase 7 (showing user the comparison), Phase 8b auto-appends the episode to the memory bank
2. Future `/fractal-decide` Phase 1 (strategy generation) starts by calling `/fractal-recall <task slug>` and biases strategy proposal toward what worked previously

This closes the loop: past decisions inform future decisions, weighted by Wigner.

---

## 7. The decision — paper §4.6

### 7.1 The formula

After M ticks:

```
counts[label] = #{i : walker_i alive AND walker_i.init_action_label == label}
winner_label = argmax_label counts[label]
representative = walker with label==winner_label and highest current R
return representative.init_commit_sha
```

**Implementation** ([`scripts/fractal_loop.py:cmd_decide`](../scripts/fractal_loop.py) lines 280-328):

```python
counts = Counter()
label_to_walkers = {}
for w in state["walkers"]:
    if w["alive"] and w["init_action_label"]:
        counts[w["init_action_label"]] += 1
        label_to_walkers.setdefault(w["init_action_label"], []).append(w)

winner_label = counts.most_common(1)[0][0]
representative = max(label_to_walkers[winner_label], key=latest_R)
```

### 7.2 Why bincount is enough (and what it represents)

The bincount over `init_action_label` is the **marginal posterior** over initial actions. From the convergence theorem (deep dive [`01_cloning_mathematics.md`](../../../work/02_deep_dives/01_cloning_mathematics.md) Theorem 5):

```
P̂[ ℓ_i = a ]  →  R(x | a) / R_TOT  as N → ∞
```

That is: after M ticks, the fraction of walkers whose initial action was `a` approximates the reward density of the sub-cone starting with action `a`. The argmax is the action whose sub-cone has highest expected reward.

This is **paper §3.2**: "Decision = argmax_a ID(a) where ID(a) ∝ entropy of P_S given initial action a." The plugin uses bincount as the empirical approximation.

### 7.3 Why we extract `init_commit_sha`, not the full walker branch

The init_commit_sha is the SHA of the **first commit** on the winning walker's lineage. That commit alone embodies the strategy choice. The walker's subsequent commits (continuations at t > 0) are *exploratory rollouts* — they're not the decision, they're projections used to evaluate the decision.

This matches Atari: at each FMC.decide, the output is **one action** applied for **one frame**. The walker's 14-frame rollouts are scaffolding. Same here: the walker's 2-3-commit rollouts are scaffolding; only the first commit goes to main.

**Why this is critical**: cherry-picking the entire walker branch would apply work that the walker did with **stochastic continuation prompts** — work whose only purpose was to score the strategy. That work might be of low quality. Only the t=0 init commit was generated with the explicit strategy intention.

---

## 8. The outer loop — `/octopus`

### 8.1 The pseudocode

```
GOAL = $ARGUMENTS
K_MAX = 10
THRESHOLD = 0.95

iteration = 0
goal_score = 0.0

while iteration < K_MAX and goal_score < THRESHOLD:
    decision = run_fractal_decide(goal=GOAL)         # one /fractal-decide invocation
    if decision.confidence < 0.50:
        ASK_USER(continue / stop / inspect)

    git checkout MAIN_BRANCH
    git cherry-pick decision.winner_init_commit_sha
    if cherry_pick_failed:
        STOP and ask user

    goal_score = run_goal_judge(main_HEAD, GOAL)
    iteration += 1

if goal_score >= THRESHOLD:
    REPORT "GOAL REACHED in {iteration} iterations"
else:
    REPORT "BUDGET EXHAUSTED, partial progress: {goal_score}"
```

**Specification**: [`commands/octopus.md`](../commands/octopus.md).

### 8.2 Why goal-checking via a judge sub-agent

Tests passing on main is a NECESSARY condition for goal-reached, but rarely sufficient. The goal might be "implement feature X" — tests for X might pass (R_tests=1) without the feature actually being complete (because the test scope doesn't cover all aspects).

The judge ([`agents/fractal-judge.md`](../agents/fractal-judge.md)) reads the diff and evaluates against the goal description in natural language. It returns a scalar `goal_score ∈ [0, 1]`:

- 1.0 = fully complete
- 0.95 = essentially complete, only polish remains (THRESHOLD default)
- 0.7 = substantial progress, more work needed
- 0.0 = no meaningful progress

This is the same mechanism Atari uses for `game_over()` — a problem-specific terminal condition. For Atari it's KO or 100 points or timeout. For coding it's "judge says ≥ 0.95."

### 8.3 Why the confidence gate

If the FMC decision had confidence < 0.50, it means the walker bincount was split nearly evenly. Two interpretations:

1. **The two strategies are equally good**: any choice is fine. Pick one and proceed.
2. **The reward signal is too noisy**: the swarm couldn't distinguish. Risk: applying random-flavored decisions to main pollutes the trunk.

Without more data, we can't tell. The conservative choice is to STOP and ask the user. The orchestrator surfaces this in [`commands/octopus.md`](../commands/octopus.md) Phase 1b.

### 8.4 Why cherry-pick conflicts stop the loop

If cherry-pick fails with merge conflict, it means: the walker did its work assuming a base state, and the main branch is now in a divergent state from that base. Continuing automatically would either:

1. Force a resolution that may be wrong (bad)
2. Skip this iteration silently (worse — the swarm "decided" something but main didn't get it)

Stopping for user intervention is the only safe option. The orchestrator surfaces conflicts and offers `git reset --hard $START_HEAD` as the reset.

---

## 9. The convergence guarantee, made explicit for the plugin

From deep dive [`05_smc_particle_filter_view.md`](../../../work/02_deep_dives/05_smc_particle_filter_view.md) §3.1, the Del Moral 2004 theorem applies. For the plugin's specific configuration:

| Parameter | Value | Implication |
|---|---|---|
| N (walkers) | 3 (default) | Expected error in action selection ~1/√3 ≈ 58% |
| N (raised) | 10 | ~32% |
| N (raised) | 30 | ~18% (Atari paper's regime) |
| N (Atari paper) | 300 | ~6% |
| M (ticks) | 3 (default) | Mixing time of the Markov chain — see deep dive 01 §7 |
| α | 1.0 (default) | Standard exploitation |
| β | 1.0 (default) | Standard exploration |
| ess_threshold | 0.7 | Skip clone if 70%+ effective walkers |

These are all tunable via `fractal_loop.py init` flags. See [`USAGE.md`](USAGE.md) §"Tuning N and M" for guidance.

---

## 10. End-to-end mathematical certification

The chain of certified correctness, layer by layer:

| Layer | Certification |
|---|---|
| `relativize` | Test 1 of test_fractal_math.py: properties verified deterministically |
| Clone probability formula | Test 2: 5 edge cases verified arithmetically |
| Convergence to Gibbs delta (R=0.9, 0.5, 0.1, M=10, N=3) | Test 3: walker_0 wins 200/200 runs |
| Close-top distribution (R=1.0, 0.95, 0.0, M=8) | Test 4: 59%/41%/0% — non-trivial Gibbs split |
| Dead walker squeezed out (R=0.9, 0.5, 0.0, M=20) | Test 5: walker_2 wins ≤ 0.3% of runs |
| ESS-adaptive cloning trigger | CLI smoke test: ESS=2.68 > 2.10 → cloning_skipped=True |
| init_commit_sha propagation through clones | CLI smoke test: winner_init_commit_sha matches expected |
| Walker dual-mode protocol | Documented in agents/fractal-walker.md, NOT integration-tested with real LLM |
| /fractal-decide M-tick orchestration | Documented in commands/fractal-decide.md, NOT integration-tested |
| /octopus outer loop | Documented in commands/octopus.md, NOT integration-tested |

The test runner ([`tests/test_fractal_math.py`](../tests/test_fractal_math.py)) executes all 5 math tests in <1 second deterministically. Run it before any commit that touches `scripts/fractal_reward.py` or `scripts/fractal_loop.py`.
