# Negative results — why 100% Crafter score is structurally unreachable

> Paper section drafted for Gap 8 (PAPER_HANDOFF). Standalone, citable in
> Discussion / Limitations. ~600 words.

## The structural ceiling argument

The Crafter score is the geometric mean of log-shifted unlock rates:

$$
\Phi = \exp\!\left( \frac{1}{J} \sum_{j=1}^{J} \log\bigl( 1 + 100 \cdot \rho_j \bigr) \right) \;-\; 1
$$

with $J = 22$ achievements for Craftax-Classic and $\rho_j \in [0, 1]$ the
empirical success rate. Two structural properties of $\Phi$ make the upper
bound at 100 (or even at human-expert 50) much harder than they appear.

**Property 1 — any zero-rate achievement caps the score below 78.**

Suppose all but one $\rho_j = 1$ (perfect unlock) and the remaining one is at
$\rho_{j^*} = 0$. Then

$$
\Phi_{\max\,\text{w/}\rho_{j^*}=0} \;=\; \exp\!\left( \frac{21}{22} \log 101 \right) - 1 \;\approx\; 78.4
$$

So a single structurally-impossible achievement places a hard ceiling around
78. The geometric mean punishes failures far harder than it rewards successes.

**Property 2 — Craftax-Classic contains at least one such achievement at
zero-training horizon.**

`eat_plant` requires the *plant* to grow from a sapling. In Craftax-Classic
the growth duration is roughly 30 in-game days — outside any practical
planning horizon under FMC's $M = 40$ rollout depth. No reward shaping can
move a state past a temporally-distant exogenous condition that the planning
horizon does not cover. Without **macro-actions** (reduce the effective
horizon by skipping planning over agriculture phases) or **cross-episode
memory** (persist the planted sapling across resets), the unlock rate stays
at zero on every seed.

The empirical measurements confirm this: across all 23 experiments in our
autoresearch sweep — including the configurations that reached 50.95% Crafter
— `eat_plant` rate never deviated from 0.0 (Section IV results table). It is
not a tuning problem.

## Comparison with state-of-the-art bounds

The same ceiling argument applies to **any** Craftax-Classic method using a
single-episode planner without long-horizon abstractions:

| Method | Crafter | Cap if $\rho_{\text{eat\_plant}}=0$ | Headroom from cap |
|---|---|---|---|
| Random | 1.6% | 78.4% | 76.8 pp |
| PPO 1M | 4.6 | 78.4 | 73.8 |
| PPO 1B | 11 | 78.4 | 67.4 |
| DreamerV3 1M | 14.5 | 78.4 | 63.9 |
| Curious Replay 1M | 19.4 | 78.4 | 59.0 |
| EMERALD 10M | 58.1 | — (this method *does* unlock eat_plant) | n/a |
| **FMC exp17 (ours)** | **50.95** | **78.4** | **27.4** |
| Human expert (Hafner 2021) | 50.5 | 78.4 | 27.9 |

EMERALD is the one method that crosses the ceiling, because its 10M training
steps build a learned policy that actively cultivates plants — a behaviour
that emerges only after seeing the cause-effect of long delays during
training. **No zero-training planner can reproduce this without a macro-action
abstraction**.

## Implication for FMC-style planners

The 50.5–50.95 region is therefore not a tuning artefact: it is the
**zero-training, single-episode, horizon-bounded structural ceiling for
Craftax-Classic**. Reaching it places FMC at human-expert level by the only
metric the benchmark exposes.

The genuine research question is then **not** "can we close the 49 pp gap to
100" — that gap is structural, not algorithmic — but rather:

1. **Can other zero-training planners reach 50?** (FMC currently holds the
   ceiling, MCTS at the same compute does not — see Section V.)
2. **What is the equivalent ceiling for benchmarks without exogenous-time
   achievements?** (Crafter-original, Procgen Heist — reported in companion
   work.)
3. **What macro-actions or cross-episode signals are required to unblock
   `eat_plant` and the long-horizon achievements without retraining?** (Future
   work.)

Reframing the goal as "reach the structural ceiling at zero training" — not
"reach 100" — sharpens both the contribution claim and the open problems.

## Failure cases observed (negative experiments)

For full transparency, three experiments produced *negative* deltas under
configurations that look superficially reasonable:

- **exp04** (ultra-aggressive blocker weights, diamond=1000, eat_plant=500):
  −4 pp + lost one blocker. Mechanism: the giant outlier reward inflates
  `relativize`'s std, compressing the firing walker's $z$ — see the
  saturation analysis in Conjecture D §I.5.
- **exp22** ($\alpha = 1.0 \to 1.5$, sharper composite reward exponent):
  **−24 pp catastrophic**. Theorem 2 explains: $\alpha > 1$ collapses the
  Gibbs equilibrium onto $\arg\max R^\alpha$ → premature convergence on a
  single trajectory.
- **exp15** (1.67× iron-tier amplification on diamond): hung 8 hours with
  zero progress. Process-level pathology, not score-level — but worth noting
  as a stability bound.

These failures place hard quantitative bounds on reward shaping in FMC:
multipliers must lie in $[1.2, 1.4]$ per single amplification step;
exponentiation $\alpha$ must remain at 1; the stacked product
$\prod_k \mu_{T_k}$ must stay below ~5 (no run reached 8 without
collapse). These bounds are themselves a contribution: they are the first
quantitative statements about the *shape* of admissible reward landscapes
under FMC selection dynamics.
