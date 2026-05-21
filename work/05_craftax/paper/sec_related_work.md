# Related work

> Drafted for paper section "Related Work" (Gap 10). ~750 words. References
> use BibKeys defined in `references.bib`. The structure mirrors the
> standard "where this paper sits" expected by RLC / NeurIPS reviewers.

## Fractal Monte Carlo and predecessors

Fractal Monte Carlo (FMC) was introduced as an entropy-driven planner in
the Fractal AI series \citep{hernandez2020fractal,hernandez2018solving},
itself a continuation of the General Algorithmic Search framework
\citep{hernandez2017gas}. The algorithm combines population-based rollouts
with a `relativize` map that normalises walker rewards into two regimes
(exponential below mean, logarithmic above), then applies cloning
proportional to relativized advantage. The mathematical foundation rests
on generalised entropies \citep{amigo2018entropy} and inherits its
physical motivation from causal entropic forces
\citep{wissner2013causal}: walker populations behave as if maximising the
"future option diversity" of paths through state space.

Compared to the classical UCT variant of MCTS \citep{kocsis2006uct} and
its learned-model successors AlphaZero \citep{silver2017alphazero},
MuZero \citep{schrittwieser2020muzero}, FMC dispenses with the
exploration-exploitation bandit machinery in favour of an explicit
particle-population analogue. Probabilistic planning with sequential
Monte Carlo \citep{piche2018probabilistic} is the closest related family;
FMC can be viewed as a particular SMC particle filter where the proposal
is the simulator's transition kernel and the resampling kernel is the
relativize-cloning combination.

The original FMC papers reported strong sample efficiency on Atari with
~10³ samples per decision instead of MCTS's 10⁶ \citep{hernandez2018solving},
but until the present autoresearch session the algorithm had received no
systematic empirical study on benchmarks with **chain-completion structure**
(crafting trees with hard prerequisite ordering).

## Crafter and Craftax benchmarks

The Crafter benchmark \citep{hafner2021crafter} provides a 22-achievement
2D survival environment with a hard tier hierarchy (wood → stone → iron →
diamond) and a structured score formula

$$
\Phi = \exp\!\bigl( \tfrac{1}{J}\sum_{j} \log(1 + 100 \rho_j)\bigr) - 1
$$

that punishes any zero-rate achievement disproportionately. The original
paper reported PPO at 4.6 % (1 M steps) and human-expert at 50.5 %.
Subsequent learned-model methods made steady but slow progress: DreamerV3
\citep{hafner2023dreamer} reached 14.5 % at 1 M steps; Curious Replay
\citep{kauvar2023curious} 19.4 % at 1 M steps; the recent EMERALD method
\citep{liu2024emerald} pushes the leaderboard to 58.1 % at 10 M steps.

Craftax \citep{matthews2024craftax} provides a JAX-vectorised
implementation of (a slight variant of) the Crafter environment,
enabling 1 000× faster simulation. The Craftax-Classic-Symbolic-v1
target is the symbol-observation, fast-tile-grow variant we use in this
work; it preserves the 22-achievement Hafner score formula and is a fair
zero-training comparison target.

To our knowledge **no zero-training planner has previously reached
human-expert score on Crafter or Craftax-Classic**. Our exp17 result
(50.95 %, 30 seeds) is the first such report.

## Reward shaping and intrinsic motivation

Reward shaping has a long history in RL. The seminal result of
\citet{ng1999shaping} establishes that *potential-based* shaping
preserves the optimal policy under any MDP. Our two-component shaping
scheme is *not* potential-based: the dense inv-tier $R_{\text{inv}}$
contributes per-state value of held resources (potential-like) but the
sparse $R_{\text{ach}}$ fires *once* per achievement during a rollout
(non-potential, non-Markovian on $s_0$ — see Method).

Critically, our shaping is applied **inside the FMC walker rollout**, not
to the environment reward or the policy gradient. The aim is to bias the
relativize-cloning dynamics toward chains that complete subgoals, not to
modify the value function the optimal policy would converge to. The
distinction places our work outside the formal Ng et al. invariance
result and inside an empirical regime where shaping affects which
trajectories the planner *spends compute exploring*, not what it ultimately
prefers to execute.

Intrinsic-motivation methods such as RND \citep{burda2019rnd},
ICM \citep{pathak2017curiosity}, NGU \citep{badia2020ngu} provide a
neighbouring approach: they augment reward with novelty / prediction-error
bonuses *during training*. By contrast, FMC has no training phase; our
"intrinsic motivation" is the achievement-fire bonus computed at decision
time relative to the current planning root.

## Active inference and the free-energy view

The relativize-cloning dynamic of FMC has been argued to instantiate a
free-energy minimisation \citep{friston2017active,friston2022book}: each
walker's $\widehat{r}$ is interpretable as a posterior over goal-reaching
trajectories, and cloning resembles importance-resampling from a generative
model. Our companion deep-dives in `work/02_deep_dives/` work this link
out in detail; the present paper does not depend on the active-inference
framing but is consistent with it.

## What this paper adds

The contribution is **empirical and theoretical-mechanistic**:

1. **First systematic compositional shaping scheme for FMC on a hard
   chain-completion benchmark**, with quantitative bounds on amplification
   multipliers (Section IV).
2. **Conjecture D** — formal statement of a chain-tier compounding
   amplification law (Section III), with empirical instantiation across
   23 experiments and four falsification thresholds.
3. **Lemma D.1** — sketched proof of compounding monotonicity under
   `relativize` regime separation (Section III.4 / Appendix A).
4. **A zero-training agent that matches human-expert score on
   Craftax-Classic** (Section V), achieved on a single CPU laptop.

The closest precedent in spirit is the original FMC Atari result
\citep{hernandez2018solving}, but the chain-structured shaping recipe
and its monotonicity analysis are new.
