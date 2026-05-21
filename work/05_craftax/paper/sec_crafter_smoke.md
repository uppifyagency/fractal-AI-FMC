# Cross-benchmark preliminary on Crafter-original (Section 6.1 / Appendix C)

> Preliminary 3-seed smoke test on Crafter-original (Hafner 2021) to
> probe whether Conjecture D extends beyond Craftax-Classic. Result:
> directionally consistent but **far below significance** at the
> low-compute config feasible in a single CPU session. Full study
> requires N=512, M=40 — the same configuration that produced 50.6%
> on Craftax — and is left to companion work.

## Setup

We installed `crafter==1.8.3` and ported FMC to use deepcopy-based
walker branching (since Crafter-original does not expose `set_state`).
Per-step env cost ≈ 0.6 ms, deepcopy ≈ 1.2 ms. At N = 32 walkers,
M = 12 rollout depth, max_steps = 200, episodes complete in ~70 s on
a single Apple M1 Pro CPU (vs ~110 s for the full N = 512, M = 40
config on Craftax-Classic).

The reduced N and M were chosen to keep the smoke test under
~10 minutes wall-clock per seed. They are *not* the configuration
that produced the headline 50.6 % on Craftax — they are roughly
**16 × smaller in N and 3.3 × smaller in M**, totaling ~50 × less
compute per decision.

## Result

| Method | Crafter (Crafter-original) | n seeds | mean_ach | wall/episode |
|---|---:|---:|---:|---:|
| FMC v4 (no shaping) | 3.62 % | 3 | 6.00 | 66 s |
| FMC exp17 (full shaping) | 3.77 % | 3 | 6.33 | 73 s |
| **Δ** | **+0.15 pp** | — | **+0.33** | — |

Both runs use the same seed bank (42, 43, 44) for paired comparison.
The +0.15 pp Δ is positive but well below any reasonable noise floor
at n = 3.

## What this preliminary result does and does not show

**It does** show:

- The FMC port to Crafter-original works mechanically. Crafter has
  `__deepcopy__`-able state, so branching is cheap (~1 ms).
- exp17 shaping does not break the algorithm on Crafter-original —
  episodes complete, achievements unlock, no crashes.
- The directional sign of the Δ is *positive* (exp17 > v4) on every
  one of the three seeds.

**It does NOT** show:

- That Conjecture D's quantitative law transfers to Crafter-original.
  +0.15 pp at n = 3 is statistically meaningless.
- That the same 50.6 % score can be reached on Crafter-original.
  Without scaling N to 512 and M to 40, the FMC selection dynamics
  cannot produce the same chain-completion behaviour we observed on
  Craftax-Classic.

## Why we expect a positive result with full compute

The Conjecture D mechanism (Lemma D.1) depends on three quantities:
$N$ (walker population), $M$ (rollout depth), and the bonus weight $w_j$.
Theorem T1 gives the regime-separation threshold

$$
w_{\min}(z^*, \sigma_{\text{base}}, p)
= \frac{\sigma_{\text{base}} z^*}{\sqrt{(1-p)(1 - p - z^{*2} p)}}.
$$

At $N = 32$, $\sigma_{\text{base}}$ is much higher (small-sample variance)
and the firing probability $p$ is much lower (only 32 walkers explore).
Both effects make $w_{\min}$ larger, requiring the fixed bonus weights
(e.g. $w_{\text{iron-pickaxe}} = 200$) to do more work per walker.
Theorem T2 also gives looser Hoeffding bounds at small $N$:

$$
\mathbb{P}(|N_{\text{clone}} - \mathbb{E}[N_{\text{clone}}]| \ge t)
\le 2 \exp(-2 t^2 / (N - 1))
$$

For $N = 32$ vs $N = 512$, the same 5 % tail bound corresponds to
$t/\mathbb{E}[N_{\text{clone}}] \approx 0.4$ vs $0.07$ — at small N the
firing trajectory dominance is statistically weaker by a factor of ~6.

## What the next agent should do

Run a full cross-benchmark replication at the **matched** FMC config
(N=512, M=40), 30 seeds:

```bash
cd work/05_crafter_original_port
JAX_PLATFORMS=cpu python fmc_crafter.py \
    --seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71 \
    --shaping v4 \
    --N 512 --M 40 --max_steps 500 \
    --out_json results_full_v4_30seed.json
# Then same for --shaping exp17
```

Estimated wall-clock: ~1–3 hours per seed on M1 Pro (Crafter-original
is single-threaded, no JAX vectorization). Total ~30–90 hours per
config, or ~2–6 days for the full v4-vs-exp17 paired comparison.

Modal cloud or a multi-core server would reduce this to ~6–18 hours.
The right test of Conjecture D's cross-benchmark generality requires
this scale; the present 3-seed N=32 result is not it.

## Honest framing for the paper

In the paper's main body (Section 6.1 or Appendix C), the cross-
benchmark result is reported as **preliminary** with the caveat that

- $n = 3$ seeds is descriptive, not inferential
- the FMC config is 50× smaller than the configuration that produced
  the 50.6 % headline on Craftax-Classic
- the +0.15 pp Δ is consistent with no-effect, with directionally
  positive sign

This is the **most honest** framing given the data we have. Upgrading
to a stronger claim requires the full-compute replication described
above — which is infrastructurally feasible but exceeds the budget
of a single-CPU overnight session.
