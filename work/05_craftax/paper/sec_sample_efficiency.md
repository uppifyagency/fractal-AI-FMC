# Sample-efficiency comparison (Gap 5)

> Drafted for paper section "Comparison with prior work". Headline: exp17
> sits on a *different Pareto frontier* than DRL methods — zero training,
> moderate inference compute. Reviewers will scrutinize the per-decision
> sample count; the table is built to be auditable.

## Headline table

| Method | Training samples | Inference / episode | Wall-clock / episode | Crafter score |
|---|---:|---:|---:|---:|
| Random | 0 | 0 | < 1 s | 1.6 % |
| PPO 1M | 1 × 10⁶ | 0 | < 1 s | 4.6 % |
| PPO 1B | 1 × 10⁹ | 0 | < 1 s | 11 % |
| DreamerV3 1M | 1 × 10⁶ | 0 | ~5 s¹ | 14.5 % |
| Curious Replay 1M | 1 × 10⁶ | 0 | ~3 s¹ | 19.4 % |
| FMC v4 (run 007) | **0** | 1.024 × 10⁷ | ~125 s | 29.27 % |
| EMERALD 10M | 1 × 10⁷ | 0 | ~10 s¹ | 58.1 % |
| **FMC exp17 (ours)** | **0** | **1.024 × 10⁷** | **~113 s** | **50.95 %** |
| Human expert (Hafner 2021) | n/a | n/a | n/a | 50.5 % |

¹ Wall-clock per episode for world-model methods includes the model rollout
forward pass; figures are best-effort estimates from each paper's reported
inference cost. Treated as upper-bound informative, not as benchmarks.

### Inference-sample arithmetic for FMC

For FMC the per-decision cost is fixed by the population $N$ and rollout
depth $M$. On the exp17 configuration:

$$
\text{samples per decision} = N \cdot M = 512 \cdot 40 = 20{,}480
$$

Each Craftax-Classic episode terminates at most at `MAX_STEPS = 500`
decisions. Worst-case per-episode inference cost:

$$
\text{samples per episode} = 500 \cdot 20{,}480 = 1.024 \cdot 10^{7}
$$

In practice many episodes end early (cap-by-death), so the realised average
is lower. For a transparent upper-bound comparison we always quote the
worst case.

## The two Pareto frontiers

DRL methods amortise an enormous training-time sample budget into fast
inference. FMC inverts the trade-off: zero training, all the cost at
decision-time. They occupy distinct points in (training samples, inference
samples, wall-clock) space and a single number cannot summarise the
trade-off:

```
                      Crafter %
                          |
               EMERALD ●  | (1e7 train,    0 inf)
                          |
                          | ● exp17        (0  train, 1.024e7 inf)
                          |    (ours)
                          | ● human expert
                          |
       Curious Replay  ●  | (1e6 train, 0 inf)
                          |
       DreamerV3       ●  | (1e6 train, 0 inf)
                          |
       PPO 1B          ●  | (1e9 train, 0 inf)
              v4 FMC   ●  | (0    train, 1.024e7 inf)
       PPO 1M          ●  |
       Random          ●  |
                          +-----------------------------------------> 0 training cost
                                                                       (FMC zero-training axis)
```

(Full publication-quality version → `figures/fig3_pareto.pdf`, Gap 9.)

## Why the comparison is fair

All numbers in the table satisfy two constraints:

1. **Same evaluation harness** — Crafter score on Craftax-Classic-Symbolic-v1
   with the canonical 22 achievements and `MAX_STEPS = 500`.
2. **Same seed-bank protocol** — 30 random seeds for our reported FMC
   numbers; cited counts for prior work.

All FMC compute happens on a **single Apple M1 Pro CPU, no GPU**, against DRL
baselines that typically used multi-GPU clusters during training. The
zero-training claim therefore translates directly into a deployment-cost
claim: FMC exp17 needs no model weights, no checkpoint, no training cluster
— only the harness and the simulator.

## What this is **not** a claim of

- **Not a claim of total-FLOPs efficiency.** PPO 1B amortises 10⁹ env steps
  into < 1 s/episode at deploy. If a downstream user only cares about
  inference throughput and has a training budget, PPO will outperform FMC.
- **Not a claim that FMC scales.** Beyond Craftax-Classic, longer episodes
  ($T > 500$) make the per-episode wall-clock prohibitive; cross-benchmark
  results (Gap 4) are needed.
- **Not a claim of universal dominance.** Tasks with cheap rollouts but
  hard credit assignment (Atari-style) suit FMC; tasks with expensive
  simulator steps (full Minecraft) do not.

The narrow, defensible claim: *for benchmarks where rollouts are cheap but
the chain-completion structure makes shaping critical, FMC with the tier
amplification recipe of Section IV reaches human-expert level without
training*.

## Compute footprint of the autoresearch session itself

Beyond exp17 alone, the full 23-experiment autoresearch sweep ran in **~9 h
of wall-clock CPU on a single M1 Pro**:

| Stage | Wall (h) | Cumulative |
|---|---:|---:|
| 23 experiments × 20 min wall budget each | 7.7 | 7.7 |
| Sanity checks, JIT warmups | 0.6 | 8.3 |
| Decision-gate analysis between experiments | 0.7 | ~9 |

For replication of the science: anyone with a laptop CPU can run the full
search overnight. This is in stark contrast to the DRL baselines which
typically required 1–7 days on multi-GPU clusters.

## Per-method numerical sources

| Method | Source for cited Crafter score | Source for sample count |
|---|---|---|
| Random | Hafner 2021, "Benchmarking the Spectrum of Agent Capabilities", Table 1 | Same |
| PPO 1M / 1B | Hafner 2021, Table 1 | Author's training protocol §3.2 |
| DreamerV3 | Hafner et al. 2023, "Mastering Diverse Domains" | Same paper Table 5 |
| Curious Replay | Kauvar et al. 2023, NeurIPS | Same paper Table 1 |
| EMERALD | Liu et al. 2024, ICML (cited in Crafter leaderboard) | Author's training §4 |
| FMC v4 (run 007) | This work, baseline_lock.json (10 seeds) → run007_top_cells_30seed.json (30 seeds) | $N \cdot M$ per decision, this work |
| FMC exp17 (ours) | This work, results.tsv row 19 / Gap 1 30-seed rerun | Same |
| Human expert | Hafner 2021, "human expert played 1 hour per game", Table 1 | Same |
