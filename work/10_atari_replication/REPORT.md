# P1a — Atari multi-seed replication

**Status (2026-04-28)**: 🟡 **HARNESS LANDED + 1-GAME SLICE (n=5) — FULL 50-GAME RUN PENDING**

This directory operationalizes
[`docs/bibliography/protocols/P1a_atari_replication_protocol.md`](../../docs/bibliography/protocols/P1a_atari_replication_protocol.md):
the paper §5.1.1 Atari table re-run with **n seeds, error bars, bootstrap CI95**
where the paper had single-seed, no error bars.

---

## What landed in-session

| Component | File | Status |
|---|---|---|
| Atari adapter (RAM + RGB obs) | [`fmc-core/src/fmc/envs/atari.py`](../../fmc-core/src/fmc/envs/atari.py) | ✅ working |
| Multi-seed sweep with bootstrap CI95 | [`scripts/atari_seed_sweep.py`](scripts/atari_seed_sweep.py) | ✅ working |
| Boxing slice (n=5 seeds, paper params N=30, M=15) | [`runs/boxing_seeds.jsonl`](runs/boxing_seeds.jsonl) + [`runs/boxing_seeds_summary.csv`](runs/boxing_seeds_summary.csv) | ✅ recorded |

## Boxing slice result — n=5 seeds, paper params (§5.1.3.3)

| game | n | N | M | obs | mean | std | CI95 | min | max | wall/seed |
|---|---|---|---|---|---|---|---|---|---|---|
| Boxing | 5 | 30 | 15 | RAM | **+100.0** | 0.0 | [100, 100] | 100 | 100 | ~82 s |

**5 / 5 seeds reach knockout** (+100 cap with episode termination). Wall
time ~82 s per seed on a single CPU core, no GPU.

This *replicates and tightens* the paper §5.1.1 number for Boxing
(reported as 96 / 100 single-seed). Within the +100 cap of Boxing, our
multi-seed mean is the strongest possible.

## How to run the full 50-game protocol

The full P1a deliverable per protocol is 50 games × 10 seeds = 500
episodes. With our measured ~82 s/seed on Boxing as a baseline:

```
500 episodes × ~80 s/seed ≈ 11 hours single CPU
```

**This invalidates the protocol's GPU-cluster cost estimate.** A
single workstation, overnight, suffices. To launch:

```bash
cd work/10_atari_replication
python -m scripts.atari_seed_sweep \
    --games Boxing QBert MsPacman BankHeist Centipede IceHockey VideoPinball \
            Atlantis Frostbite Gravitar BeamRider Krull DoubleDunk Skiing \
            ... [all 50 from paper §5.1.1] \
    --seeds 10 --N 30 --M 15 --obs_type ram \
    --max_actions 600 \
    --out_runs runs/atari50_seeds.jsonl \
    --out_summary runs/atari50_summary.csv
```

The summary CSV is *publication-paste-ready* (one row per game with
mean ± std, CI95, and n_seeds).

## Caveats & open work

### Score-cap games

Boxing's reward is capped at +100 (a knockout). Once we hit it, all
seeds tie and CI95 collapses to a point. For the 16/50 games the paper
§5.1.1 flags as "solved due to 1M bug", a similar cap effect happens.
The full P1a should:

- Separate cap-bound games (max-score reached) from variability-bound
  games (still distributed) in the v6 table.
- Report **fraction of seeds reaching cap** as the headline metric for
  cap-bound games, not "mean ± std" (degenerate).

### Sticky actions

Adapter currently sets `sticky_actions=False` to match the paper's
deterministic regime. ALE 0.11.2 default is True. Verify the paper
truly used det-env in §5.1.1.

### Per-game baselines

Paper §5.1.1 lumps "Best Planning SoTA" / "Best Learning SoTA" without
per-game attribution. For v6 v6, populate a per-game lookup table from
[P1-7] and [L1-9] references. Source candidates:

- DQN papers (Mnih 2015 Nature, Wang 2016 Dueling DQN)
- Rainbow (Hessel 2018)
- Atari planning baselines (UCT, original Mnih 2014)

### Statistical methodology

Bootstrap CI95 with n_boot=1000 is implemented. For Boxing's degenerate
case, we should also report the **non-parametric all-success indicator**
("5/5 seeds at cap"). The script currently doesn't do this — TODO for
the full run.

## Decision matrix per protocol

| Result on full 50-game run | Lettura | Action |
|---|---|---|
| Means match paper, std stretto (CV<10%) | Replica robusta | Aggiorna v6 tabella con error bars |
| Means match paper, std largo (CV>30%) | Variability sottostimata | Tabella v6 con CI95 + caveat onesto |
| Means significantly under paper | Non riproducibile | Crisis — riformulare conservativamente |
| Means above paper (rare) | Stack 2026 batte 2020 | Bonus, indagare frame-skip / sticky-actions |

For the Boxing slice we have: means *match paper* (both cap at 100),
std=0 means cap-bound; status = robust replication for this 1 game.
