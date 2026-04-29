# P3 — RAM vs IMG ablation

**Status (2026-04-28)**: 🟡 **HARNESS LANDED + 1-CELL SLICE — FULL 1280-CELL SWEEP PENDING**

This directory operationalizes
[`docs/bibliography/protocols/P3_ram_vs_img_ablation_protocol.md`](../../docs/bibliography/protocols/P3_ram_vs_img_ablation_protocol.md):
the §5.1.3.3 RAM-vs-IMG claim ("RAM beats IMG by 161.47% on average") tested
across $(N, M)$ grid to see whether the advantage is robust or an artifact
of the specific $\{N=30, M=15\}$ paper setting.

---

## What landed in-session

| Component | File | Status |
|---|---|---|
| Atari adapter (RAM + RGB obs, same code path) | [`fmc-core/src/fmc/envs/atari.py`](../../fmc-core/src/fmc/envs/atari.py) | ✅ working |
| Parametric sweep driver | [`scripts/ram_img_sweep.py`](scripts/ram_img_sweep.py) | ✅ working |
| Aggregator (CSV + RAM/IMG ratio per cell) | [`scripts/aggregate.py`](scripts/aggregate.py) | ✅ working |
| Boxing micro-cell (RAM vs RGB, n=2, N=30, M=15) | [`runs/boxing_ram_vs_rgb.jsonl`](runs/boxing_ram_vs_rgb.jsonl) + [`runs/boxing_summary.csv`](runs/boxing_summary.csv) | ✅ recorded |

## Boxing slice result (paper params, N=30, M=15, n=2)

| game | N | M | RAM mean | RGB mean | Δ (RAM − RGB) | RAM actions-to-win | RGB actions-to-win |
|---|---|---|---|---|---|---|---|
| Boxing | 30 | 15 | +100.0 | +100.0 | **0.0** | 119 (mean) | 104 (mean) |

**On Boxing both obs types saturate the +100 cap — the paper's "RAM > IMG"
claim is invisible at this game.** Both reach knockout, but RGB reaches
it slightly *faster* in actions (n=2 too small to claim significance, but
worth flagging — opposite direction from paper).

This is a **known limitation of cap-bound games**: any algorithm that
solves them produces the same headline number, and only secondary metrics
(actions-to-win, reward-rate) can differentiate. The full P3 needs
non-cap-bound games to actually test the claim.

## How to run the full protocol

Per protocol §"Sweep parametrico":

```
games = [Atlantis, BankHeist, Boxing, Centipede, IceHockey,
         MsPacman, QBert, VideoPinball]
N     = [30, 60, 120, 240]
M     = [10, 15, 30, 60]
seeds = 1..5
obs   = RAM, RGB
```

= 1280 cells. Wall-time budget at our measured ~80 s / cell:

```
1280 × ~80 s ≈ 28 hours single CPU
```

Same conclusion as P0/P1a: the protocol's "GPU compute" estimate was
overprovisioned. **One workstation, ~24-30 hours, runs the full P3.**

```bash
cd work/11_ram_vs_img_ablation
python -m scripts.ram_img_sweep \
    --games Atlantis BankHeist Boxing Centipede IceHockey \
            MsPacman QBert VideoPinball \
    --N 30 60 120 240 --M 10 15 30 60 --seeds 5 \
    --max_actions 600 --out runs/p3_full.jsonl

python -m scripts.aggregate --runs runs/p3_full.jsonl --out runs/p3_summary.csv
```

## Decision matrix per protocol

After full sweep, read RAM/RGB ratio across (N, M):

| Pattern | Lettura | Action |
|---|---|---|
| RAM/RGB ratio stable ~1.6 across all (N, M) | Vantaggio robusto | Mantieni claim, generalizza |
| RAM/RGB > 1 ma decresce con N | Artefatto del low-budget regime | Specifica claim |
| Ratio collassa a 1 al crescere di N | Solo settings poveri vedono RAM > IMG | Riformula: matters solo a low budget |
| Pattern game-dependent | Selection bias §5.1.3.3 | Documenta game-dependence |
| Ratio < 1 (RGB beats RAM) | Paper claim falsificato | Riformula completamente |

The Boxing slice tentatively suggests the **last row** (RGB ≥ RAM on
Boxing in actions-to-win), but n=2 is way under-powered. Full sweep
needed.

## Caveats

- **n=5 is sub-publication.** For a real paper claim, n=10 with bootstrap
  CI95 is needed (use `atari_seed_sweep.py` from P1a for that).
- **RAM is bit-exact, RGB has aliasing/flicker.** This *explains* any
  RAM advantage but doesn't *quantify* it across $(N, M)$ — that's what
  the ablation is for.
- **L2 distance on flattened RGB (210×160×3 = 100 800 dims) may be
  pathological.** The cosine-distance variant or perceptual distance
  could change the result. Worth testing as a sub-ablation.
