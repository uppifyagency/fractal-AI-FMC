# Fractal Coding Loop — Claude Code plugin

> *Fractal Monte Carlo planning per decisioni di coding. Spawn N walker sub-agent in parallelo (worktree-isolated), compute reward composto, scegli vincitore.*

[![status: PoC](https://img.shields.io/badge/status-PoC-orange)]()
[![based-on: Hernández-Cerezo 2020](https://img.shields.io/badge/based--on-Fractal%20AI%20arXiv%3A1803.05049-blue)]()

## Cosa fa

Quando hai una decisione di coding ambigua (es. "refactor X", "aggiungi feature Y", "scegli tra approccio A e B"), il plugin:

1. Genera **3 approcci candidati** distinti
2. Lancia **3 walker sub-agent in parallelo**, ognuno in un git worktree isolato, ognuno implementa un approccio
3. Calcola **reward composto moltiplicativo** per ogni walker (test pass + lint + diff size + goal-alignment)
4. Mostra una **tabella di confronto** con confidence
5. Ti chiede quale mergeare
6. **Salva la memoria** (Wigner-weighted) per recall futuri

Algoritmo: il [Fractal Monte Carlo](https://arxiv.org/abs/1803.05049) di Hernández-Cerezo & Duran-Ballester (2020), specificato per il dominio coding.

## Comandi

| Slash command | Cosa fa |
|---|---|
| `/fractal-decide [task]` | Lancia uno sciame N=3 e decide. **Il comando principale.** |
| `/fractal-recall [query]` | Mostra le memorie più rilevanti (Wigner-weighted) |
| `/fractal-memory-show` | Stato completo della memory bank |

## Sub-agent

| Agent | Ruolo |
|---|---|
| `fractal-walker` | Esegue UNO degli approcci, ritorna JSON strutturato |
| `fractal-judge` | Valuta goal-alignment di un walker, ritorna scalar [0,1] |

## Installazione

Vedi [`INSTALL.md`](INSTALL.md). TL;DR:

```bash
# 1. Symlink the plugin into Claude Code's plugin directory
mkdir -p ~/.claude/plugins
ln -s "$(pwd)/plugin/fractal-coding-loop" ~/.claude/plugins/fractal-coding-loop

# 2. Restart Claude Code
# 3. In any git repo: /fractal-decide "your task"
```

## Esempio d'uso

```
> /fractal-decide refactor the auth middleware to use JWT instead of session cookies

[Claude main agent generates 3 approaches:]
  1) in-place: replace session calls with JWT middleware, minimal changes
  2) extract-module: pull auth into separate package, update imports, add JWT
  3) test-first: write JWT integration tests, then implement

[Spawns 3 walkers in parallel, each in a worktree]

[After ~5 min, walkers return:]

| # | Approach | Tests | Lint | Diff | Reward | Conf |
|---|---|---|---|---|---|---|
| 1 | in-place | 12/12 ✓ | 0 | +84/-39 | 4.21 | 47% |
| 2 | extract-module | 12/12 ✓ | 0 | +156/-72 | 3.78 | 32% |
| 3 | test-first | 13/13 ✓ | 1 | +212/-39 | 5.34 | 67% |

Walker 3 wins (67% confidence, +21% reward over runner-up).
Reasons: same correctness, plus added 1 new integration test.

> Quale mergeo? (1/2/3/cancel)
```

## Architettura tecnica

```
fractal-coding-loop/
├── .claude-plugin/plugin.json    Plugin manifest
├── commands/
│   ├── fractal-decide.md         Main command (orchestrator instructions)
│   ├── fractal-recall.md
│   └── fractal-memory-show.md
├── agents/
│   ├── fractal-walker.md         Walker protocol
│   └── fractal-judge.md          Goal-alignment evaluator
├── scripts/
│   ├── fractal_reward.py         Multiplicative reward + virtual reward
│   └── fractal_memory.py         Wigner-weighted memory bank
└── tests/
    └── e2e_test.sh               Integration test
```

### Reward formula (paper §2.2.2 + Pareto Frontiers blog 2016)

```
R = R_alive · R_tests · (1 + R_lint) · (1 + R_diff) · (1 + R_goal)

R_alive  ∈ {0, 1}    HARD: zero se compile_ok=false
R_tests  ∈ [0, 1]    test passed/total
R_lint   ∈ [0, 1]    1 / (1 + log(1 + warnings))
R_diff   ∈ [0, 1]    max(0, 1 - lines/200)
R_goal   ∈ [0, 1]    da fractal-judge sub-agent
```

Poi `relativize()` cross-walker e `VR = R^α · Dist^β` (paper §4.2).

### Memory weighting (Slide doc 2020)

```
weight(memory) = Wigner(loss/avg_loss) / (1 + log(1 + visits))
Wigner(x) = (π/2) · x · exp(-π/4 · x²)
```

Picca a `x ≈ 1` (memoria a difficoltà media): **automatic curriculum learning**.

## Limiti del PoC stasera

- **N=3 walker hardcoded**: nessun setting dinamico
- **M=1 tick** (single roll-out per walker): no cloning multi-tick
- **Reward fissa**: nessun self-improving reward function (Phase 3)
- **Single-task scope**: niente octopus hierarchy (Phase 4)

Roadmap: vedi [`docs/vision/fractal_coding_loop.md`](../../docs/vision/fractal_coding_loop.md) §V7.

## Licenza

MIT. Concetti algoritmici da Hernández-Cerezo & Duran-Ballester (2020), arXiv:1803.05049.
