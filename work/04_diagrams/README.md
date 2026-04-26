# 04 — Diagrammi C4 / Mermaid

**Goal verificabile**: tutti i diagrammi Mermaid in questa cartella sono renderizzabili con `mmdc` o GitHub native rendering, e ogni diagramma ha al massimo 15 nodi (vincolo di leggibilità).

## Indice

| File | Cosa diagramma | Livello C4 |
|---|---|---|
| [`01_fmc_context.md`](01_fmc_context.md) | Contesto: chi usa FMC, con quali sistemi parla | C4 — System Context |
| [`02_fmc_container.md`](02_fmc_container.md) | I "contenitori" software di un agente FMC | C4 — Container |
| [`03_fmc_components.md`](03_fmc_components.md) | I componenti interni (Swarm, Walker, ecc.) | C4 — Component |
| [`04_fmc_dataflow.md`](04_fmc_dataflow.md) | Flusso dati di una decisione (sequence diagram) | n/a — sequence |
| [`05_fmc_state_machine.md`](05_fmc_state_machine.md) | Macchina a stati di un walker (born → clone/die) | n/a — state |
| [`06_fragile_rl_architecture.md`](06_fragile_rl_architecture.md) | Architettura cognitiva di Fragile Mechanics | C4 — Container |
| [`07_fragile_rl_layers.md`](07_fragile_rl_layers.md) | Stratificazione: Sieve, Geometry, Gauge fields, Memory | C4 — Component |

## Convenzioni di rendering

- Tutti i diagrammi sono in **Mermaid 10+** (compatibile con GitHub).
- Per generare PNG/SVG offline:
  ```bash
  npx @mermaid-js/mermaid-cli -i 01_fmc_context.md -o 01_fmc_context.png
  ```
- Color palette consistente:
  - **Walker / particella**: blu chiaro (`#cce5ff`)
  - **Componente algoritmico**: verde (`#d4edda`)
  - **Sistema esterno**: grigio (`#e2e3e5`)
  - **Storage / data**: giallo (`#fff3cd`)
  - **Boundary / interface**: rosso chiaro (`#f8d7da`)
