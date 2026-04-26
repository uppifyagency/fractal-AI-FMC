# 03 — FMC Components Diagram (C4 Level 3)

**Scopo**: zoom dentro lo Swarm Manager — i componenti che fanno il lavoro algoritmico.

```mermaid
flowchart LR
    subgraph SwarmManager["Swarm Manager"]
        init["init_swarm()<br/>copia x₀ in N walker"]:::comp
        step["step_walkers()<br/>perturba + simula dt"]:::comp
        score["virtual_reward()<br/>R · Dist^β"]:::comp
        compas["get_clone_compas()<br/>scelta partner"]:::comp
        clone_cond["clone_condition()<br/>(VR_k - VR_i) / VR_i"]:::comp
        clone_do["perform_clone()<br/>copia stato + init_dec"]:::comp
        stop["stop_condition()<br/>limite sample / death"]:::comp

        init --> step
        step --> score
        score --> compas
        compas --> clone_cond
        clone_cond --> clone_do
        clone_do --> step
        step --> stop
        stop -- "loop" --> step
        stop -- "fine" --> decide["weight_actions()<br/>argmax(init_dec)"]:::out
    end

    classDef comp fill:#d4edda,stroke:#155724,color:#000
    classDef out fill:#cce5ff,stroke:#004085,stroke-width:2px,color:#000
```

## Pseudocodice del loop

```python
def run_swarm():
    init_swarm()                          # x₀ → N walker
    while not stop_condition():           # default: budget di sample
        clone_condition()                 # 1) calcola VR e probabilità clone
        step_walkers()                    # 2) perturba ed evolvi simulator
        clone()                           # 3) applica i clone effettivi
    return weight_actions()               # decisione = bincount(init_decisions).argmax()
```

## Stato del singolo walker

```mermaid
classDiagram
    class Walker {
        +int id
        +State env_state
        +Action initial_decision
        +float reward
        +float virtual_reward
        +float distance
        +bool is_alive
        +int parent_id
    }

    class Swarm {
        -np.ndarray walkers_id  
        -np.ndarray rewards
        -np.ndarray distances
        -np.ndarray virtual_rewards
        -np.ndarray init_ids
        -DataStorage data
        +clone_condition()
        +perform_clone()
        +step_walkers()
        +virtual_reward() float
        +run_swarm()
    }

    Swarm "1" *-- "N" Walker : contiene
```

## Trasformazione `relativize` (essenza matematica del scoring)

Senza relativize la moltiplicazione `R · Dist` esplode quando uno dei due è grande. Con relativize entrambi sono **z-score normalizzati e poi compressi**:

```mermaid
flowchart LR
    raw["raw value v"] --> norm["v_N = (v - μ) / σ"]
    norm --> branch{"v_N ≤ 0?"}
    branch -- "sì (negativo)" --> exp["v' = exp(v_N)"]
    branch -- "no (positivo)" --> log["v' = 1 + ln(1 + v_N)"]
    exp --> out["v' ∈ ℝ⁺"]
    log --> out
```

Proprietà:
- `v' > 0` sempre
- Comprime outlier positivi (log)
- Espande outlier negativi (exp)
- Preserva ordinamento

Implementato in [`relativize_vector` (FractalAI_old)](../../repos/FractalAI_old/fractalai/swarm.py#L16) e in [`relativize` (fragile)](../../repos/fragile/src/fragile/fractalai.py#L27).
