# 04 — FMC Sequence Diagram (data flow di una singola decisione)

**Scopo**: mostrare temporalmente ciò che accade durante una decisione FMC: `state → action`.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Swarm
    participant Walkers as Walker Pool (N)
    participant Sim as Simulator
    participant Score as VirtualRewardCalc
    participant Cloner

    User->>Swarm: decide(state x₀)
    Swarm->>Walkers: init: copia x₀ × N
    Swarm->>Walkers: assegna initial_decisions casuali

    loop M tick (t = 1 .. M)
        Swarm->>Walkers: perturba degrees_of_freedom
        Walkers->>Sim: state, action, dt
        Sim-->>Walkers: next_state, reward
        Walkers->>Score: rewards + observations
        Score->>Score: relativize(R), relativize(Dist)
        Score-->>Walkers: VR_i = R_i · Dist_i
        Walkers->>Cloner: VR vector + random partners
        Cloner->>Cloner: P(clone) = (VR_k - VR_i) / VR_i
        Cloner-->>Walkers: copia stati + init_dec dei vincenti
    end

    Swarm->>Walkers: read init_decisions dei vivi
    Swarm->>Swarm: bincount(init_decisions)
    Swarm-->>User: action = argmax
```

## Note temporali

- Il loop interno **M tick** è il "cuore di pianificazione" — dura `M = τ/dt` iterazioni.
- A `t=1` ogni walker usa `initial_decision` (decisione di prova).
- A `t>1` ogni walker perturba liberamente (esplorazione del cono).
- Il cloning **non è sincronizzato per default**, ma in pseudocodice del paper lo è (per chiarezza).

## Sample budget

In ogni tick, ogni walker fa una `Simulator.step()`. Quindi:

```
total_samples ≈ N × M
```

Per Atari, paper usa `N=30, M=15 → 450 sample/decisione`. Confronto con MCTS UCT: 150 000 sample/decisione (~333× di più).
