# 02 — FMC Container Diagram (C4 Level 2)

**Scopo**: zoom dentro l'agente FMC, mostrare i "container" software (moduli/processi).

```mermaid
flowchart TB
    subgraph FMCAgent["Agente FMC"]
        config[/"Config<br/>τ, N, M, α, β"/]:::cfg

        swarm["Swarm Manager<br/>(orchestra walker)"]:::core
        walkers["Walker Pool<br/>(N copie dello stato)"]:::core
        scoring["Virtual Reward Calculator<br/>VR = R · Dist^β"]:::core
        cloner["Cloner<br/>(propaga walker promettenti)"]:::core
        decider["Decider<br/>(argmax sui init_decisions)"]:::core

        sim_iface["Simulator Interface<br/>(step, reset, clone_state)"]:::iface
        reward_iface["Reward Interface<br/>(R: state → ℝ⁺)"]:::iface
        dist_iface["Distance Metric<br/>(L2, custom)"]:::iface
    end

    sim[("Simulator<br/>(esterno)")]:::ext
    user([Utente])

    user --> config
    config --> swarm

    swarm --> walkers
    walkers --> sim_iface
    sim_iface --> sim
    sim --> sim_iface
    sim_iface --> walkers

    walkers --> reward_iface
    walkers --> dist_iface
    reward_iface --> scoring
    dist_iface --> scoring
    scoring --> cloner
    cloner --> walkers

    walkers --> decider
    decider --> action[/"Action a*"/]:::out

    classDef core fill:#d4edda,stroke:#155724,color:#000
    classDef iface fill:#f8d7da,stroke:#721c24,color:#000
    classDef cfg fill:#fff3cd,stroke:#856404,color:#000
    classDef ext fill:#e2e3e5,stroke:#383d41,color:#000
    classDef out fill:#cce5ff,stroke:#004085,color:#000
```

## Container per ruolo

| Container | Cosa fa | File rappresentativo (FractalAI_old) |
|---|---|---|
| Swarm Manager | Orchestra il loop scanning → cloning → decision | [`fractalai/swarm.py:174-617`](../../repos/FractalAI_old/fractalai/swarm.py#L174) |
| Walker Pool | Tiene gli N walker in memoria | `Swarm.observations`, `Swarm.walkers_id` |
| Virtual Reward Calc | Calcola `VR = R · Dist^β` | `Swarm.virtual_reward()` linee 469-480 |
| Cloner | Probabilità + applicazione del clone | `Swarm.clone_condition()`, `Swarm.perform_clone()` |
| Decider | Sceglie l'azione finale | `FractalMC.weight_actions()` linee 94-107 |
| Simulator Interface | Astrazione su gym/MuJoCo/custom | `fractalai/environment.py` |
| Reward Interface | Permette `custom_reward` callable | `Swarm.__init__(custom_reward=...)` |
| Distance Metric | Default L2 sulla osservazione | `Swarm.evaluate_distance()` linee 451-462 |

## Boundaries (Interface)

I tre **interface in rosso** sono il punto di estensione dell'algoritmo: cambiando metrica di distanza, reward function, o simulator, lo stesso FMC si applica a domini totalmente diversi (Atari, robotica, finanza, simulazione fisica).
