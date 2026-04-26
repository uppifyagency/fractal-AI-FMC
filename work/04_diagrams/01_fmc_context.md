# 01 — FMC System Context Diagram (C4 Level 1)

**Scopo**: mostrare chi (umano e sistema) interagisce con un agente FMC e con quali sistemi esterni parla.

```mermaid
flowchart TB
    user([Utente / Operatore])
    coach([Designer del task])

    fmc[/"Agente FMC<br/>(Fractal Monte Carlo)"/]:::core

    sim[("Simulator / Environment<br/>Atari, MuJoCo, Custom")]:::ext
    reward[("Reward function R(x)<br/>scalare ≥ 0")]:::ext
    sensor[("Sensori reali<br/>(opzionale, mondo fisico)")]:::ext
    actuator[("Attuatori<br/>(opzionale, mondo fisico)")]:::ext

    user -- "imposta τ, N, α" --> fmc
    coach -- "definisce R(x)" --> reward
    coach -- "configura modello dinamica" --> sim

    fmc -- "x_t, action a_t" --> sim
    sim -- "x_{t+dt}" --> fmc
    fmc -- "valuta R(x)" --> reward
    reward -- "scalar" --> fmc

    sensor -.real-world.-> fmc
    fmc -.real-world.-> actuator

    classDef core fill:#cce5ff,stroke:#004085,stroke-width:2px,color:#000
    classDef ext fill:#e2e3e5,stroke:#383d41,color:#000
```

## Chiave di lettura

- **Utente / Operatore**: chi esegue l'agente al runtime. Sceglie iperparametri (`τ`, `N`, balance `α/β`).
- **Designer del task**: chi progetta il problema. Definisce la *reward* e il *simulator*.
- **Simulator**: il cuore del forward-thinking. Per Atari è l'emulatore stesso; per il razzo, una fisica custom; per la robotica, MuJoCo/PyBullet.
- **Reward function**: deve rispettare R(alive) > 0, R(dead) = 0. Spesso composta moltiplicativamente.
- **Sensori/Attuatori**: presenti solo se l'agente è deployato nel mondo reale. Non necessari per benchmark RL.

## Differenza chiave vs RL classico

In RL tradizionale, l'agente **impara** una policy `π_θ(a|s)` da un dataset. Qui invece:

- Non c'è apprendimento
- L'agente **pianifica online** ad ogni step usando il simulator come "occhio sul futuro"
- Niente network, niente gradiente, niente training loop
