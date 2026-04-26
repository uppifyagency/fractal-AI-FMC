# 07 — Fragile-RL: Stratificazione del libro

**Scopo**: vista a livelli del documento "Fragile Mechanics" (10 parti, ~50 capitoli).

```mermaid
flowchart TB
    subgraph PartI["Part I — Foundations"]
        f1["POMDP framework"]:::found
        f2["Bounded rationality"]:::found
        f3["Channel capacity C"]:::found
    end

    subgraph PartII["Part II — Sieve"]
        s1["60 diagnostic nodes"]:::sieve
        s2["Barrier methods"]:::sieve
    end

    subgraph PartIII["Part III — Architecture"]
        a1["VQ-VAE shutter"]:::arch
        a2["K, z_n, z_tex split"]:::arch
    end

    subgraph PartIV["Part IV — Control"]
        c1["Belief dynamics"]:::ctrl
        c2["Coupling windows"]:::ctrl
    end

    subgraph PartV["Part V — Geometry"]
        g1["Capacity-constrained metric"]:::geo
        g2["WFR geometry"]:::geo
        g3["Holographic interface"]:::geo
        g4["Geodesic jump dynamics"]:::geo
    end

    subgraph PartVI["Part VI — Fields"]
        fi1["Reward field V"]:::field
        fi2["Helmholtz/Hodge geometry"]:::field
        fi3["Information bound (area law)"]:::field
    end

    subgraph PartVII["Part VII — Cognition"]
        co1["Universal Governor"]:::cog
        co2["Lorentzian memory"]:::cog
        co3["Topological fission"]:::cog
        co4["Computational metabolism"]:::cog
    end

    subgraph PartVIII["Part VIII — Multi-agent / Gauge"]
        ga1["Standard Model of Cognition"]:::gauge
        ga2["Game tensor"]:::gauge
        ga3["Parameter sieve"]:::gauge
        ga4["Covariant cross-attention"]:::gauge
    end

    subgraph PartIX["Part IX — Economics"]
        ec1["Proof of Useful Work"]:::econ
    end

    subgraph PartX["Part X — Appendices"]
        ap1["Derivations"]:::appx
        ap2["Parameter tables"]:::appx
        ap3["FAQ + formal proofs"]:::appx
    end

    PartI --> PartII
    PartI --> PartIII
    PartIII --> PartIV
    PartIV --> PartV
    PartV --> PartVI
    PartVI --> PartVII
    PartVII --> PartVIII
    PartVIII --> PartIX
    PartII --> PartVII

    classDef found fill:#cce5ff,color:#000
    classDef sieve fill:#fff3cd,color:#000
    classDef arch fill:#d4edda,color:#000
    classDef ctrl fill:#d4edda,color:#000
    classDef geo fill:#f8d7da,color:#000
    classDef field fill:#f5c6cb,color:#000
    classDef cog fill:#d1ecf1,color:#000
    classDef gauge fill:#bee5eb,color:#000
    classDef econ fill:#e2e3e5,color:#000
    classDef appx fill:#f8f9fa,color:#000
```

## Mappa parte → contributo

| Parte | Contributo nuovo | Cosa ripackagea |
|---|---|---|
| I | Definizione operativa di "bounded rationality controller" | POMDP, Friston AIF |
| II | The Sieve come sistema di safety-by-construction | Lagrangian methods, barrier funcs |
| III | Latent decomposition `(K, z_n, z_tex)` esplicita | VQ-VAE, β-VAE, InfoNCE |
| IV | Coupling windows + belief on hybrid manifold | Particle filters, EKF/UKF |
| V | **Capacity-constrained metric law** + WFR + holographic | Optimal transport (WFR), info geometry, gauge theory |
| VI | Reward come PDE source (Helmholtz/screened Poisson) | Hodge decomposition, electromagnetic analogy |
| VII | Lorentzian memory + topological fission + thermodynamic hysteresis | Self-attention, ontology learning, Landauer bound |
| VIII | **Standard Model of Cognition** = SU(N_f) × SU(2) × U(1) | Yang-Mills, Standard Model of physics |
| IX | Proof of Useful Work (PoUW) per coordinazione multi-agent | Bitcoin PoW, mechanism design |
| X | Reference + proof formali | n/a |

## Cuore concettuale

Il framework dichiara di essere **una "cosmologia" dell'agente cognitivo**: parte dai principi minimi (capacità informativa finita) e deduce la struttura dell'agente come **conseguenza geometrica/gauge-teorica**.

> *Standard RL appears as a degenerate limit of the Fragile Agent when geometry is flattened (G → I), capacity is unbounded (|K| → ∞), and the Sieve is disabled (Ξ_crit → ∞).* — `intro_agent.md`

Ovvero: tutto RL classico è un caso limite "piatto" di Fragile Mechanics.
