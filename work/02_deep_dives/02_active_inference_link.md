# Deep Dive 02 — Fractal AI ↔ Active Inference (Friston)

> **Stato**: outline. Da espandere con dimostrazioni formali e citazioni puntuali.

## Tesi

> *Fractal AI è un'implementazione operativa del Free Energy Principle di Friston, dove il free energy expected è approssimato via Monte Carlo con uno sciame di walker.*

## Mappa concettuale (da espandere)

| Active Inference (Friston) | Fractal AI |
|---|---|
| Generative model $p(o, s)$ | Simulator + initial state |
| Posterior beliefs $q(s)$ | Walker distribution su $X_H(x_0, t)$ |
| Variational free energy $F[q]$ | Sub-optimality coefficient (paper §3) |
| Expected free energy $G[\pi]$ | $\sum_t D_H(P_R, P_S)$ |
| Epistemic value | termine di distanza $d^\beta$ |
| Pragmatic value | termine di reward $R^\alpha$ |
| Active inference loop | FMC loop scanning → cloning → decision |
| Markov blanket | Cone causale $X(x_0, \tau)$ |

## Outline delle sezioni da scrivere

1. **Setup di Friston**: definizione di FEP, generative model, recognition density
2. **Mappatura formale**: come l'EFE si scompone in $G = -\mathbb{E}_q[\log p(o)] + D_{KL}(q \| p)$
3. **Equivalenza dei termini**: epistemic ≡ exploration, pragmatic ≡ exploitation
4. **Differenze**: Friston usa gradient descent variazionale; FMC usa SMC. Stessa quantità minimizzata, diversi solver.
5. **Vantaggi di FMC su AIF**: non serve gradiente del modello, lavora su simulator black-box
6. **Vantaggi di AIF su FMC**: ha una teoria di apprendimento (update del generative model) che FMC non ha
7. **Sintesi**: FMC come "AIF inference engine + planning" senza la parte di learning

## Riferimenti chiave da espandere

- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nat. Rev. Neurosci. 11(2): 127-138.
- Friston, K., FitzGerald, T., Rigoli, F., et al. (2017). *Active inference: a process theory*. Neural Computation 29(1): 1-49.
- Da Costa, L., Parr, T., Sajid, N., et al. (2020). *Active inference on discrete state-spaces: a synthesis*. arXiv:2001.07203.
- Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). *Active inference: demystified and compared*. Neural Computation 33(3): 674-712.

---

*Da espandere a 600-1200 righe con derivazioni formali. Priorità: media (vedi index in [`README.md`](README.md)).*
