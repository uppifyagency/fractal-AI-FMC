# Plasma FMC — bibliografia verificata

> Ogni claim numerico in questo progetto deve essere tracciabile a una di queste fonti.
> Tutte le URL sono state verificate il 2026-04-26.

## A. Macchina di riferimento (TCV @ EPFL)

| Tag | Cosa fornisce | Fonte |
|---|---|---|
| **TCV-overview** | R₀=0.88 m, a=0.25 m, B_T≤1.5 T, I_p≤1 MA, κ≤2.8, δ∈[-0.7,1] | [EPFL SPC TCV page](https://www.epfl.ch/research/domains/swiss-plasma-center/research/tcv/research_tcv_tokamak/) + [Reimerdes et al. *Nucl. Fusion* 62 (2022)](https://strathprints.strath.ac.uk/80270/1/Reimerdes_etal_NF_2022_Overview_of_the_TCV_tokamak_experimental_programme.pdf) |
| **TCV-coils-LRP755** | 16 shaping coils E1-E8 (R=0.505 m) + F1-F8 (R=1.3095 m), 3 T coils, OH circuit | [EPFL Infoscience LRP-755-13](https://infoscience.epfl.ch/record/210807) — autoritativo |
| **TCV-coils-freegs** | Stesse posizioni implementate in `freegs.machine.TCV()` (validate community-wide) | [freegs source `machine.py`](https://github.com/freegs-plasma/freegs/blob/main/freegs/machine.py) |
| **TCV-control-loop** | SCD a 25 kHz, Degrave RL policy a 10 kHz | [Galperti et al. *Fusion Eng. Des.* 2024](https://www.sciencedirect.com/science/article/pii/S0920379624004915) |
| **TCV-current-limits** | E/F coils: 7.7 kA; OH coils: 20 kA flat-top | [EPFL Infoscience LRP-755-13](https://infoscience.epfl.ch/record/210807) |
| **TCV-aux-heating** | 9 girotroni (4.3 MW ECRH) + 2 NBI (1.3 MW each) | [EPFL SPC heating page](https://www.epfl.ch/research/domains/swiss-plasma-center/research/tcv/research_tcv_heating/) |

## B. Riferimento metodologico (deep RL su TCV)

| Tag | Cosa fornisce | Fonte |
|---|---|---|
| **Degrave-2022** | Architettura RL+sim, 19 canali tensione coil, target shape control | [Degrave et al. *Nature* 602:414 (2022)](https://www.nature.com/articles/s41586-021-04301-9) |
| **Tracey-2024** | Limitazioni pratiche del 2022, follow-up | [Tracey et al. *Fusion Eng. Des.* 2024](https://www.sciencedirect.com/science/article/pii/S0920379624000140) |

## C. Simulatori di plasma open-source

| Tag | Versione | Cosa fa | Fonte |
|---|---|---|---|
| **freegs-0.8.2** | installed | Grad-Shafranov free-boundary statico (NumPy/SciPy) | [GitHub freegs-plasma/freegs](https://github.com/freegs-plasma/freegs) |
| **FreeGSNKE-paper** | — | GS evolutivo con Newton-Krylov, validato MAST-U | [Amorisco et al. *Phys. Plasmas* 31:042517 (2024)](https://pubs.aip.org/aip/pop/article/31/4/042517/3286904/FreeGSNKE-A-Python-based-dynamic-free-boundary) |
| **TORAX** | — | Trasporto core 1D, JAX-differentiable | [Citrin et al. arXiv:2406.06718](https://arxiv.org/html/2406.06718v2) |

## D. Fisica del plasma — formule canoniche

### D.1 Limite di Greenwald (densità massima)

$$
n_{\text{GW}} \,[10^{20}\,\text{m}^{-3}] = \frac{I_p \,[\text{MA}]}{\pi \cdot a^2 \,[\text{m}^2]}
$$

**Fonte**: Greenwald, M. *et al.* "A new look at density limits in tokamaks." *Nuclear Fusion* 28:2199 (1988). Rivisitato in [Greenwald, *PPCF* 44:R27 (2002)](https://doi.org/10.1088/0741-3335/44/8/201).

### D.2 Limite di Troyon (β-limit)

$$
\beta_N = \beta \,[\%] \cdot \frac{a \,[\text{m}] \cdot B_T \,[\text{T}]}{I_p \,[\text{MA}]} \quad \text{con } \beta_N^{\max} \approx 2.8 \cdot 4 \ell_i
$$

dove $\beta = 2\mu_0 \langle p\rangle/B_T^2$. **Fonte**: Troyon, F. *et al.* "MHD-limits to plasma confinement." *Plasma Phys. Control. Fusion* 26:209 (1984).

### D.3 Safety factor q₉₅ (approssimazione)

$$
q_{95} \approx \frac{5 \cdot a^2 \cdot B_T \cdot \kappa_{\text{shape}}}{R_0 \cdot I_p \,[\text{MA}]} \cdot f_{\text{shape}}(\delta, \kappa)
$$

**Fonte**: Wesson, J. *Tokamaks*, 4th ed., Oxford University Press, 2011, §3.6.

### D.4 IPB98(y,2) energy confinement scaling

$$
\tau_E^{\text{IPB98(y,2)}} = 0.0562 \cdot H_{98} \cdot I_p^{0.93} \cdot B_T^{0.15} \cdot P^{-0.69} \cdot n_e^{0.41} \cdot M^{0.19} \cdot R^{1.97} \cdot \epsilon^{0.58} \cdot \kappa^{0.78}
$$

con I_p [MA], B_T [T], P [MW], n_e [10¹⁹ m⁻³], M [amu], R [m], ε=a/R, κ=elongation.
**Fonte**: ITER Physics Basis, *Nucl. Fusion* 39:2175 (1999), Table 7.

### D.5 Miller parametric LCFS (per reference shapes)

$$
R(\theta) = R_0 + a \cdot \cos\!\left(\theta + \arcsin(\delta) \cdot \sin\theta\right)
$$
$$
Z(\theta) = Z_0 + \kappa \cdot a \cdot \sin(\theta)
$$

**Fonte**: Miller, R.L. *et al.* "Noncircular, finite aspect ratio, local equilibrium model." *Phys. Plasmas* 5:973 (1998), [DOI:10.1063/1.872666](https://doi.org/10.1063/1.872666).

### D.6 Equazione di evoluzione dei circuiti coil

$$
\mathbf{M} \frac{d\mathbf{I}}{dt} + \mathbf{R} \cdot \mathbf{I} = \mathbf{V}(t)
$$

dove $\mathbf{M}$ è la matrice di mutua induttanza (incluso plasma-coil coupling), $\mathbf{R}$ è la matrice resistiva diagonale.
**Fonte**: Walker, M.L. & Humphreys, D.A. "Valid coordinate systems for linearized plasma shape response models in tokamaks." *Fusion Sci. Technol.* 50:473 (2006).

## E. Algoritmo FMC

| Tag | Riferimento | Implementazione |
|---|---|---|
| **FMC-paper** | [Hernández-Cerezo & Duran-Ballester. arXiv:1803.05049v5 (2020)](https://arxiv.org/abs/1803.05049) | `repos/fragile/src/fragile/fractalai.py` |
| **FMC-replication** | Boxing 96/100 in 7 min, M1 Pro NumPy | `work/03_atari_replication/` |
| **FMC-craftax** | 6.87% Crafter score, 0 training | `work/05_craftax/` |

## F. Convenzioni di unità

Tutto in **SI strict** internamente, eccetto dove formula richiede unità "engineering":
- Lunghezze: m
- Correnti: A
- Tensioni: V
- Densità: m⁻³
- Energia: J
- Potenza: W
- Tempo: s
- B-field: T
- Pressione: Pa

In *output* possiamo presentare in unità engineering (MA, kA, kV, 10¹⁹m⁻³) per leggibilità.

## G. Ipotesi documentate (gap chiusi a mano)

| ID | Assunzione | Razionale | Fonte alternativa cercata |
|---|---|---|---|
| H1 | Voltage rails ±1500 V su E/F coils | Ordine di grandezza tipico thyristor PS, EPFL non pubblica esatti | LRP-755-13, Fasoli 2023, [Tandfonline 2022](https://www.tandfonline.com/doi/full/10.1080/15361055.2022.2043511) — non trovato |
| H2 | Resistenza coil R = 1 mΩ uniforme | Ordine di grandezza per coil rame raffreddati a 77K | Standard plasma engineering |
| H3 | Mutua induttanza calcolata via Neumann formula tra anelli filiformi | Approssimazione finite-section trascurata in Phase 1 | Jackson, *Classical Electrodynamics* §5.17 |

Tutte le ipotesi sono override-abili da `config/tcv_geometry.yaml`.
