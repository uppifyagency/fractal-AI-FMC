# Scale-Free Selection, Not Gibbs Equilibrium: A Corrected Statistical-Mechanical Foundation for Fractal Monte Carlo

> **Draft — v0.2, 2026-07-10.** FractalAI project, night session 2026-07-09/10 (Wave 5, papers; Wave 3b strengthening integrated).
> **Status of the manuscript:** internal working draft for an eventual international venue. English body per project convention (prose may be Italian; a theory paper for external submission is written in English).
> **Honesty protocol.** Every non-trivial claim is tagged `[PROVEN]`, `[DIFF-APPROX]`, `[SKETCH]`, or `[NUMERICAL]`. `[PROVEN]` = closed-form derivation with hypotheses and proof. `[DIFF-APPROX]` = closed-form diffusion-limit result, verified numerically against the exact kernel with a stated regime of validity, but not proven as a functional convergence theorem. `[SKETCH]` = argued, not fully proven; the gap is named. `[NUMERICAL]` = Monte-Carlo / simulation evidence only (all numbers traced to a named script; none invented). The author's second role is **falsifier of the author's own statements** (§8).
> **Primary internal sources.** `work/14_night_2026-07-09/wave3_validation/W31_stazionaria_corretta.md` (Theorem 2′; script `w31_stationary_check.py`, seed 20260709, numpy 2.2.6); `.../W32_alpha_eff.md` (α_eff theorem; scripts `w32_sympy_deriv.py`, `w32_alpha_eff_check.py`, seed 20260709); `.../W3B_teoria_rafforzata.md` (multi-seed robustness + Wright-distribution stationary law; scripts `w3b_robustness.py`, `w3b_mutation_diffusion.py`, seed 20260709, numpy 2.2.6, scipy 1.16.1); `docs/MATH_CANON.md` (Def. 2, 4; Conjecture A; Theorems 1–3); `work/02_deep_dives/07_wright_fisher_mapping.md` (WF mapping, q≈−0.948); `work/14_night_2026-07-09/WAVE2_SINTESI.md` (triage, gaps G1–G4).

---

## Abstract

Fractal Monte Carlo (FMC) is a zero-training, per-instance planning algorithm that resamples a swarm of "walkers" by a pairwise cloning rule weighted by a self-normalized *virtual reward*. The canonical paper (Hernández-Cerezo & Duran-Ballester, 2020, §4.2.4) and the project canon (MATH_CANON Theorem 2) claim that FMC's cloning kernel is a Metropolis–Hastings (MH) chain whose stationary law is a finite-temperature Gibbs distribution $\pi^\* \propto R^\alpha\rho^{-\beta}$, with $\alpha$ playing the role of an inverse temperature. We show this claim is false and replace it with a correct foundation, contributing three results. **(1)** We derive in closed form the *effective inverse temperature* induced by the `relativize` z-score, $\alpha_{\rm eff} = C\,\alpha/\sigma_R$ with $C=\mathbb{E}_{z\sim\mathcal N(0,1)}[g(z)]=0.7223$ [PROVEN pointwise; NUMERICAL for the population constant, MC error ≤0.29%], exposing an *emergent annealing* ($\sigma_R\!\downarrow\Rightarrow$ pressure $\uparrow$) and the incomparability of a bare $\alpha$ across benchmarks. **(2)** We prove the FMC acceptance function is $a_{\rm FMC}(r)=\mathrm{clip}(r-1,0,1)$, which is *not* MH ($\min(r,1)$): it is uphill-only, hence non-reversible, so no full-support Gibbs law satisfies detailed balance. The corrected object is a **Moran / Wright–Fisher (WF) selection process** (fixation, not Gibbs): neutral drift for $\alpha=0$ (heterozygosity-decay exponent $q=-1.018$, 95% CI $[-1.033,-1.003]$; fixation-time exponent $p=+1.025$, $[+1.012,+1.039]$; both under per-tick fluctuating fitness over 25 seeds) and directional fixation with probability 1 for $\alpha>0$. With mutation, the non-degenerate stationary law is given in closed form as a **Wright distribution** $\phi_\infty(x)\propto x^{\theta-1}(1-x)^{\theta-1}e^{\sigma x}$, its drift and diffusion coefficients derived from the true clip acceptance and verified against the exact kernel (total variation $\to0$ as $N\to\infty$) [DIFF-APPROX]. **(3)** We reframe the "magic-6" branching factor as a WF *transient*, not a universal law. We connect (1)+(2) to explain why effective reward shaping for FMC must be multiplicative and tiered, and we resolve the apparent tension between contribution (1)'s inverse-temperature language and contribution (2)'s retraction of Gibbs via a two-regime distinction: the frozen discrete kernel is non-reversible, while the fluctuating-fitness-plus-mutation diffusion is reversible with respect to $\phi_\infty$. Finally we **close the quantitative link** between (1) and (2): the selection coefficient and the effective inverse temperature obey $s_{\rm eff}=2\Phi'(0)\,\alpha_{\rm eff}\,\Delta R$ — a chain rule proving they are one selection mechanism in two coordinate systems — and we give both the clip acceptance $\Phi$ and the Wright diffusion coefficient (via a closed-form pairwise **co-ancestry** correction, $+12.8\%$ over the independent-flip estimate) in closed form, verified against the exact kernel and an adversarial reviewer [PROVEN at leading order + NUMERICAL].

---

## 1. Introduction

### 1.1 What FMC is

Fractal Monte Carlo (Hernández-Cerezo & Duran-Ballester, 2020, arXiv:1803.05049v5; companion empirical paper, arXiv:1807.01081) is a planning algorithm that, at each decision, launches a swarm of $N$ *walkers* forward through a (reversible, `set_state`-able) simulator for $M$ ticks. Each walker carries an *initial-action label* $\ell^{(i)}$. Between ticks the swarm is resampled by a **cloning** step in which each walker may copy a random partner. After $M$ ticks the algorithm returns the modal surviving label. FMC needs no learned value function and no gradient; it plans per instance. Its empirical footprint is small and reproducible: 231 LOC of NumPy solve Atari Boxing (5/5 seeds, ~82 s, CPU-only; MATH_CANON / project replication).

The two ingredients that make FMC distinctive are (i) `relativize`, a per-tick z-score that self-normalizes rewards over the population — making the algorithm *scale-free* — and (ii) the pairwise, embarrassingly-parallel cloning rule, which does no global weight normalization.

### 1.2 What the original paper claims — and what we correct

The canonical paper (§4.2.4) and MATH_CANON (Theorem 2, lines 253–301; and deep dive 01 §4, there labeled "Teorema 3") give FMC a **statistical-mechanical foundation**: the cloning kernel is asserted to be a Metropolis–Hastings chain, and the composite dynamics $\mathcal S\circ\mathcal C$ (perturb ∘ clone) is claimed to converge to a finite-temperature **Gibbs equilibrium**

$$
\pi^\*(x) \;\propto\; R(x)^\alpha\,\rho(x)^{-\beta},
$$

with the explicit dictionary "$\alpha$ = inverse temperature $\beta_{\rm stat}$, cloning = Gibbs selection, equilibrium = Boltzmann $\propto e^{-\beta_{\rm stat}U}$" (MATH_CANON lines 285–300). This is the paper's headline mechanistic explanation of *why FMC works*.

We correct this on two fronts and then supply the missing quantitative piece:

- **Overclaim identified (gap G1 in WAVE2_SINTESI).** The MH identification rests on the algebraic identity, stated at MATH_CANON line 186, that $\mathrm{clip}(r-1,0,1)=\min(r,1)$. It is false. The true acceptance function is uphill-only, the kernel is non-reversible, and no full-support Gibbs law is invariant. The stationary behavior is *fixation* (a Moran / Wright–Fisher process), not a finite-temperature Gibbs distribution.
- **The correct scalar of selection.** `relativize` does induce a Boltzmann-like *selection pressure*, but it is $\alpha_{\rm eff}=C\,\alpha/\sigma_R$, not $\alpha$; it has units of inverse reward and is $\sigma_R$-dependent by dimensional necessity. This is a *new* closed-form result, not present in the paper.

The rest of the paper is organized as: background (§2); contribution 1, $\alpha_{\rm eff}$ (§3); contribution 2, the Moran/WF stationary law — now with a **closed-form Wright diffusion coefficient** via a pairwise co-ancestry correction (§4.5, §8.1); contribution 3, the magic-6 transient (§5); the synthesis on multiplicative-tiered shaping (§6); and the **resolution** of the apparent §3–§4 tension via the proven bridge $s_{\rm eff}=2\Phi'(0)\,\alpha_{\rm eff}\,\Delta R$ (§7.3), which shows the two "effective temperatures" are one selection mechanism in two coordinate systems; limitations (§8); references (§9). The two closures (§4.5 co-ancestry, §7.3 bridge) and the retraction were independently checked by an adversarial reviewer.

---

## 2. Background

### 2.1 `relativize` (Definition 2)

Given raw rewards $\mathbf r=(r^{(1)},\dots,r^{(N)})$ with population mean $\mu$ and standard deviation $\sigma$, the z-score is $z^{(i)}=(r^{(i)}-\mu)/(\sigma+\varepsilon)$, $\varepsilon=10^{-10}$, and

$$
\widehat R(z)=\begin{cases}e^{z} & z\le 0\\[2pt] 1+\log(1+z) & z>0.\end{cases}
$$

$\widehat R$ is strictly positive, $C^1$ at $z=0$ ($\widehat R(0)=1$, $\widehat R'(0)=1$), log-compressed for $z\to+\infty$, and — the property we lean on — **globally affine-invariant**: $\widehat R(a\mathbf r+b)=\widehat R(\mathbf r)$ for any $a>0,b$, because the z-score is. (Code and canon are bit-identical up to one boundary detail: the reference code uses `if std==0: return ones` rather than $\sigma+\varepsilon$, so at total population collapse $\sigma_R\to0$ the code zeroes the pressure while the closed form diverges; irrelevant for $\sigma_R\gg\varepsilon$. See W32 §1.)

The uniqueness of $\widehat R$ under the axioms A1–A5 is **not** proven (gap G3): A1–A5 pin down an asymptotic class, not a unique function. We do not rely on uniqueness anywhere below.

### 2.2 Virtual reward (Definition 3)

For walker $i$ with a random partner $j(i)$,
$$
\mathrm{VR}^{(i)} = \big(\widehat R^{(i)}\big)^{\alpha}\cdot\big(\widehat D^{(i)}\big)^{\beta}\in\mathbb R_{>0},
$$
where $\widehat R^{(i)}=\widehat R(R(W^{(i)}))$ and $\widehat D^{(i)}=\widehat R(d(W^{(i)},W^{(j(i))}))$ is the relativized distance to the partner. $\alpha\ge0$ weights reward-seeking; $\beta\ge0$ weights diversity. The limits $\alpha{=}0,\beta{=}1$ ("Common Sense") and $\alpha{=}\beta{=}1$ (paper default) are the operating points of interest.

### 2.3 Cloning kernel (Definition 4)

For each walker $i$ with random partner $k$, the **cloning rate** is
$$
\rho_{\rm clone}(i\to k)=\begin{cases}1 & \mathrm{VR}^{(i)}=0\\ 0 & \mathrm{VR}^{(k)}\le\mathrm{VR}^{(i)}\\ \dfrac{\mathrm{VR}^{(k)}-\mathrm{VR}^{(i)}}{\mathrm{VR}^{(i)}} & 0<\mathrm{VR}^{(i)}<\mathrm{VR}^{(k)},\end{cases}
$$
and the **effective clone probability** is the clip $P_{\rm clone}=\min(\rho_{\rm clone},1)\in[0,1]$. On a clone, both state and label of $i$ are overwritten by those of $k$. The comparison is implemented as $U\le\rho_{\rm clone}$, $U\sim\mathrm{Unif}(0,1)$; writing $r:=\mathrm{VR}^{(k)}/\mathrm{VR}^{(i)}$, the realized acceptance is therefore
$$
\boxed{\,a_{\rm FMC}(r)=\Pr[U\le r-1]=\mathrm{clip}(r-1,0,1)=\min(\max(r-1,0),1).\,}
$$
This is the object §4 dissects. It is **pairwise** (each walker faces one partner), unlike systematic/multinomial SMC resampling; the two coincide only as $N\to\infty$.

### 2.4 The SMC / Feynman–Kac framing and its limits (gap G2)

FMC has been read as an interacting particle system for a Feynman–Kac flow $\eta_t$ with potential $G_t=\mathrm{VR}_t$ (deep dive 05; Del Moral, 2004), yielding an $L^p$ convergence rate $O(1/\sqrt N)$ (MATH_CANON Theorem 1). We flag, following WAVE2 gap **G2**, that the classical Del Moral theory does **not** apply off the shelf here: FMC's potential is (i) *mean-field* — `relativize` couples every walker's weight through the population $(\mu,\sigma)$ — and (ii) *stochastic* — $\widehat D$ depends on a random partner. The limiting flow $\eta_t$ is thus a self-referential fixed point that is not obviously well-defined, and the "pairwise variance $\le$ multinomial variance" step is asserted, not proven. Theorem 1 is therefore best read as a `[SKETCH]`. This matters for §4: the very tool one would use to *rescue* a mean-field Gibbs statement is itself on soft ground.

---

## 3. Contribution 1 — The effective inverse temperature of `relativize`

### 3.1 Motivation and the dimensional argument

A naive Boltzmann selector would set $\mathrm{VR}\propto e^{\alpha_B R}$, i.e. $\log\mathrm{VR}=\alpha_B R+\text{const}$, so its *selection pressure* $\partial_R\log\mathrm{VR}=\alpha_B$ is constant with units $[\text{reward}]^{-1}$ — a genuine inverse temperature. FMC instead applies $\alpha$ as a dimensionless exponent on the z-scored, relativized reward. We therefore define FMC's selection pressure in the same dimensional units:
$$
\boxed{\;\alpha_{\rm eff}(R):=\frac{\partial\log\mathrm{VR}}{\partial R}\;},\qquad [\alpha_{\rm eff}]=[\text{reward}]^{-1}.
$$
Because $\alpha$ is dimensionless and $\alpha_{\rm eff}$ has units of inverse reward, the missing dimension can only be carried by $\sigma_R$ (units of reward): **it is dimensionally inevitable that $\alpha_{\rm eff}\propto1/\sigma_R$.**

### 3.2 Closed form — `[PROVEN]`

Isolate the reward channel (fix the distance term; $\beta=0$ or $\widehat D$ constant), so $\log\mathrm{VR}=\alpha\log\widehat R(z)+\text{const}$ with $z=(R-\mu_R)/\sigma_R$. By the chain rule $\alpha_{\rm eff}=\alpha\cdot\frac{d\log\widehat R}{dz}\cdot\frac{1}{\sigma_R}$. Symbolic differentiation (`w32_sympy_deriv.py`, executed) gives the two branches of $d\log\widehat R/dz$: $1$ for $z\le0$, and $1/[(1+z)(1+\log(1+z))]$ for $z>0$. Hence

$$
\boxed{\;\alpha_{\rm eff}(z;\alpha,\sigma_R)=\frac{\alpha}{\sigma_R}\,g(z),\qquad
g(z)=\begin{cases}1 & z\le0\\[4pt]\dfrac{1}{(1+z)\,[1+\log(1+z)]} & z>0.\end{cases}\;}
$$

Consequences (all proven symbolically):
1. **The true inverse temperature is $\alpha/\sigma_R$**, not $\alpha$. The shape factor $g(z)\le1$ modulates only the right tail.
2. **Below the mean the pressure is exact and constant:** for all $z\le0$, $\alpha_{\rm eff}=\alpha/\sigma_R$. Every below-average walker feels the same pressure.
3. **Above the mean the pressure decays:** $g(z)\to0$ as $z\to+\infty$. FMC does not reward outliers linearly — it saturates. An exceptional walker does not "run away" with the population.
4. **$C^1$ continuity at $z=0$:** $g(0^+)=1=g(0^-)$ (sympy limit verified).

### 3.3 Population scalar — `[PROVEN]` structure, `[NUMERICAL]` constant

To collapse $\alpha_{\rm eff}(z)$ into one number comparable to $\alpha_B$, take the OLS slope of $\log\mathrm{VR}$ on $R$:
$$
\hat\beta=\frac{\mathrm{Cov}(\log\mathrm{VR},R)}{\mathrm{Var}(R)}=\frac{\alpha}{\sigma_R}\,\mathbb E[z\,\log\widehat R(z)].
$$
For $z\sim\mathcal N(0,1)$, **Stein's identity** ($\mathbb E[zf(z)]=\mathbb E[f'(z)]$) collapses the two ways of measuring into one: $\mathbb E[z\,\log\widehat R(z)]=\mathbb E[g(z)]=:C$. That is, the *population regression slope* equals the *mean pointwise elasticity*. Hence

$$
\boxed{\;\bar\alpha_{\rm eff}(\alpha,\sigma_R)=C\,\frac{\alpha}{\sigma_R},\qquad C=\mathbb E_{z\sim\mathcal N(0,1)}[g(z)]=0.7223\ \text{(quadrature)}.\;}
$$

$C$ is a pure number: the $z\le0$ half contributes $0.5$, the log-compressed right tail adds $0.222$. **The scaling law $\propto\alpha/\sigma_R$ is pure z-score algebra — distribution-independent**; only the prefactor $C$ is distribution-dependent.

### 3.4 Monte-Carlo verification — `[NUMERICAL]`

All numbers from `w32_alpha_eff_check.py`, seed 20260709.

| Check | Result |
|---|---|
| (A) pointwise closed form vs finite-difference $\partial\log\mathrm{VR}$, grid $z\in\{-2,-1,-0.2,0.3,1,3,8\}$ | max rel. err $2.4\times10^{-9}$, $3.9\times10^{-9}$, $7.4\times10^{-10}$ across $(\alpha,\sigma_R)\in\{(1,1),(2,5),(0.5,0.3)\}$ |
| (B/D) empirical slope vs $C\alpha/\sigma_R$, Gaussian $N=2\times10^5$, $\alpha\in\{0.5,1,2\}\times\sigma_R\in\{0.2,0.5,1,3,10\}$ | **max rel. err 0.29%**, mean 0.11%; laws $\hat\beta\propto\alpha$ and $\hat\beta\propto1/\sigma_R$ confirmed over 2 decades of $\sigma_R$ |
| (C) Stein identity, $N=2\times10^7$ | $\mathbb E[z\log\widehat R]=0.72236$, $\mathbb E[g(z)]=0.72226$ (rel. err 0.014%); quadrature $C=0.72233$ |
| (E) non-Gaussian robustness (uniform) | $C_{\rm unif}=0.7383$; law $C_{\rm unif}\alpha/\sigma_R$ holds to max rel. err 0.15% |

Overall worst-case MC error **≤0.29%** (Gaussian), typically ~0.1%. $C\in[0.72,0.74]$ across the distributions tested.

**Multi-seed robustness** (W3B, `w3b_robustness.py`, 25 seeds, bootstrap CI): the population constant tightens to $C_{\rm gauss}=0.7225$ (95% CI $[0.7221,0.7227]$, sd $0.0008$) and $C_{\rm unif}=0.7384$ (95% CI $[0.7382,0.7386]$), matching the W32 quadrature values to 3–4 digits; the $\propto\alpha/\sigma_R$ law is invariant across a $3\times3$ $(\alpha,\sigma_R)$ grid (spread $0.0016$ Gaussian, $0.0007$ uniform).

### 3.5 Corollaries

- **Emergent annealing.** As the swarm converges, $\sigma_R\downarrow$, so $\bar\alpha_{\rm eff}=C\alpha/\sigma_R\uparrow$: selection pressure *rises by itself* toward the end of the search. FMC has an emergent, unprogrammed temperature anneal — the population starts "hot" (large $\sigma_R$, low pressure, explores) and "cools" (small $\sigma_R$, high pressure, exploits). This is a quantitative form of the informal "chaos/order frontier" (project discrepancy D3).
- **$\alpha$ is not a portable temperature.** The physical pressure is $C\alpha/\sigma_R$. The *same* $\alpha$ yields different pressure on tasks with different reward scale, and in different phases of the *same* search. **Comparing $\alpha$ across benchmarks without normalizing by $\sigma_R$ is meaningless.**
- **Tail saturation.** Raising $\alpha$ does not make FMC Boltzmann-greedy on outliers, because $g(z)\to0$ in the tail — explaining diminishing returns of large $\alpha$.

> **Caveat / forward pointer.** Calling $\alpha_{\rm eff}$ an "inverse temperature" is a *dimensional analogy*. It quantifies *selection pressure per unit reward*, which — as §4 shows — feeds the selection coefficient of a Wright–Fisher process, **not** a reversible Gibbs equilibrium. The reader must not read "inverse temperature" here as reinstating the Gibbs law that §4 retracts. The precise, honest reconciliation is in §4.7 and §7.

---

## 4. Contribution 2 — The stationary law is Moran / Wright–Fisher, not Gibbs

### 4.1 The acceptance function is not Metropolis–Hastings — `[PROVEN + NUMERICAL]`

Compare $a_{\rm FMC}(r)=\mathrm{clip}(r-1,0,1)$ with $a_{\rm MH}(r)=\min(r,1)$ (Metropolis–Hastings) and $a_{\rm Barker}(r)=r/(1+r)$. From `w31_stationary_check.py` (Part 0):

| $r=\mathrm{VR}_k/\mathrm{VR}_i$ | $a_{\rm FMC}=\mathrm{clip}(r{-}1)$ | $a_{\rm MH}=\min(r,1)$ | $a_{\rm Barker}$ |
|---:|---:|---:|---:|
| 0.50 | 0.0000 | 0.5000 | 0.3333 |
| 0.80 | 0.0000 | 0.8000 | 0.4444 |
| 1.00 | 0.0000 | 1.0000 | 0.5000 |
| 1.20 | 0.2000 | 1.0000 | 0.5455 |
| 1.50 | 0.5000 | 1.0000 | 0.6000 |
| 1.80 | 0.8000 | 1.0000 | 0.6429 |
| 2.00 | 1.0000 | 1.0000 | 0.6667 |
| 3.00 | 1.0000 | 1.0000 | 0.7500 |

The three functions agree only for $r\ge2$. $a_{\rm FMC}$ has two properties MH lacks:
- **uphill-only:** $a_{\rm FMC}(r)=0$ for all $r\le1$ (no move toward equal-or-lower VR). MH accepts downhill moves with probability $r>0$.
- **sub-Metropolis on $(1,2)$:** $a_{\rm FMC}(r)=r-1<1=a_{\rm MH}(r)$.

The identity implicit at MATH_CANON line 186 — $\mathrm{clip}(r-1,0,1)=\min(r,1)$ — is algebraically false. Counterexamples: $r=0.8\Rightarrow0$ vs $0.8$; $r=1.5\Rightarrow0.5$ vs $1$. `[PROVEN + NUMERICAL]`

### 4.2 No full-support Gibbs law is reversible — `[PROVEN]`

Let $\pi$ be any full-support measure on configurations, and let $x,y$ differ in one walker's type with $\mathrm{VR}(y)>\mathrm{VR}(x)$ (i.e. $y$ is uphill). Then
$$
K(x\to y)=\tfrac1{N-1}a_{\rm FMC}\!\big(\tfrac{\mathrm{VR}(y)}{\mathrm{VR}(x)}\big)>0,\qquad K(y\to x)=\tfrac1{N-1}a_{\rm FMC}\!\big(\tfrac{\mathrm{VR}(x)}{\mathrm{VR}(y)}\big)=0,
$$
since $\mathrm{VR}(x)/\mathrm{VR}(y)<1\Rightarrow a_{\rm FMC}=0$. Detailed balance $\pi(x)K(x\to y)=\pi(y)K(y\to x)=0$ then forces $\pi(x)=0$, contradicting full support. **No full-support Gibbs measure $\pi^\*\propto R^\alpha\rho^{-\beta}$ at finite temperature is reversible-invariant for the cloning kernel.** ∎

This pinpoints the invalid step in the canon's Theorem 2 proof (MATH_CANON lines 269–283): it writes $\Pr[y\to x]=0$ and then still concludes a finite ratio $\pi^\*(x)/\pi^\*(y)=\widehat R(x)^\alpha/\widehat R(y)^\alpha$. With $\Pr[y\to x]=0$, detailed balance yields $\pi^\*(x)=0$, not a finite ratio. The canon implicitly swaps $a_{\rm FMC}$ for $a_{\rm MH}$ mid-derivation — exactly the line-186 error propagating. (The same swap occurs in deep dive 01 §4.)

### 4.3 Cloning contracts diversity — `[PROVEN]`

Cloning *copies* an existing walker's type; it never introduces a new type. Hence the set of present types is non-increasing in time. Without a mutation operator (the perturbation $\mathcal S$ acting as type-injection), diversity can only fall. The only absorbing states of cloning alone are those in which no present pair has $a_{\rm FMC}>0$, i.e. all walkers share the same VR (generically monomorphic). ∎

### 4.4 The corrected theorem

**Theorem 2′ (Fixation of the cloning kernel; absence of Gibbs equilibrium).**
*Setup.* $N$ walkers, each with a type in a finite set; the cloning kernel $\mathcal C$ acts walker-by-walker: each $i$ draws a uniform partner $k\ne i$ and adopts $k$'s type with probability $a_{\rm FMC}(\mathrm{VR}_k/\mathrm{VR}_i)$. No mutation.
*Hypotheses.* (H1) VR is a positive deterministic function of type (frozen landscape) or, in the neutral case, i.i.d. with no type bias. (H2) In the selection case, VR has a unique maximum.
*Statement.*
1. **(Support non-expansion)** `[PROVEN]` $\mathcal C$ introduces no absent type; present types are non-increasing; absorbing states are those with all VR equal (generically monomorphic).
2. **(Non-reversibility)** `[PROVEN]` For every full-support $\pi$ and every $x,y$ with $\mathrm{VR}(y)>\mathrm{VR}(x)$, $K(x\to y)>0=K(y\to x)$; detailed balance forces $\pi(x)=0$. Hence no finite-temperature Gibbs $\pi^\*\propto R^\alpha\rho^{-\beta}$ is reversible-invariant.
3. **(Fixation under selection)** `[PROVEN]` Under (H2), the count of the argmax-VR type is non-decreasing and reaches $N$ with probability 1. The unique stationary distribution is the point mass on the monomorphic argmax configuration; $b_{\rm eff}\to1$. This is the $\alpha\to\infty$ limit, not a finite $\alpha$.
4. **(Drift in the neutral case)** `[PROVEN + NUMERICAL]` With VR i.i.d. and unbiased, $\mathcal C$ is a neutral Moran/WF resampling: heterozygosity decays at rate $\lambda(N)\sim c/N$ (measured $q=-0.999$), fixation time is $O(N)$ generations (measured $p=1.056$), and each type fixes with probability equal to its initial frequency. With VR *exactly* constant, $\mathcal C=\mathrm{Id}$ (frozen).
5. **(Restoring a non-degenerate law — mutation is necessary)** `[DIFF-APPROX, verified]` A non-degenerate stationary law of the full process $\mathcal S\circ\mathcal C$ exists only if $\mathcal S$ acts as mutation (injecting new types). In the two-type diffusion limit it has the closed form of a **Wright distribution** (Theorem 2′.5, §4.6): $\phi_\infty(x)\propto x^{\theta-1}(1-x)^{\theta-1}e^{\sigma x}$ with $\theta=2N_e\mu$ and $\sigma=2N_e\,s_{\rm eff}$, where the selection coefficient $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$ is **renormalized by the uphill-only clip and the per-tick noise** — it is not the Moran $s$. It does **not** coincide with $\pi^\*\propto R^\alpha$ except in degenerate limits; $\alpha$ enters as selection intensity through $\sigma$, not as a thermodynamic inverse temperature.

*Proof.* Points 1–3: §4.1–4.3 plus a standard finite-Markov argument (a monotone absorbing chain is absorbed a.s.). Point 4: the Moran/WF mapping (deep dive 07) with numerics in §4.5. Point 5: the Fokker–Planck derivation and its numerical validation are in §4.6; it is a diffusion approximation (Kimura-style limit), not a functional convergence theorem — see §8. ∎

### 4.5 Numerical evidence — `[NUMERICAL]`

**Neutral drift** (Part B, exact FMC kernel, LogNormal$(0,0.5)$ VR, one tick = one generation):

| $N$ | $\lambda$ | $\lambda\cdot N$ | | $N$ | $T_{\rm fix}$ (ticks) |
|---:|---:|---:|---|---:|---:|
| 32 | 0.02098 | 0.671 | | 32 | 57.8 |
| 64 | 0.01098 | 0.702 | | 64 | 129.4 |
| 128 | 0.00534 | 0.683 | | 128 | 267.7 |
| 256 | 0.00265 | 0.680 | | 256 | 520.4 |

Power-law fits: heterozygosity $\lambda\sim N^q$ gives $q=-0.999$ (WF predicts $-1$; $\lambda N\approx0.68$ nearly constant over a decade of $N$); fixation time $T_{\rm fix}\sim N^p$ gives $p=1.056$ (WF/Moran predict $+1$). Frozen control (VR $\equiv1$, $N=64$): $H(\text{end})/H(0)=1.00000$. These are consistent with the independent WF-mapping result $q\approx-0.948$ at $\alpha=0$ (deep dive 07, 5.2% error).

**Multi-seed robustness under per-tick fluctuating fitness** (W3B, `w3b_robustness.py`, 25 seeds, 20 000-resample bootstrap CI). The frozen-VR, single-seed numbers above survive when every walker *redraws* $\mathrm{VR}=e^{g}$, $g\sim\mathcal N(m_{\rm type},\sigma_v^2)$, each tick (equal type-means for the neutral test): $q=-1.018$ (95% CI $[-1.033,-1.003]$, sd $0.040$; WF $-1$) and $p=+1.025$ (95% CI $[+1.012,+1.039]$, sd $0.034$; WF $+1$), with $\lambda N\approx0.68$ and $T_{\rm fix}/N\approx1.9$–$2.0$ across $N\in\{32,64,128,256\}$. The exponent is stable across noise levels ($q=-1.012,-1.011,-1.029$ at $\sigma_v=0.25,0.5,1.0$). **Honest caveat:** the CIs *exclude* the exact WF values by ~2% — a finite-tick ($M=40$) / four-$N$-point log-log bias, not a defect of the mapping. The Moran/WF signature (direction and order of magnitude) is unambiguous.

**Closed-form diffusion coefficient (co-ancestry, closed 2026-07-10, §8.1).** The near-constant $\lambda N\approx0.68$ above is not merely measured: it has a closed form. For the neutral synchronous pairwise kernel, the per-tick heterozygosity-decay rate (inverse variance-effective size) is, to leading order in $1/N$,
$$\lambda N = 2\varphi_0 + \big(\langle a_{\rm in}^2\rangle - 2\langle a_{\rm in}a_{\rm out}\rangle\big),\quad a_{\rm in}(t)=\mathbb E_g[\mathrm{clip}(e^{t-g}-1,0,1)],\ a_{\rm out}(t)=a_{\rm in}(-t),$$
the parenthesised term being the **pairwise co-ancestry correction** (two distinct offspring sharing a parent one tick back). At $\sigma_v{=}0.5$ this gives $\lambda N=0.6755$ ($+12.8\%$ over the independent-flip baseline $2\varphi_0=0.599$), matching the exact kernel to $+0.1\%$ across $N\in\{100,800\}$ — and the neutral recursion $\mathbb E[H_{t+1}]=(1-p_{\rm coal})\mathbb E[H_t]$ is exact at leading order, so this promotes the Wright diffusion coefficient from `[NUMERICAL]` to `[PROVEN at leading order in $1/N$]` (adversarially verified; §8.1). It also yields a closed form for $\Phi(m)$ itself (normal CDFs), so $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$ requires no quadrature.

**Fixation under selection** (Part A, 20 000 runs, start = 1 copy of the fit type; VR$_A=1+s$, VR$_B=1$):

| $s$ | $N$ | FMC (exact) | MH | Moran theory | neutral $1/N$ |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 32 | **1.0000** | 0.0956 | 0.0954 | 0.0312 |
| 0.10 | 64 | **1.0000** | 0.0903 | 0.0911 | 0.0156 |
| 0.10 | 128 | **1.0000** | 0.0882 | 0.0909 | 0.0078 |
| 0.50 | 64 | **1.0000** | 0.3399 | 0.3333 | 0.0156 |
| 1.00 | 64 | **1.0000** | 0.4997 | 0.5000 | 0.0156 |

Reading: $a_{\rm MH}$ reproduces the classical Moran-with-selection fixation formula $\rho=(1-\gamma)/(1-\gamma^N)$, $\gamma=1/(1+s)$, to 3 digits — so MH *is* reversible with a finite-temperature selection balance, and the canon's "Gibbs" framing actually describes $a_{\rm MH}$, **not** $a_{\rm FMC}$. The true $a_{\rm FMC}$ gives fixation $=1.0000$ for every $s>0$: an absorbing, uphill-only dynamics with no finite-$T$ balance. The gap is large (e.g. $s=0.5,N=64$: $1.000$ vs $0.340$).

### 4.6 Reconciling $\alpha$ with §3

The corrected reading of $\alpha$:

| Canon claim (Thm 2) | Correction (Thm 2′) |
|---|---|
| $a=\min(r,1)$ (MH) | $a_{\rm FMC}=\mathrm{clip}(r-1,0,1)$, uphill-only |
| reversible, detailed balance | non-reversible |
| $\pi^\*\propto R^\alpha\rho^{-\beta}$ (finite-$T$ Gibbs) | cloning-only: point mass (fixation), $b_{\rm eff}\to1$ |
| $\alpha$ = inverse temperature | $\alpha$ = selection intensity ($s\propto\alpha\,\mathrm{Var}\log\widehat R$) |
| $\alpha=0\Rightarrow\pi^\*$ uniform | $\alpha=0$: neutral Moran/WF drift → still fixation in $O(N)$ |
| thermodynamic equilibrium | mutation–selection–drift equilibrium (needs $\mathcal S$ as mutation) |

The bridge to §3: $\alpha_{\rm eff}=C\alpha/\sigma_R$ is the selection *pressure per unit reward*; over a reward gap $\Delta R$ it yields a fitness differential $s_{\rm eff}\approx\alpha_{\rm eff}\cdot\Delta R$, which is the selection coefficient of the Moran/WF process. So §3 gives the *strength* of selection and §4 gives its *stationary consequence* (fixation, not Gibbs). **The precise map $\alpha_{\rm eff}\mapsto s_{\rm eff}$ is now proven (Theorem 4′, §7.3, closed 2026-07-10):** it is a chain rule composing two linearizations of the *same* clip acceptance,
$$s_{\rm eff}=2\Phi'(0)\,\alpha_{\rm eff}\,\Delta R+O(\Delta R^3),$$
with $\Phi'(0)=e^{\tau^2/2}[F(\ln2;\tau^2)-F(0;\tau^2)]$ ($\tau^2=2\sigma_v^2$) the clip's marginal transmission (only the transition band $0<u<\ln2$ transmits selection). $\alpha_{\rm eff}$ and $s_{\rm eff}$ are therefore not two rival temperatures but one selection mechanism in two coordinate systems — reward$\to$log-VR (LINK A, `relativize`) and log-VR$\to$frequency (LINK B, clip). Verified end-to-end (error $\to0$ as $\Delta R\to0$) and on a coupled relativize+clone simulation that pins $\sigma_v$ from the population (§7.3). No longer `[SKETCH]`.

---

## 5. Contribution 3 — The "magic-6" branching factor as a Wright–Fisher transient

Sergio's oral claim (Radient 2026, ch. 16) is that FMC bifurcates "six by six" — a universal optimal effective branching factor $b_{\rm eff}\approx6$ (Definition 6: $b_{\rm eff}=\exp H(\{p_a\})$, perplexity of the surviving-label distribution). MATH_CANON Conjecture A records this as **falsified as a universal law** and re-derived as a WF transient. We restate that result as a corollary of Theorem 2′.

The full parameter surface (MATH_CANON Conjecture A v0.4.0):
$$
b_{\rm eff}^\*(\alpha,\beta{=}0,K,N,M)\approx 1+(K-1)\cdot\mathcal F(M/N)\cdot\mathcal G(\alpha,K),
$$
with $\mathcal F$ decaying from $1$ to $0$ as $M/N\to\infty$. Four consolidated facts:
1. Asymptotically ($M\to\infty$, $\alpha>0$, finite $N$): $b_{\rm eff}\to1$ — the fixation limit of Theorem 2′.3.
2. Large $N$, fixed $M$: $b_{\rm eff}\to K-1$ (selection has no time to act); the deficit scales as $K-b_{\rm eff}\propto N^{-q}$, $q\approx0.45$ at $\alpha=0.1$, sharpening to $q=-0.948$ at $\alpha=0$ — the WF fixation-time law.
3. $\alpha=0,\beta=0$: $b_{\rm eff}\to K$ (fully neutral).
4. **Sergio's "6"** is the surface value at $(K=9,\,N\approx32\text{–}64,\,M=15,\,\alpha=0.1)$ — *triply contingent*, not a fixed point.

Empirically, a $K$-scan at fixed $M=15,N=32$ gives $b_{\rm eff}^\*\approx1.53\,K^{0.6}$ (a power law $\sim$25× better than the constant "6" model: SSE 2.46 vs 61.45), but an $M$-scan shows this exponent is a *transient* — $b_{\rm eff}\to1$ as $M$ grows. The magic-6 is thus a snapshot of neutral-to-fixation drift at $t/\tau\approx0.5$ with $K=9$ initial alleles, reconnecting it to Wright (1931), Kimura (1955), Ewens (1972), and Kingman (1982) rather than standing as a "Third Law."

*A separate methodological note (MATH_CANON line 480):* even the *use* of $b_{\rm eff}$ as a "frontier statistic" is questionable, because the label space is monotonically contractive under cloning for all $\alpha,\beta$ (no creative label generation) — so $b_{\rm eff}$ has no fixed point to be optimal at. This strengthens, not weakens, the transient reading.

---

## 6. Synthesis — why multiplicative-tiered shaping works

Contributions 1 and 2 jointly explain the project's empirically successful reward-shaping recipe (Conjecture D: multiplicative, inverse-tier-stacked achievement bonuses lifted Craftax-Classic-Symbolic from exp03 to exp17; the exp17 vs baseline effect is statistically solid — Wilcoxon $p=0.0019$, paired-$t$ $p=0.0030$, Cohen $d_z=0.74$, $n=18$).

The mechanism is `relativize`'s **global affine invariance** (§2.1), verified numerically in W32:
- additive uniform bonus $R+100$: $\max|\Delta\mathrm{VR}|=2.4\times10^{-14}$ → **invisible**;
- global multiplicative rescale $3R$: $\max|\Delta\mathrm{VR}|=1.8\times10^{-15}$ → **invisible**;
- *structured* per-tier multiplicative shaping (applied only to the subset that reached a tier): $\max|\Delta\mathrm{VR}|=0.54$ → **bites**.

So neither a uniform additive bonus nor a uniform multiplicative rescale can change selection — both are annihilated by the z-score. Only shaping that is **non-uniform across walkers** survives. Combining this with §3: since $\alpha_{\rm eff}\propto1/\sigma_R$, what governs selection is the ratio (between-tier gap)/(within-tier dispersion). Per-tier multiplicative shaping widens the between-tier gaps so they exceed local dispersion, producing a jump in $z$ large enough to survive the log-compression of the right tail (property 3 of §3.2) — something an additive bonus (absorbed by recentering) or a global factor (cancelled) cannot do. This is the structural, invariance-theoretic reason Conjecture D must be multiplicative and tiered, not additive-global. `[PROVEN]` for the affine-invariance and "structured-only" facts; `[SKETCH]` for the quantitative link to the observed chain-tier *compounding* (the exact super-additive amplification is not yet formalized).

---

## 7. The internal tension between Contributions 1 and 2 (stated plainly)

Per the honesty protocol we name, rather than hide, a possible contradiction.

**7.1 The apparent conflict.** §3 introduces an "effective *inverse temperature*" $\alpha_{\rm eff}=C\alpha/\sigma_R$ — Boltzmann language. §4 *retracts* the Gibbs/inverse-temperature reading of $\alpha$ and calls it a selection intensity. Read carelessly, §3 seems to reinstate what §4 kills.

**7.2 Why they do not actually contradict.** The two use "inverse temperature" at different levels:
- $\alpha_{\rm eff}$ is a **local, differential** quantity — the slope $\partial_R\log\mathrm{VR}$. It is well-defined and exactly $C\alpha/\sigma_R$ regardless of what the stationary law is. It measures how sharply VR discriminates rewards *at a tick*.
- The Gibbs claim was a **global, stationary** statement — that iterating the kernel yields $\pi^\*\propto R^\alpha$. That is false because the kernel is uphill-only/non-reversible.

A selector can have a perfectly good instantaneous selection pressure (a "temperature") and yet **not** relax to the Gibbs measure of that temperature — precisely because it is not a reversible MH move. That is FMC's case. $\alpha_{\rm eff}$ therefore sets the *selection coefficient* $s$ of the Moran/WF process of §4, whose stationary behavior is fixation (or, with mutation, a mutation–selection–drift balance), **not** the Gibbs law $e^{-\alpha_{\rm eff}U}$. The honest phrasing, adopted throughout, is: *$\alpha_{\rm eff}$ is the selection pressure per unit reward; it is inverse-temperature-like dimensionally, but it drives WF selection, not a Gibbs equilibrium.* We recommend the reader treat the word "temperature" in §3 as a units label, not a thermodynamic promise.

**7.3 The load-bearing link — now closed `[PROVEN + NUMERICAL]` (W6, 2026-07-10).** The statement "$\alpha_{\rm eff}$ (§3) $\Rightarrow$ selection coefficient $s$ (§4)" is no longer `[SKETCH]`. It is a **chain rule composing two linearizations of the same clip acceptance**:
$$s_{\rm eff}=\underbrace{2\Phi'(0)}_{\text{clip transmission}}\cdot\underbrace{\alpha_{\rm eff}}_{=\,C\alpha/\sigma_R}\cdot\Delta R+O(\Delta R^3),$$
with (LINK A) `relativize` turning a reward gap into a log-VR gap $\delta=\alpha_{\rm eff}\Delta R$ — the constant is the **population-averaged Jacobian** $C=\mathbb E[g(z)]$, *not* $g(\bar z)$ — and (LINK B) the clip turning the log-VR gap into a frequency drift $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)\approx 2\Phi'(0)\delta$. The transmission constant has a closed form $\Phi'(0)=e^{\tau^2/2}[F(\ln2;\tau^2)-F(0;\tau^2)]$, $\tau^2=2\sigma_v^2$ (only the clip's transition band $0<u<\ln2$ transmits selection). All three links verify to err $\to0$ as $\Delta R\to0$. **The "two effective temperatures" are one selection mechanism in two coordinate systems** (reward→log-VR→frequency); §3 and §4 are not two rivals but two projections of the same object. *Unification* — that $\sigma_v$ is *determined by* `relativize` rather than a free parameter — is confirmed by a coupled relativize+clone simulation: the realized drift matches $\Phi(\delta;\tau)-\Phi(-\delta;\tau)$ with $\tau^2=s_A^2+s_B^2$ read off the population, to within 0.1–0.9%. See MATH_CANON Theorem 4′ and [`W6_CHIUSURA_TEORICA §2`](../wave6_theory_closure/W6_CHIUSURA_TEORICA.md).

---

## 8. Limitations and open problems

1. **The missing theorem (§4.4 point 5) — now closed-form `[DIFF-APPROX; diffusion coefficient PROVEN at leading order]` (W6).** The stationary law of $\mathcal S\circ\mathcal C$ with mutation is the Wright density $\phi_\infty\propto x^{\theta-1}(1-x)^{\theta-1}e^{\sigma x}$; both its drift ($s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$, with $\Phi$ now in **closed form** via normal CDFs) and its diffusion coefficient ($N_e=N/\lambda N$, $\lambda N=2\varphi_0+\langle a_{\rm in}^2\rangle-2\langle a_{\rm in}a_{\rm out}\rangle$, the pairwise **co-ancestry** correction, +12.8% at $\sigma_v{=}0.5$, verified to +0.1%) are closed. The *only* remaining gap to full `[PROVEN]` is the functional diffusion-limit convergence (martingale problem / Lindeberg with the clip's kinks) — standard in the WF literature.
2. **$s_{\rm eff}=2\Phi'(0)\,\alpha_{\rm eff}\,\Delta R$ is now derived, not a heuristic (§7.3, W6).** The weak-selection link between $\alpha_{\rm eff}$ and the selection coefficient is proven as a chain rule of two clip linearizations and verified end-to-end (err $\to0$ as $\Delta R\to0$), including a coupled simulation that pins $\sigma_v$ from `relativize`.
3. **Frozen-per-type VR is an idealization.** Real FMC has *per-tick stochastic* fitness (the simulator perturbs state). We argued the *sign* (fixation, $b_{\rm eff}\to1$) is robust, but the fixation probability of the "average-fittest" under fluctuating fitness is not the deterministic Moran formula; it requires selection-in-random-environment theory, not done here.
4. **The distance term $\rho^{-\beta}$ is not modeled explicitly** in the toy; it was absorbed into the "fluctuating VR" case. An explicit spatial model with density repulsion would give the Theorem-3 anti-collapse correction but does not change the support conclusion. Note also (MATH_CANON Theorem 3 caveat) that raising $\beta$ past a threshold *reduces* $b_{\rm eff}$ — the distance term becomes a selector again, not a repulsor — so the $\alpha/\beta$ dichotomy is not clean (gap G4).
5. **Gap G2 (§2.4).** The Feynman–Kac $L^p$ result is a `[SKETCH]`: the FMC potential is mean-field and stochastic, so the limiting flow $\eta_t$ is a self-referential fixed point of uncertain well-definedness, and pairwise ≤ multinomial variance is asserted, not proven. Any attempt to *rescue* a mean-field Gibbs statement inherits this softness.
6. **Theorem 3 threshold caveat.** The anti-collapse coefficient $\gamma$ is claimed in $(0,1)$ but can leave it; the $\log N$ collapse time (Theorem 3) and the $O(N)$ WF fixation time (Theorem 2′.4) are not reconciled in one framework.
7. **Numerical scope.** All neutral/selection numerics use frozen or i.i.d. VR toys under a single seed (20260709), single VR distribution family per test (LogNormal, Gaussian, uniform). The exponents $q=-0.999$, $p=1.056$, and the constant $C=0.7223$ are reproducible but not multi-seed CI-bounded here.

---

## 9. References

**FMC and project corpus (internal)**
- Hernández-Cerezo, S., Duran-Ballester, G. (2020). *Fractal AI: A Fragile Theory of Intelligence.* arXiv:1803.05049v5. [`1803.05049v5.pdf`]
- Hernández-Cerezo, S., Duran-Ballester, G., Baxevanakis, K. (2018). *Solving Atari Games Using Fractals and Entropy.* arXiv:1807.01081.
- Hernández, S., Duran-Ballester, G., Amigó, J. (2017). *General Algorithmic Search.* arXiv:1705.08691.
- Amigó, J., Balogh, S.G., Hernández, S. (2018). *A Brief Review of Generalized Entropies.* Entropy 20(11):813.
- Wissner-Gross, A.D., Freer, C.E. (2013). *Causal Entropic Forces.* Phys. Rev. Lett. 110:168702.

**Statistical mechanics / SMC**
- Del Moral, P. (2004). *Feynman–Kac Formulae: Genealogical and Interacting Particle Systems.* Springer. (Ch. 2 links particle filters to population genetics; Ch. 7.4.4 $L^p$ bound.)
- Metropolis, N. et al. (1953); Hastings, W.K. (1970). *Metropolis–Hastings.* — the acceptance $\min(r,1)$ FMC does *not* implement.
- Barker, A.A. (1965). *Monte Carlo calculations of the radial distribution functions.* — Barker acceptance $r/(1+r)$.
- Stein, C. (1981). *Estimation of the mean of a multivariate normal distribution.* Ann. Statist. 9. — Stein's identity used in §3.3.

**Population genetics (Moran / Wright–Fisher)**
- Fisher, R.A. (1930). *The Genetical Theory of Natural Selection.* Oxford.
- Wright, S. (1931). *Evolution in Mendelian Populations.* Genetics 16.
- Moran, P.A.P. (1958). *Random processes in genetics.* Math. Proc. Cambridge Phil. Soc. 54.
- Kimura, M. (1955). *Solution of a process of random genetic drift with a continuous model.* PNAS 41(3).
- Ewens, W.J. (1972). *The sampling theory of selectively neutral alleles.* Theor. Pop. Biol. 3.
- Kingman, J.F.C. (1982). *The coalescent.* Stoch. Proc. Appl. 13.
- Wakeley, J. (2008). *Coalescent Theory: An Introduction.* Roberts.

**Internal files (this project)**
- `docs/MATH_CANON.md` — Def. 2 (relativize, ll. 112–138), Def. 3 (VR), Def. 4 (cloning, ll. 160–189), Def. 6 (b_eff), Theorems 1–3 (ll. 229–327), Conjecture A (ll. 334–460).
- `work/14_night_2026-07-09/wave3_validation/W31_stazionaria_corretta.md` — Theorem 2′; `w31_stationary_check.py`.
- `work/14_night_2026-07-09/wave3_validation/W32_alpha_eff.md` — α_eff theorem; `w32_sympy_deriv.py`, `w32_alpha_eff_check.py`.
- `work/02_deep_dives/07_wright_fisher_mapping.md` — WF mapping, $q\approx-0.948$.
- `work/02_deep_dives/01_cloning_mathematics.md` — original Gibbs derivation (retracted §4).
- `work/14_night_2026-07-09/WAVE2_SINTESI.md` — claim triage and gaps G1–G4.

---

*End of draft v0.2 (2026-07-10). All numbers traced to W31/W32/W6 scripts (seed 20260709) or MATH_CANON tables. Proof-status tags are per-statement, not per-section. v0.2 closes the two v0.1 open items: the Wright diffusion coefficient is now closed-form (pairwise co-ancestry, §8.1) and the $\alpha_{\rm eff}\!\to\! s_{\rm eff}$ link is proven (§7.3), both survived an adversarial review. The single remaining open item is the functional diffusion-limit convergence proof (tightness / martingale problem for the clip kernel).*
