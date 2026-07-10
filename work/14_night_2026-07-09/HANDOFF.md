# HANDOFF — Sessione notturna 2026-07-09/10 · Validare, raffinare, potenziare il core di FMC

**Direttiva** (`/goal`): analizzare il lavoro FMC con subagent Opus 4.8 (effort medio), identificare la forza del core; validarlo, raffinarlo, potenziarlo; sviluppare le ricerche di Sergio; **completare i paper del professore** con FMC applicato a problemi SW/HW 2026 dove può dare una svolta concreta.

**Metodo**: 5 wave di subagent Opus in parallelo (13 subagent totali), ogni claim con verifica eseguita (sympy/numpy, statistica) e citazioni file:riga. Ruolo assunto: research associate **+ falsificatore**. L'orchestratore ha verificato di persona i rilievi load-bearing (aritmetica MH, derivata α_eff).

---

## 1. Risultato in una frase

Il contributo più forte della notte **non** è "FMC batte un baseline industriale" (non lo fa, allo stato) ma **una fondazione teorica corretta del perché FMC funziona** — che *corregge un overclaim del paper originale* — più **un gate predittivo economico e validato** che dice *a priori dove* FMC-base è competitivo. Due bozze di paper sono pronte.

---

## 2. Core validato e raffinato (→ MATH_CANON v0.8.0)

### 2.1 La forza reale del core (cosa lo rende speciale)
1. **Scale-freeness via invarianza affine di `relativize`** — l'unica proprietà dimostrata senza buchi; auto-normalizza la reward sulla popolazione → nessun tuning di scala, ogni shaping affine-globale è invisibile.
2. **Resampling pairwise embarrassingly-parallel** — O(N) confronti locali, nessuna normalizzazione globale.
3. **Decisione per marginalizzazione delle etichette** — decisore discreto senza risolvere Bellman (l'innovazione vs un particle filter).

### 2.2 Correzioni al canone (mark-not-delete, come per Cong. A)
- **Teorema 2 (Gibbs) RITRATTATO** [W3-1, verificato dall'orchestratore]. L'accettazione FMC è $a_{\mathrm{FMC}}(r)=\operatorname{clip}(r-1,0,1)$, **non** Metropolis $\min(r,1)$ (coincidono solo per $r\ge2$; controesempi $r=0.8,1.5$). Uphill-only ⇒ non reversibile ⇒ nessuna Gibbs a supporto pieno; cloning-only ⇒ massa puntuale ($b_{\rm eff}\to1$). Corretto l'errore a `MATH_CANON` Def. 4 riga 186 e in deep dive 01 §4.
- **Teorema 2′ (nuovo)** — selezione Moran/Wright-Fisher: con selezione il fittest fissa con **prob. 1**; caso neutrale drift $q=-1.018$ CI$[-1.033,-1.003]$, $p=+1.025$ (25 seed, fitness fluttuante). $\alpha$ = intensità di selezione, non temperatura termodinamica.
- **Teorema 2′.5 (nuovo, [DIFF-APPROX verificata])** — legge stazionaria con mutazione = **distribuzione di Wright** $\varphi_\infty\propto x^{\theta-1}(1-x)^{\theta-1}e^{\sigma x}$, coefficienti dalla vera accettazione uphill-only; TV→0.016 a N=800. Residuo aperto: correzione **+13%** di $N_e$ (co-ancestry pairwise) → chiuderla la promuove a [DIM].
- **Teorema 4 (nuovo) — temperatura inversa effettiva** $\boxed{\alpha_{\rm eff}=C\,\alpha/\sigma_R}$ [W3-2, [DIM] pointwise + [NUM] ≤0.29%, $C_{\rm gauss}=0.7225$]. Tre corollari: **annealing emergente** (σ_R↓→pressione↑, aggancio quantitativo a Cong. B/D3), **incomparabilità di α tra benchmark**, **shaping obbligatoriamente moltiplicativo-tiered** (spiega meccanicamente Cong. D).

---

## 3. Onestà empirica (→ correzione CLAUDE.md + memoria)

**Overclaim ritrattato** [W3-3, numeri da json reali]: "50.95% = human-expert 50.5%" è insostenibile as-is (50.95% = aggregato del run n=11; media per-episodio 30%; CI95 ±11-13pp; non like-for-like — su Crafter-original a pixel FMC fa 3.77%). Incoerenza JSON confermata (`statistical_validation.json:63-64` dice ~50.5% per-seed mentre il campo reale riga 24 è 30.04%).

**Claim difendibile e forte**: exp17 vs baseline v4 = **+22.1pp appaiato**, Wilcoxon $p=1.9\times10^{-3}$, $d_z=0.74$, n=18; sblocca iron→diamond zero-training. Per "human-expert" servono: like-for-like su Crafter-original a compute pieno, contrasto 2×2 per la super-additività, replica Procgen.

---

## 4. Applicazioni 2026 (Wave 4) — spike reali, gated da E2

**Gate E2 di divergenza** [W3-4]: `disp_ratio = disp_M/disp_1 ≥ 3.0` → FMC-fit. Validato 6/6 su control (Rocket 24.9 … lineari 1.9-2.4); la soglia cade al confine di stabilità A≈0.93.

| Dominio | Verdetto E2 (a priori) | Esito reale FMC vs baseline | Concordanza |
|---|---|---|---|
| Control (Rocket/Pendulum/Nav/CartPole) | DIVERGE | FMC ≫ random | ✅ |
| Quantum routing linear5 | DIVERGE (3.05) | pari/batte SABRE di poco (−0.57 SWAP, p=0.011) | ✅ |
| Quantum routing grid3x3 | COLLAPSE (2.94) | perde 25/28 (p<1e-4) | ✅ |
| Logic synthesis (10 circuiti AIG) | COLLAPSE (1.3-2.3) | pareggia greedy (+11.4 vs +11.9%), 37× costo | ✅ |
| Plasma TCV (retrospettivo) | COLLAPSE | null result (M18) | ✅ |

**Verdetto applicativo onesto**: nessun breakthrough as-is. FMC-base è competitivo *solo* nel regime E2-fit e lì *solo alla pari* a costo 100-300× (quantum) / 37× (synthesis). Il candidato #1 del landscape survey (logic synthesis) ha **fallito il proprio gate E2** — falsificazione di un fit sopravvalutato. **Il valore emerso è la validità predittiva cross-dominio del gate E2**: uno strumento di triage che avrebbe evitato lo spreco plasma-M18.

Cause dei "pari": env troppo naive (quantum in-order, non sfrutta la commutazione del DAG come SABRE; synthesis con operatori quasi-deterministici → reward-plateau → swarm non diverge). Percorsi verso un vero breakthrough (registrati nei report W4A/W4B): env con DAG front-layer + device reali + budget GPU (quantum); landscape rugoso EPFL-hard + technology mapping area×delay + operatori stocastici (synthesis).

---

## 5. Deliverable prodotti

- **MATH_CANON v0.8.0** — Teorema 2 ritrattato, 2′/2′.5/4 aggiunti, changelog + indice aggiornati.
- **Paper teorico**: [`wave5_papers/PAPER_THEORY_effective_temperature_FMC.md`](wave5_papers/PAPER_THEORY_effective_temperature_FMC.md) — "Scale-Free Selection, Not Gibbs Equilibrium". 3 contributi (α_eff, Moran/WF retraction, magic-6 transient) + risoluzione della tensione a due regimi. **È il "completamento" del paper del professore**: fornisce la meccanica statistica corretta e corregge gli overclaim originali.
- **Paper sistemistico**: [`wave5_papers/PAPER_SYSTEMS_divergence_gate_FMC.md`](wave5_papers/PAPER_SYSTEMS_divergence_gate_FMC.md) — "A Cheap A-Priori Divergence Gate Predicts Where FMC Planning Helps".
- **Codice eseguito** in `wave3_validation/`: `w31_stationary_check.py`, `w32_alpha_eff_check.py`, `w3b_robustness.py`, `w3b_mutation_diffusion.py`, `w34_e2_smoke.py`; in `wave4_applications/`: `w4a_quantum_routing.py`, `w4b_logic_synthesis.py`.
- **Analisi corpus** in `wave1_corpus/` (W1A-D) + `WAVE2_SINTESI.md`.
- **Tooling installato**: `fmc-core` (editable), `qiskit` 2.5.0 (SABRE), `aigverse` 0.1.1.

---

## 6. Prossimi passi (ordinati per valore/costo)

**Teoria (chiudono buchi noti dei paper):**
1. Correzione +13% di $N_e$ (co-ancestry pairwise) → promuove Teorema 2′.5 a [DIM]. Collega al lavoro Wright-Fisher del branching (Cong. A).
2. Ponte α_eff↔s_eff in un'unica derivazione (chiude la tensione §7 del paper teorico).
3. Estensione a K tipi → Ewens (1972).

**Empirica (elevano Cong. D a legge / il claim a human-expert):**
4. Like-for-like su Crafter-original a compute pieno (N=512, M=40, 30 seed, appaiato).
5. Contrasto 2×2 (inv-tier ⊕ ach-fire) a n=30 per la super-additività.
6. Replica Cong. D su Procgen (2° family).

**Applicazioni (verso un vero breakthrough):**
7. Quantum routing v2: env con DAG front-layer + SABRE completo + device IBM reali + budget GPU `fragile`, obiettivo depth/error-rate.
8. Logic synthesis v2: benchmark EPFL-hard + technology mapping (reward non-monotòno) + operatori stocastici; ri-passare E2 prima.
9. Nuovi candidati dal landscape (W1D): chip placement (impatto max, crisi AlphaChip). **Regola d'oro: smoke test E2 PRIMA di ogni investimento.**

**Nota di processo**: due subagent Wave 5 sono caduti sul limite di sessione (reset 1am Europe/Rome) ma avevano già scritto i file; l'orchestratore ha completato canone/CLAUDE/memoria manualmente.
