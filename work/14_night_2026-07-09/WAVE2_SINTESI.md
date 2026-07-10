# Wave 2 — Sintesi: forza del core FMC, triage dei claim, gap aperti

*Sessione notturna 2026-07-09. Integrazione delle 4 analisi Wave 1 (W1-A core matematico, W1-B evidenza empirica, W1-C corpus Sergio/congetture, W1-D landscape applicativo). Ruolo: research associate + falsificatore.*

---

## 1. La forza del core FMC (cosa lo rende speciale)

Tre proprietà, in ordine di solidità:

1. **Scale-freeness via invarianza affine di `relativize` (Def. 2)** — *l'unica proprietà dimostrata senza buchi*. FMC auto-normalizza la reward sulla popolazione a ogni tick: nessun tuning di scala, e ogni shaping affine-globale è invisibile. Corollario operativo forte: lo shaping efficace **deve** essere tier/moltiplicativo (spiega perché la Cong. D funziona con inv-tier stacking e non con reward additive). → Candidata a proprietà *headline* di un paper teorico.
2. **Resampling pairwise embarrassingly-parallel (Def. 4)** — O(N) confronti locali, nessuna normalizzazione globale dei pesi. Vantaggio architetturale reale su SMC systematic e su MCTS sequenziale. È ciò che dà i 231 LOC NumPy / 7 min / no-GPU su Boxing.
3. **Decisione per marginalizzazione delle etichette** — bincount delle `initial_decision` sopravvissute → decisore discreto senza risolvere Bellman. È l'innovazione propria di FMC rispetto a un particle filter (che stima uno stato, non decide un'azione).

**Sintesi**: FMC = un SMC/Feynman-Kac particle system il cui potenziale è auto-normalizzato (scale-free) e la cui uscita è una decisione discreta per voto delle etichette. Il valore non è "batte MCTS" (claim ancora fragile) ma **"pianificazione per-istanza, zero-training, scale-free, parallela, su qualsiasi simulatore reversibile"**.

---

## 2. Triage dei claim (validato / direzionale / fragile)

### 2.1 Matematica (W1-A, verificato in parte da me)

| Claim | Stato | Nota |
|---|---|---|
| Invarianza affine di `relativize` | ✅ dimostrato | headline |
| **Teorema 2 (detailed balance → Gibbs $\pi^*\propto R^\alpha\rho^{-\beta}$)** | ❌ **overclaim (G1)** | **Verificato da me**: riga 186 afferma clip$(VR_k/VR_i-1)=\min(VR_k/VR_i,1)$; falso per ogni $r\in(1,2)$ (P_clone$=r-1<1$, P_MH$=1$). La regola di cloning NON è MH standard. Con Pr$[y\to x]=0$ (righe 272-273) il detailed balance non chiude. La stazionaria vera non è Gibbs a T finita (contraddetta anche da $b_{\rm eff}\to1\ \forall\alpha>0$). |
| Teorema 1 (convergenza $L^p$, $O(1/\sqrt N)$) | ⚠️ sketch, G2 | Feynman-Kac di Del Moral non si applica direttamente: potenziale mean-field (relativize su $\mu,\sigma$) + stocastico (distanza a partner random). $\eta_t$ non ben definita (punto fisso auto-referenziale). Bound pairwise≤multinomial asserito, non provato. |
| Unicità di `relativize` (P8) | ⚠️ G3 | A1-A5 fissano una *classe asintotica*, non una funzione unica. "Unica" falso come formulato. |
| Teorema 3 (anti-collasso $\beta$) | ⚠️ G4 | $\gamma$ può uscire da $(0,1)$; tempi $\log N$ vs Wright-Fisher $O(N)$ non riconciliati; caveat "$\beta$ alto = di nuovo selettore" (verificato dai dati rocket) mina la dicotomia $\alpha/\beta$ su cui poggia Cong. E. |

**Conseguenza**: il canone dichiara onestamente l'incompletezza (App. B) MA **G1 non è segnalato → è overclaim**. Va corretto.

### 2.2 Empirica (W1-B)

| Claim | Stato |
|---|---|
| **exp17 ≫ baseline v4** (Craftax-Classic-Symbolic) | ✅ **solido**: Wilcoxon p=0.0019, t-appaiato p=0.0030, Cohen $d_z=0.74$, n=18; sblocca iron_pickaxe 33% / iron_sword 11% / diamond 5.6% |
| **FMC risolve Boxing** | ✅ solido: 5/5 seed +100, CPU ~82s |
| Conjecture D (ablation LOO monotòna) | ⚠️ direzionale: ogni tier −4.8/−7.7pp MA nessun test di significatività sui Δ, L1 a n=1, e **l'ablation monotòna non distingue compounding da shaping additivo** (difetto ammesso in `peer_review_self.md`). Serve sweep denso multipli + prova di super-additività |
| FMC vs MCTS (D2) | ⚠️ direzionale: Boxing n=3, MCTS non tunato, 1 gioco. Serve full P0 (420 ep, ~7h CPU, scriptabile) |
| **"50.95% = human-expert 50.5%"** | ❌ **fragile / overclaim**: (i) 50.95% è l'aggregato, la media per-episodio è **30%**; (ii) CI95 = [36.85, 59.46], ±11-13pp; (iii) **non like-for-like** (Craftax-Classic-Symbolic vs Crafter-original a pixel; sull'ambiente umano FMC fa **3.77%**); (iv) aggregato trainato da pochi episodi-eroe |
| "Optimum strutturale a 50.95%" | ❌ fragile: exp17=exp18=exp19 identici → informazione zero su 11 seed fissi, non un attrattore |
| RAM vs IMG (P3) | ❌ fragile: n=2, segnale opposto al paper |

**Storia difendibile**: *"reward shaping tier moltiplicativo raddoppia le prestazioni di FMC vs baseline (p<0.01, d_z=0.74) e sblocca la catena iron→diamond in zero-training"*. Tutto ciò che dice "human-expert" o "legge di compounding" o "optimum 50.95%" **eccede i dati**.

### 2.3 Congetture (W1-C)

| Cong. | Verdetto | Prossimo passo decisivo |
|---|---|---|
| A — branching ~6 | ❌ falsificata come legge / ✅ snapshot contingente | **Pubblicabile** (teorico). Residuo: forma di $\mathcal G(\alpha,K)$; derivare $6=\arg\max_b H(b)$ |
| B — frontiera caos/ordine | 🔴 aperta e in difficoltà (3/3 $\Psi$ compromesse) | $\lambda_1(\delta_0)$ scale-resolved: firma SOC o abbandono |
| C — FMC > DRL su OOD | 🟡 aperta, direzionale (caveat: non a compute pari) | confronto a **budget-campioni fissato** |
| D — chain-tier compounding | 🟡 verificata su 1 task | **replica su 2° benchmark (Procgen)** + super-additività |
| E — self-preservation da entropia causale | ✅ E1/E2/E1-LLM offline verificati; ❌ north-star online falsificata | organo di percezione prima del world-model (via non testata) |

**Intuizioni di Sergio non ancora formalizzate (massimo potenziale teorico)**:
1. **Cross-entropy collapse come definizione di intelligenza** (F12): l'intelligenza non massimizza entropia, rende $P(\text{transizione})\propto R$. Ancorata a Th.2, verificata numericamente (log-Pearson 0.77-0.86 toy). Mai centro di un paper.
2. **$\alpha_{\rm eff}$, bias di temperatura inversa di `relativize`** — scoperta originale, derivata già disponibile, forma chiusa $\alpha_{\rm eff}(\alpha,\sigma_R)$ mancante. Trattabile con `sympy`. *Un teorema che aspetta di essere scritto.*
3. **Frontiera come self-organized criticality** — la $\delta_0$-dipendenza di $\lambda_1$ potrebbe *essere* la firma della criticità.

### 2.4 Applicazioni (W1-D)

**Profilo strutturale ideale** (checklist): HARD = [set_state atomico, step_batch, sim <10ms/step]; FIT = [**divergenza dei walker entro l'orizzonte (E2)** ← filtro decisivo, ripianificazione per-istanza dove il DRL soffre OOD, azioni discrete K∼4-20]; per Cong. D anche [reward sparsa a catena di sub-goal discreti].

**No-fit** (lezione plasma M18): dinamica lineare/convessa (niente divergenza), sim lento (sec/step), no set_state, task memory-dependent.

**Candidati breakthrough 2026 ordinati per (impatto × fit)/costo**:
1. **Logic synthesis (ABC operator sequencing)** — fit altissimo, costo basso, motore FMC-base. Miglior rapporto.
2. **Chip floorplanning/placement** — impatto grezzo massimo (crisi riproducibilità AlphaChip = debolezza che FMC annulla).
3. **Compiler pass ordering** — forte ma tassato da sim 10-100ms (solo offline/vettorizzato).
4. **Quantum circuit compilation** — niche-forte, costo basso (Qiskit/TKET pip-installabili).
5. **Procgen** — scommessa accademica per replica Cong. D.

**Gate universale**: prima di ogni candidato, *smoke test E2* — perturbando l'azione iniziale, i walker divergono entro M step? Se no, stop.

---

## 3. Gap aperti, ordinati per (valore × trattabilità stanotte)

**Alto valore, eseguibile su CPU stanotte:**
- **V1 — Correggere Teorema 2 + derivare la stazionaria vera** (Moran/Wright-Fisher, caso neutrale $\alpha=0$, già confermato empiricamente $q=-0.948$). Prima correggere l'errore MH di riga 186. → *valida e raffina il core*.
- **V2 — Forma chiusa di $\alpha_{\rm eff}(\alpha,\sigma_R)$** con verifica numerica. → *nuovo teorema, potenzia il core*.
- **V4 — Restatement onesto del risultato Craftax** + audit delle incoerenze interne (`statistical_validation.json`). → *onestà per il paper D*.
- **A1 — E2 smoke-test harness** validato su env fmc-core noti (good/bad) → operazionalizza il gate applicativo.

**Alto valore, richiede più compute / dati (candidati Wave 4+):**
- Super-additività Cong. D (sweep denso multipli 1.4→6.67×).
- Full P0 FMC vs MCTS (~7h CPU).
- Replica Cong. D su Procgen.
- Spike breakthrough su un candidato EDA/quantum.

**Teorico di frontiera (paper originale):**
- V3 — Formalizzazione del cross-entropy collapse come definizione operativa di intelligenza (ancorata alla stazionaria corretta di V1).

---

## 4. Decisione per la Wave 3

Lancio 4 subagent Opus in parallelo su lavoro **eseguibile e verificabile** stanotte:
- **W3-1** V1: stazionaria corretta + fix G1 (sympy + numpy Moran/WF).
- **W3-2** V2: forma chiusa $\alpha_{\rm eff}$ (sympy + verifica Monte Carlo).
- **W3-3** V4: restatement onesto Craftax + audit incoerenze (numeri veri dai json).
- **W3-4** A1: E2 smoke-test harness + validazione su env fmc-core + scoping candidato #1.

Ogni subagent produce artefatti verificabili (derivazioni, script, numeri) che io controllo prima di consolidare nel MATH_CANON.
