# W1-D — Landscape applicativo di FMC: dove eccelle strutturalmente, dove no

> **Wave 1 / Sessione notturna 2026-07-09.** Autore: research associate + scettico (Opus 4.8, effort medio).
> **Scope**: mappare le proprietà strutturali che fanno eccellere o fallire Fractal Monte Carlo (FMC), poi
> filtrare i domini SW/HW 2026 dove una svolta concreta è plausibile. Distinzione netta tra **fit strutturale
> forte** (dedotto da meccanismo + evidenza) e **speculativo** (plausibile ma non verificato).
>
> **Fonti primarie lette**: `CLAUDE.md` §"Project briefing"; `DominiDaIndagare.md`;
> `work/06_plasma_fmc/README.md` + `m18_hierarchical/docs/iteration_1_alert.md` (il fallimento plasma);
> `docs/architecture/tier1_repos_teardown.md` (capacità stack fragile/plangym);
> `docs/MATH_CANON.md` §Congettura D; `work/05_craftax/autoresearch/HANDOFF.md`.

---

## 0. Premessa metodologica: due value proposition distinte, non una

L'errore più comune nel discutere "dove applicare FMC" è trattarlo come un blocco monolitico. Il corpus mostra
che ci sono **due meccanismi separabili**, con fit strutturali diversi:

- **(A) FMC-base** — planner a sciame che massimizza entropia causale (`F = T_c ∇_X S_c`, limite discreto di
  Wissner-Gross) su un simulatore reversibile, **senza training**, **ri-pianificando per-istanza**. Il vantaggio
  competitivo è: *nessun gap train/test perché non c'è training* (`DominiDaIndagare.md:82`, `:132`).
- **(B) Congettura D** — ricetta di reward shaping (inv-tier denso + ach-fire sparso, tier-weighted, sweet-spot
  amplificazione 1.2–1.4×) per task a **chain gerarchica di sub-goal discreti** (`MATH_CANON.md:527-539`).

Il fallimento plasma (M18) e il successo Craftax mostrano che **(A) e (B) hanno prerequisiti diversi**. Confondere
i due porta a proporre Congettura D dove serviva solo FMC-base (è esattamente l'errore di M18). Tengo questa
distinzione per tutto il documento.

---

## 1. Proprietà strutturali: il profilo ideale (e come si rompe)

### 1.1 Prerequisiti HARD (senza questi FMC non esiste proprio)

Dallo stack teardown (`tier1_repos_teardown.md:11`, `:16`, `:50-64`):

| # | Prerequisito | Perché è non-negoziabile | Fonte |
|---|---|---|---|
| H1 | **Simulatore con `set_state()`/`get_state()` atomico** | FMC clona i walker: deve poter *ripristinare* uno stato arbitrario e ramificare da lì. È letteralmente "ciò che fa esistere FMC". | `tier1_repos_teardown.md:11`, `:50-57` |
| H2 | **`step_batch` vettorizzabile su N walker** | Ogni decisione = N×M chiamate al simulatore. Il call-site `fragile/core.py:716` è `env.step_batch(...)`. | `tier1_repos_teardown.md:16`, `:62` |
| H3 | **Sim veloce: < 10 ms/step, idealmente < 1 ms** | N×M chiamate per azione. A 50 ms/step, N=30 M=20 = 30 s/decisione → solo offline. | `DominiDaIndagare.md:5`, `:142` |

**Implicazione operativa**: se il "problema" è nel mondo reale senza gemello digitale reversibile (es. controllo
kHz in tempo reale di MHD plasma, `project_fusion_transfer.md`: "Real-time control … is OUT of scope"), FMC-base
è escluso a priori. Serve un simulatore, e serve che sia snapshottabile e veloce.

### 1.2 Proprietà che fanno ECCELLERE FMC (fit forte)

| # | Proprietà | Meccanismo | Evidenza |
|---|---|---|---|
| E1 | **Spazio azioni discreto o gestibile** (K∼4–20) | Il campionamento delle perturbazioni è naturale su azioni discrete; Atari K∼18, Craftax K=17. | `MATH_CANON.md:576`; Atari repl. |
| E2 | **Dinamica non-lineare / landscape rugoso → divergenza dei walker entro l'orizzonte T** | **È LA condizione critica.** FMC funziona *solo se* walker diversi finiscono in stati qualitativamente diversi: la `relativize` standardizza il reward, e se tutti i walker convergono la varianza è ~0 → il segnale sparso viene azzerato → cloning = random. | `m18_hierarchical/docs/iteration_1_alert.md:38-59` |
| E3 | **Il valore è nella ri-pianificazione per-istanza (OOD documentato per il DRL)** | FMC ri-pianifica da zero su ogni istanza nuova → "gap train/test = 0 by construction". Dove il DRL soffre OOD, FMC non ha "OOD" perché non ha "ID". | `DominiDaIndagare.md:82`, `:132` |
| E4 | **Reward sparsa ma strutturata a catena di sub-goal (per Congettura D)** | Chain wood→stone→iron→diamond: inv-tier denso + ach-fire sparso danno compounding monotonico. | `MATH_CANON.md:527-553` |
| E5 | **Esplorazione critica + sopravvivenza** (α→0 = Common Sense) | La massimizzazione dell'entropia causale tiene l'agente "vivo" mentre esplora, senza reward di sopravvivenza (Congettura E1). | `MATH_CANON.md:596-601`; `DominiDaIndagare.md:108` |
| E6 | **Orizzonte plannabile** (il sub-goal chiave è entro ∼M step dal presente) | FMC pianifica τ avanti; se il payoff è entro l'orizzonte, lo vede. | `DominiDaIndagare.md:117-118` |

### 1.3 Proprietà che fanno FALLIRE FMC (no-fit) — dedotte dal caso plasma

Il fallimento M18 su plasma è la miglior lezione negativa che abbiamo. Non fu un bug: fu una **incompatibilità
strutturale prevista dalla congettura stessa** (`iteration_1_alert.md:55-59`). I fallimenti sono l'immagine
speculare di E1–E6:

| # | Anti-proprietà | Cosa succede | Fonte |
|---|---|---|---|
| F1 | **Dinamica quasi-deterministica / lineare / convessa** | I walker convergono tutti sulla stessa traiettoria di gradiente entro pochi tick → **nessuna divergenza** → `relativize` azzera il segnale → FMC ≈ random. Il sim plasma M2 (lineare) fa esattamente questo anche con voltage_std=200V. | `iteration_1_alert.md:48-53` |
| F2 | **Achievement a soglia su stato continuo** | La soglia "err < 20" scatta per *tutti* i walker quasi-simultaneamente → fire_bonus uniforme → contributo zero a `relativize`. Non c'è sparsità. | `iteration_1_alert.md:54-59` |
| F3 | **Sim lento (> 10–100 ms/step)** | N×M esplode; solo offline, mai interactive. (Docking Vina, Rosetta, EnergyPlus, HVAC: sec/step → scartati.) | `DominiDaIndagare.md:50-52`, `:71` |
| F4 | **Nessun `set_state` disponibile** (mondo reale, dati offline-only) | FMC non può ramificare. Sepsi/ICU/finanza-proprietaria: solo dati storici → scartati. | `DominiDaIndagare.md:58-64` |
| F5 | **Bottleneck spatial-reach oltre l'orizzonte** | Se il sub-goal non è raggiunto nei rollout di lunghezza M, *nessuno* shaping di reward aiuta (Falsifica 5). Serve memoria cross-episode o macro-azioni. | `MATH_CANON.md:561`, `:578`; `HANDOFF.md:145-163` |
| F6 | **Task memory-dependent** | FMC vanilla è memoryless → Procgen Maze/Heist (memoria richiesta) sono rischiosi. | `DominiDaIndagare.md:90` |
| F7 | **Azione continua ad alta dimensione senza story sim2real** | Adattabile (plasma M3, `fmc_plasma.py`) ma senza vantaggio competitivo dove il DRL locomotion/manipulation ha saturato l'agenda. | `DominiDaIndagare.md:15-16` |

> **La regola operativa che riassume tutto** (test di ammissione in ordine): (H) esiste un simulatore
> reversibile e veloce? → (E2/F1) perturbando l'input i walker *divergono* qualitativamente entro T? →
> (E3) c'è un DRL/heuristic incumbent che soffre OOD/generalizzazione? → (E4/F2) [solo per Cong. D] la reward
> è una catena sparsa di sub-goal *discreti* e non una soglia su continuo? Se un problema fallisce E2/F1,
> **FMC non parte proprio**, per quanto attraente sia il resto.

---

## 2. Inventario domini SW/HW 2026

Legenda fit: 🟢 forte · 🟡 moderato/condizionato · 🔴 no-fit · (spec) = speculativo, non verificato.
Colonne: **Sim rev.** (H1) · **Azioni** (E1) · **Divergenza walker** (E2/F1, il filtro killer) ·
**Reward-chain** (E4, rilevante per Cong. D) · **Baseline da battere + OOD noto** (E3).

| Dominio | Sim reversibile? | Azioni | Divergenza (E2) | Reward-chain (E4) | Baseline / OOD | Fit |
|---|---|---|---|---|---|---|
| **Chip floorplanning / macro-placement** | 🟢 stato = coord. macro, trivialmente snapshottabile; proxy wirelength/congestion veloce | 🟢 discreto su griglia (Google gridizza); posa 1 macro/step | 🟢 posare A vs B presto → layout globalmente diversi (landscape non-convesso) | 🟡 reward per lo più mono-obiettivo (wirelength), shaping possibile ma non chain "naturale" | AlphaChip (RL) sotto **crisi di riproducibilità** (Cheng et al. 2023); analitici RePlAce/DREAMPlace; SA | 🟢 (spec forte) |
| **Logic synthesis (ABC pass sequencing)** | 🟢 AIG snapshottabile; trasformazioni (rewrite/refactor/balance/resub) ms–100ms | 🟢 discreto (∼10–20 op ABC) | 🟢 phase-ordering NP-hard, non-monotono → divergenza | 🟡 riduzione nodi/area mono-obiettivo (come compiler) | resyn2/compress2 hand-tuned; DRiLLS (RL); OOD cross-circuit noto | 🟢 (spec forte) |
| **Compiler pass ordering (CompilerGym/LLVM)** | 🟢 IR + sequenza pass deterministica | 🟢 discreto ∼100–124 pass | 🟢 interazioni pass non-monotone → divergenza | 🟡 code-size riduzione mono-obiettivo | MLGO in produzione (Google); OpenTuner; **OOD esplicito** (Cummins et al. CGO 2022) | 🟢 ma tassato da **F3**: 10–100 ms/step → solo offline |
| **Quantum circuit compilation (routing/transpile)** | 🟢 DAG gate snapshottabile; SWAP/decomp veloci | 🟢 discreto (quale SWAP, quale decomposizione) | 🟢 scelte di routing → circuiti divergenti | 🟡 riduzione gate 2-qubit / depth | Qiskit SABRE, TKET (euristici, battibili); campo in crescita | 🟢 niche |
| **Datacenter/job scheduling (Park/Decima)** | 🟢 replay trace <1ms | 🟡 discreto ma combinatorio | 🟡 dipende dal workload; spesso convesso-ish | 🟡 throughput/latency, poco chain | Decima (RL); community sistemi non-ML | 🟡 (già "forte" in `DominiDaIndagare.md:41`) |
| **Adaptive Bitrate Streaming (Pensieve/Park)** | 🟢 replay trace <1ms, state ∼10-dim | 🟢 discreto ∼6 bitrate | 🟡 MDP pulito ma dinamica mite | 🔴 no | Pensieve (RL) degrada su trace OOD | 🟡 niche, MDP piccolo |
| **Test generation / coverage-guided fuzzing** | 🟡 snapshot dello stato del processo a metà esecuzione è *awkward* (concolic sì, generico no) | 🔴 mutazioni input: enorme, non strutturato | 🟢 input diversi → path diversi | 🟢 coverage = catena di eventi rari dietro guardie (magic value) — strutturalmente Craftax-like! | AFL++/libFuzzer/KLEE; ML-fuzzers | 🟡 (spec) — killer è H1 (snapshot processo) |
| **Program synthesis (DSL, I/O examples)** | 🟢 AST parziale snapshottabile; esecuzione su esempi = reward, veloce se DSL piccolo | 🟡 produzioni grammaticali, discreto ma variabile | 🟡 dipende dal DSL | 🟢 frazione esempi passati = credito parziale a catena | Enumerative + DeepCoder/DreamCoder; **LLM dominano 2026** | 🟡 (spec) — saturato da LLM |
| **RTL design / HLS da spec** | 🟡 AST snapshottabile ma sintesi/Verilator lento-ish | 🔴 generazione, non strutturata | — | — | Territorio LLM; come *logic synth* ricade su riga 2 | 🔴 come task generativo; 🟢 solo se riformulato come synthesis-search |
| **Robotica model-based (MuJoCo via plangym)** | 🟢 `mjSTATE_INTEGRATION` dà set_state (`tier1_repos_teardown.md:85`) | 🔴 continuo alta-dim | 🟡 sì ma | 🔴 locomotion è reward densa, non chain | RT-X/ETH/MIT saturano; no story sim2real | 🔴 (`DominiDaIndagare.md:15-16`) |
| **Conformer search molecolare (RDKit/MMFF)** | 🟢 coord. molecola, ms/step | 🟡 torsioni: continuo | 🟢 energy landscape multimodale → divergenza | 🟡 energia | Conformer-RL; metrica RMSD non-euclidea → rischio "embedding learnt" | 🟡 (`DominiDaIndagare.md:174-181`) |
| **Docking / protein design / crystal (Rosetta, Vina)** | 🟢 ma | — | — | — | — | 🔴 **F3**: sec/step |
| **Cache / prefetch policy** | 🟢 replay trace, nano | 🟡 reattivo per-accesso, non plannabile bene | 🔴 poco | 🔴 no | bench debole | 🔴 (`DominiDaIndagare.md:45`) |
| **Insulin dosing (UVA/PADOVA T1D)** | 🟢 ODE, ms/step | 🟢 dose discretizzabile | 🟡 ODE accoppiato: dinamica mite (rischio F1!) | 🔴 no | Fox et al. RL; OOD paziente; impatto medico | 🟡 impatto alto ma review severi + rischio F1 |
| **Procgen Benchmark** | 🟢 ∼1 ms | 🟢 discreto | 🟢 procedurale, non-lineare | 🟡/🟢 alcuni giochi hanno chain | PPO/Rainbow gap OOD 50–60% (`DominiDaIndagare.md:80`) | 🟢 (TOP accademico) |
| **Crafter / Craftax** | 🟢 ∼5 ms | 🟢 K=17 | 🟢 verificato | 🟢 chain 22 achievement (Cong. D provata qui) | DreamerV3 SOTA | 🟢 **già dimostrato 50.95%** |

---

## 3. I candidati breakthrough 2026 — filtro finale

Ordino per **(impatto potenziale × fit strutturale) / costo di validazione**. Nota di rigore su cui insisto:
per la famiglia EDA/compiler/quantum il motore competitivo è **FMC-base (A)**, non Congettura D — la reward è
per lo più un singolo obiettivo continuo (dimensione/wirelength/gate-count), non una catena gerarchica di
sub-goal discreti. La forza è **online planning + no-training + no-OOD su landscape rugoso**. Per i benchmark
game (Procgen/Crafter) il motore è **(A)+(B)** insieme. Sono onesto: chiamare "Congettura D" un successo EDA
sarebbe un abuso — è FMC-base che vince lì.

### 🥇 Candidato 1 — Logic synthesis (ABC operator sequencing)

- **Perché fit forte (A)**: AIG snapshottabile, azioni discrete (∼10–20 operatori), sim ms-scale (più veloce di
  CompilerGym), landscape di phase-ordering NP-hard e non-monotono → **divergenza dei walker garantita** (E2 ✓,
  il filtro che ha ucciso il plasma qui è soddisfatto). Baseline = script hand-tuned `resyn2`/`compress2`,
  battibili; OOD cross-circuit documentato in letteratura DRL (DRiLLS et al.).
- **Perché svolta non incrementale**: ogni chip passa per la sintesi logica; un ottimizzatore *per-netlist*
  senza training che batte gli script fissi tocca un collo di bottiglia industriale reale.
- **Costo di validazione**: **BASSO–MEDIO**. ABC è open-source, veloce; benchmark EPFL/ISCAS/MCNC sono lo
  standard de-facto. Serve un `PlanEnv` che wrappi ABC (i 4 metodi, ∼100–150 LOC, `tier1_repos_teardown.md:90`)
  + reward = node/depth reduction.
- **Rischio principale**: reward mono-obiettivo → non sfrutta Cong. D; va dimostrato che FMC-base batte
  gli script su un set ampio (>10 circuiti), non su cherry-pick.
- **Confidenza**: fit **forte (spec)** — meccanismo solido, non ancora eseguito.

### 🥈 Candidato 2 — Chip floorplanning / macro-placement

- **Perché fit forte (A)**: stato = coordinate macro, **snapshot triviale** (nessun problema di serializzazione
  come Box2D); azione discreta su griglia (posa sequenziale una macro/step); proxy reward
  (wirelength HPWL + congestion) veloce; landscape fortemente non-convesso → divergenza ✓.
- **Perché svolta non incrementale + timing 2026**: l'incumbent RL (AlphaChip, Nature 2021) è sotto **crisi di
  riproducibilità** pubblica (Cheng et al. 2023, "That Chip Has Sailed"). La critica chiave — *serve
  pre-training su molti chip* — è esattamente la debolezza che FMC annulla: **niente training, si pianifica per
  netlist**. Il pitch "no-training online planner batte l'RL pre-addestrato e contestato" è di forte richiamo.
- **Costo di validazione**: **MEDIO**. Esistono env aperti (MacroPlacement/Cheng et al. ha open-sourcato l'env
  Google-like; DREAMPlace; benchmark ISPD). Serve integrare proxy reward + gestire l'orizzonte lungo (molte
  macro) — attenzione a F5 (spatial-reach) se il numero di macro >> M.
- **Rischio principale**: orizzonte (numero macro) può eccedere M → serve placement sequenziale con reward
  intermedia densa; verificare che non degeneri in F1 su reward troppo liscia.
- **Confidenza**: fit **forte (spec)**; **impatto grezzo il più alto** dei cinque.

### 🥉 Candidato 3 — Compiler pass ordering (CompilerGym / LLVM)

- **Perché fit forte (A)**: già marcato TOP in `DominiDaIndagare.md:39`, `:121-144`. Discreto (∼100 pass),
  reversibile, OOD **esplicitamente documentato** (Cummins et al. CGO 2022), MLGO in produzione da battere.
- **Il tasso da pagare (F3)**: sim 10–100 ms/step → N×M esplode → **solo offline compilation**, mai interactive.
  Va vettorizzato (`DominiDaIndagare.md:142`, `:208`). Questo abbassa il rapporto costo-aggiustato sotto logic
  synthesis (che ha lo stesso profilo ma sim più veloce).
- **Costo di validazione**: **MEDIO** (CompilerGym mantenuto da Meta, ma la lentezza del sim impone
  parallelizzazione seria).
- **Confidenza**: fit **forte** sul meccanismo, penalizzato dalla velocità del simulatore.

### 4 — Quantum circuit compilation (qubit routing / transpilation)

- **Perché fit (A) niche-forte**: DAG del circuito snapshottabile, azioni discrete (inserzione SWAP,
  decomposizione gate), sim veloce, landscape rugoso → divergenza ✓. Baseline = euristici SABRE (Qiskit)/TKET,
  battibili. Campo 2026 in crescita.
- **Costo di validazione**: **BASSO** (Qiskit/TKET open, benchmark circuiti standard).
- **Rischio/limite**: community più piccola → **impatto medio** (svolta "niche breakthrough"), non industriale
  di massa come EDA.
- **Confidenza**: fit **forte (spec)** su meccanismo; impatto contenuto.

### 5 — Procgen (+ Crafter come secondario) — il candidato "paper sicuro"

- **Perché fit (A)+(B)**: è il candidato più vicino al lavoro già dimostrato (Crafter 50.95%,
  `CLAUDE.md`/`MATH_CANON.md:539`). Procgen è **il** benchmark per il pitch "gap train/test = 0 by construction"
  (`DominiDaIndagare.md:74-93`). Ri-testerebbe **Congettura D su un secondo task** — cosa che MATH_CANON chiede
  esplicitamente come criterio di falsificabilità (`MATH_CANON.md:565-567`).
- **Perché non incrementale (in senso scientifico)**: chiude/apre la questione se Cong. D è *legge generale* o
  *descrittiva* — alto valore epistemico per completare il paper del professore.
- **Costo di validazione**: **BASSO–MEDIO** (port Procgen→plangym; sim ∼1–5 ms).
- **Rischi**: Maze/Heist richiedono memoria (F6) → FMC vanilla può perdere lì; servono sim parallele.
- **Confidenza**: fit **forte**; impatto **accademico** (workshop garantito, main-track possibile), non
  industriale. È la scommessa a **rischio più basso** e la più coerente con lo stato attuale del progetto.

---

## 4. Sintesi del giudizio

1. **Il filtro che decide tutto è E2/F1 (divergenza dei walker).** Il fallimento plasma non fu sfortuna: fu una
   dinamica lineare/convessa che nega la divergenza. Ogni candidato va pre-testato con uno smoke check:
   *perturbando l'azione iniziale, i walker finiscono in stati qualitativamente diversi entro M step?* Se no,
   fermarsi.
2. **La famiglia vincente 2026 è combinatoria-sequenziale su landscape rugoso con simulatore reversibile e
   veloce**: EDA (logic synthesis, placement), compiler pass ordering, quantum compilation. Qui vince
   **FMC-base**, non Congettura D. Sono onesto su questa distinzione.
3. **Congettura D si applica ai task chain-sparse discreti** (Crafter provato; Procgen, program synthesis con
   esempi staged, fuzzing-coverage come candidati speculativi). Fuori da questo regime è irrilevante o dannosa
   (M18).
4. **Miglior rapporto (impatto × fit)/costo**: **logic synthesis** (fit altissimo, costo basso, impatto alto) e
   **chip placement** (impatto grezzo massimo grazie alla crisi AlphaChip, costo medio). **Quantum** è la
   scommessa niche a costo basso. **Procgen** è la scommessa accademica sicura che chiude la falsificabilità di
   Cong. D. **Compiler** è forte ma tassato dalla velocità del simulatore.
5. **Prerequisito comune non aggirabile**: per ognuno serve un `PlanEnv`/wrapper con `set_state`/`step_batch`
   (∼100–150 LOC, `tier1_repos_teardown.md:90`, `:186-201`). Nessun candidato "salta" questo passo — ed è anche
   la parte più economica.
