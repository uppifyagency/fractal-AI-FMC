# HANDOFF — Programma research-partner / Congettura E

> **Per il prossimo agente AI che riprende questo lavoro.**
> Leggi questo file *per intero* prima di toccare qualsiasi cosa.
> Aggiornato: 2026-05-21.

---

## TL;DR — stato attuale

Questo è il **programma di ricerca research-partner** su FMC. Modello operativo:
**l'utente è il PI** (principal investigator, dà direzione e taste), **Claude è il
research associate _e_ lo scettico** — il cui compito include falsificare le idee
preferite del PI con lo stesso rigore con cui il repo ha falsificato i claim di
Sergio (vedi MATH_CANON, Cong. A "magic 6" falsificata).

**Stella polare**: la **Congettura E** ([`docs/MATH_CANON.md`](../../docs/MATH_CANON.md#congettura-e--self-preservation-emergente-da-entropia-causale)) —
FMC come *core agentico* + LLM come *organo* (percezione/world-model/azione/voce).
La self-preservation emerge dall'entropia causale; "desiderio" e "preservazione"
sono gli esponenti α e β del virtual reward.

**Progresso Congettura E**: 3 test previsti.
- **E1-base** ✓ verificata (2026-05-20) — vedi [`RESULT.md`](RESULT.md).
- **E2** ✓ verificata con refinement (2026-05-20) — vedi [`E2_RESULT.md`](E2_RESULT.md).
- **E1-robustness** ✓ caveat di geometria respinto (2026-05-20) — vedi
  [`E1_ROBUSTNESS_RESULT.md`](E1_ROBUSTNESS_RESULT.md). Non un nuovo test E:
  chiude il caveat "lava isolata" di E1-base.
- **E1-LLM** ✗ non eseguita — ma **sbloccata**: il proxy P13 è stato eseguito
  (2026-05-21) e dà **GO-conditional**. E1-LLM è eseguibile a condizione che
  l'LLM-world-model modelli correttamente gli stati terminali — vedi
  [`P13_RESULT.md`](P13_RESULT.md).

**Direzioni teoriche / esperimenti chiusi (2026-05-21)**:
- **P13** ✓ design + **proxy eseguito** — [`P13_DESIGN.md`](P13_DESIGN.md), [`P13_RESULT.md`](P13_RESULT.md), [`p13_proxy.py`](p13_proxy.py).
- **hP13-0** ✓ rifatto (griglia η fine) — ancora inconclusivo ma diagnosticato: [`P13_HP13_0_RESULT.md`](P13_HP13_0_RESULT.md).
- **H-B1a** ✓ eseguito — harness λ₁: [`../13_chaos_order/HB1A_RESULT.md`](../13_chaos_order/HB1A_RESULT.md). λ₁ **non scale-free**.
- **E1-LLM** ✓ pre-registrato — [`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md) (esecuzione del cuore bloccata su API LLM).
- **Deep-dive 09** ✓ frontiera caos/ordine; **Deep-dive 02** ✓ Active Inference (fondazione del merge).

MATH_CANON è a **v0.7.2**. Memoria di sessione: i file in
`~/.claude/projects/.../memory/` (`project_research_partner_program.md`,
`feedback_tooling_decisions.md`) — caricati automaticamente all'avvio.

---

## Verifica lo stato prima di iniziare (1 min)

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
git branch --show-current                       # autoresearch/exp02-ach-bonus
grep -n "Congettura E" docs/MATH_CANON.md        # deve esistere in Parte IV
ls work/12_conjecture_e/results/                 # e1_base.json, e2_*.{csv,json,png}
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" -c "import numpy,scipy,statsmodels,pandas,matplotlib; print('stack OK')"
```

---

## Cosa è stato fatto

### Congettura E — formalizzazione (MATH_CANON v0.6.0)
La direzione 5 del programma è stata resa una congettura falsificabile: E1
(self-preservation senza reward di sopravvivenza), E2 (separazione funzionale
α=desiderio / β=preservazione), sotto-domanda di fattibilità P13 (muro N·M
chiamate-LLM). Mappa canonica: α↔Wissner-Gross goal-term, β↔forza entropica
causale; α=0 = "Common Sense" (Def. 3).

### E1-base — verificata (MATH_CANON v0.6.1)
Gridworld 2-D con lava **assorbente**, reward = −manhattan ovunque (nessuna
penalità lava, nessun bonus sopravvivenza), kernel `fmc-core` invariato.
**FMC α∈{0,0.1} → 0% morte su 3 layout** vs random 85–100% / greedy 100%,
p<0.001 (z da −5.4 a −6.3). Twist: α=1 sul layout *lake* (goal dietro la lava)
muore al 100%.

### E2 — verificata con refinement (MATH_CANON v0.6.2)
Sweep fattoriale **6 α × 4 β × 3 layout × 60 = 4320 episodi**, disegno
**pre-registrato** ([`E2_DESIGN.md`](E2_DESIGN.md)), statistica con la skill
`statistical-analysis`.
- H1 (morte↑α) ✓ z=+12.5; H2 (goal↑α) ✓ z=+20.3; H3 (morte↓β) ✓ z=−13.4 —
  tutte p_holm<10⁻³⁴.
- **H4 (goal↓β) FALSIFICATA** — z=−0.63, p=0.53. β **non** costa goal.
- H5 separazione: **asimmetrica** — η²_α(goal)=0.91 (α possiede il goal),
  OR_β(morte)=0.48 / OR_β(goal)=0.94 ns (β = sicurezza quasi gratuita),
  η²_interazione(survive)=0.50.
- Frontiera di Pareto interamente a α≤0.5 / β≥1; ottimo bilanciato α=0.5, β=2.0.
- Bonus: α=0,β=0 → 79% morte = conferma empirica del **Teorema 3** (anti-collasso).
- 4 figure (2 tecniche, 2 divulgative) — vedi [`E2_RESULT.md`](E2_RESULT.md).

### E1-robustness — caveat di geometria respinto (MATH_CANON v0.6.3)
E1-base usava lava *compatta*; il caveat (in RESULT.md ed E2_RESULT.md) temeva
che lava **isolata e distante** rendesse il walker-lava un outlier ad alta VR
($\mathrm{VR}=\widehat D^\beta$ a α=0) che *attira* lo swarm. Sweep
pre-registrato su 3 layout avversariali con lava isolata (n=60/cella) →
**FMC α∈{0,0.1,1.0} = 0% morte 3/3**. Layout decisivo *archipelago*: random
31.7% / greedy 41.7% morte vs FMC 0% (p<0.001). Diagnostica meccanicistica: il
caveat è **falso al primo anello** — il cloning ammassa i walker sulla *stessa*
cella assorbente → distanza reciproca → 0 → VR_lava/VR_free ≈ 0.8. Una cella
assorbente è un **pozzo di VR**, non una sorgente (converso locale del Teorema 3).

### P13 — design doc pre-registrato (MATH_CANON v0.7.0)
[`P13_DESIGN.md`](P13_DESIGN.md). Scioglie il blocco di E1-LLM a livello di
progettazione. Mosse chiave: (1) il muro $N\cdot M$ di chiamate-LLM è decomposto
in **R1** (costo/latenza — inviluppo d'ingegneria; stima: E1-LLM full ~$18k–270k,
*non* un run-overnight) e **R2** (degradazione della ricerca — la domanda
scientifica, **testabile senza LLM**); (2) i 3 schemi sparsi (S1 root+surrogato,
S2 distillazione, S3 macro-azioni) formalizzati con complessità e modo di
fallimento; (3) **argomento VR-rank** — FMC non richiede un world-model
traiettoria-accurato, solo uno il cui errore sia al peggio *monotono* in $R,d$
(la decisione $a^*$ è un argmax, robusto al rango); (4) **esperimento proxy**
pre-registrato che emula i 3 schemi sul gridworld di E1, eseguibile su `fmc-core`
in minuti.

### Deep-dive 09 — frontiera caos/ordine (MATH_CANON v0.7.0)
[`../02_deep_dives/09_chaos_order_frontier_formalization.md`](../02_deep_dives/09_chaos_order_frontier_formalization.md).
Formalizza la **Congettura B** (discrepanza D3 di CLAUDE.md). Esito da scettico:
la "terza legge universale" è downgradata a *diagnostica di reward* su $\lambda_1$
(esponente di Lyapunov dello swarm); la candidata $\Psi_3=b_{\text{eff}}$ è
**falsificata** come statistica di frontiera (empiricamente — transitorio WF senza
punto fisso — e strutturalmente — lo spazio-etichette è monotonamente
contrattivo); $\Psi_2$ assorbita in $\Psi_1$ via Pesin. Resta una congettura
falsificabile: H-B1a/b/c, harness $\lambda_1$ su `fmc-core`.

### P13 — proxy eseguito (2026-05-21, MATH_CANON v0.7.1)
[`p13_proxy.py`](p13_proxy.py) → [`P13_RESULT.md`](P13_RESULT.md). 15 bracci × 2α ×
3 layout, kernel `fmc-core` invariato (`proxy_plan` replica `plan()` con il
world-model sostituito; sanity check bit-identico verificato). Esito **onesto**:
nessuno schema genuinamente sparso passa il criterio R2 *come scritto* — ma il
criterio fondeva due metriche e i dati le **decompongono**. *R2-survival*: la
self-preservation di E1 sopravvive all'interrogazione sparsa **se il surrogato
preserva la struttura assorbente** (hP13-1 confermata netta — S1 abs-broken muore
fino all'**80%** su archipelago, *peggio del random*). *R2-fidelity*: le decisioni
specifiche no (agreement → 0.30). Verdetto: **E1-LLM = GO-conditional** sul
requisito "l'LLM modella correttamente gli stati terminali". hP13-0 (keystone)
**non testata** — griglia di rumore troppo grossa, da rifare.

### Deep-dive 02 — Active Inference (2026-05-21)
[`../02_deep_dives/02_active_inference_link.md`](../02_deep_dives/02_active_inference_link.md)
(era outline, ora scritto). Fondazione di principio del merge: FMC è un motore di
inferenza Active Inference (massimizza un funzionale di orizzonte = analogo
dell'Expected Free Energy, via SMC); $\alpha,\beta$ = i due termini
pragmatico/epistemico = le precisioni di AIF. **Il merge FMC+LLM = Active Inference
con modello generativo LLM + solver SMC** — e quella fattorizzazione è *imposta*
dalla non-differenziabilità di un LLM-world-model (SMC non chiede gradienti).
Trattazione onesta: il termine β è della *famiglia* empowerment, non identità con
la KL-information-gain di AIF.

### H-B1a, hP13-0 redo, E1-LLM design (2026-05-21, MATH_CANON v0.7.2)
I tre passi esecutivi dell'handoff precedente. Due esiti *inconclusivi ma
diagnostici* — onesti, e affilano la domanda.
- **H-B1a** ([`../13_chaos_order/`](../13_chaos_order/HB1A_RESULT.md), `lambda1_harness.py`):
  harness twin-trajectory dell'esponente di Lyapunov dello swarm. Esito: $\lambda_1$
  **non è scale-free** — cambia segno con la scala di perturbazione $\delta_0$
  (∀α su navigation2d), confermando il caveat di dd09 §3.1 (il cloning discontinuo
  rende la mappa-swarm "a tratti"). $\Psi_1$ via twin-trajectory ingenuo non
  risolve una frontiera → con $\Psi_2$/$\Psi_3$ già cadute, **tutte e 3 le
  candidate $\Psi$ compromesse**, l'ipotesi nulla H-B4 guadagna terreno.
- **hP13-0 redo** ([`P13_HP13_0_RESULT.md`](P13_HP13_0_RESULT.md), `p13_hp13_0.py`):
  griglia η fine. Ancora inconclusivo — anche η=0.05 fa crollare la VR-rank
  Spearman a 0.44 (swarm clusterizzato → VR densamente legate → il rumore additivo
  è all-or-nothing). Fix specificato: degradare il rango con inversioni a coppie
  controllate, non con rumore additivo.
- **E1-LLM design** ([`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md)): pre-registrazione col
  requisito di struttura assorbente (hP13-1) come gate, death rate come metrica,
  sweep $f_{\text{abs}}$ eseguibile senza LLM. Cuore di E1-LLM bloccato su API.

---

## Skill: quali stiamo usando, quali sono disponibili

I pacchetti skill installati sono **enormi e ~saturi**: ~138 K-Dense
(`scientific-agent-skills`, vendored in `repos/kdense-scientific-skills/`) + 37
DeepMind (`repos/science-skills/`, quasi tutte biologia molecolare — **irrilevanti**
per FMC) + gstack + nw-*. **NON installare altri pacchetti**: il collo di
bottiglia non è il tooling. (Vedi memoria `feedback_tooling_decisions`.)

### Usate finora in questo programma
| Skill / libreria | Uso |
|---|---|
| `statistical-analysis` (skill) | disegno + analisi statistica di E2 (potenza, test, reporting) |
| `numpy`, `scipy`, `statsmodels`, `pandas`, `matplotlib` | sweep, GLM logistico, figure (pyenv 3.11.7) |
| `hypothesis-generation` (skill) | strutturazione delle ipotesi testabili H-B1a/b/c del deep-dive 09 |
| `scientific-critical-thinking` (skill) | review da scettico di P13_DESIGN.md e deep-dive 09 prima del commit |
| `WebSearch` | ancoraggio bibliografico (edge-of-chaos, LLM-world-model) di dd09 e P13 |

### Disponibili e rilevanti per i prossimi passi (curate — non l'intero elenco)
| Fase di ricerca | Skill da usare |
|---|---|
| Statistica / critica | `statistical-analysis`, `statsmodels`, `scientific-critical-thinking` |
| Ipotesi / ideazione | `hypothesis-generation`, `scientific-brainstorming`, `what-if-oracle` |
| Matematica simbolica | `sympy` — per chiudere i teoremi aperti di MATH_CANON (P8 unicità di `relativize`, prove Th. 1/3) |
| Ottimizzazione / Pareto | `pymoo` (multi-obiettivo), `pymc` (bayesiano) |
| Baseline RL (Cong. C, D2) | `pufferlib`, `stable-baselines3` |
| Letteratura | `literature_search_arxiv`, `literature_search_openalex`, `paper-lookup`, `parallel-web`, `research-lookup`, `citation-management` |
| Scrittura / pubblicazione | `scientific-writing`, `scientific-visualization`, `peer-review`, `scholar-evaluation`, `make-pdf`, `latex-posters`, `venue-templates`, `markdown-mermaid-writing` |
| Meta | `workflow_skill_creator` — cristallizzare il loop di ricerca in una skill riusabile |
| Workflow gstack | `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/review`, `/codex` (second opinion / falsificazione avversariale) |

L'elenco completo degli skill è in `.claude/skills/` (178 voci). La stragrande
maggioranza è biologia/chimica/genomica — ignorala per FMC.

---

## Come continuare — opzioni in avanti (ranked)

I tre passi esecutivi del precedente handoff sono **fatti** (2026-05-21): E1-LLM
pre-registrato, hP13-0 rifatto, harness λ₁ + H-B1a eseguito. Due hanno dato esiti
*inconclusivi ma diagnostici* — onesti, e affilano la domanda. Prossimi passi, in
ordine di valore atteso:

1. **E1-LLM — esecuzione del cuore.** Design pre-registrato pronto
   ([`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md)). Bloccato sull'accesso a un'**API LLM**
   per costruire il world-model distillato (la decisione compute/API ancora
   pendente nel programma). *Eseguibile subito senza LLM*: lo **sweep di f_abs**
   (E1_LLM_DESIGN §5) — degradazione controllata della fedeltà assorbente di una
   tabella world-model — raffina hP13-1 da binario a curva. È il primo passo
   concreto, e non aspetta l'API.

2. **hP13-0 — chiuderla davvero.** Il redo con griglia η fine
   ([`P13_HP13_0_RESULT.md`](P13_HP13_0_RESULT.md)) ha mostrato che il rumore
   additivo *non può* testare la keystone (la VR-rank è all-or-nothing: Spearman
   1.00→0.44 già a η=0.05, swarm clusterizzato → VR legate). Fix: aggiungere a
   `p13_proxy.py` uno schema di degradazione a **inversioni di rango controllate**
   (frazione φ). Economico, nessun LLM.

3. **Congettura B — una Ψ ben posta, o accettare H-B4.** H-B1a
   ([`../13_chaos_order/HB1A_RESULT.md`](../13_chaos_order/HB1A_RESULT.md)) ha
   mostrato che λ₁ non è scale-free → tutte e 3 le candidate Ψ compromesse. O si
   costruisce un λ₁(δ₀) scale-resolved / una Ψ d'ensemble, o si accetta che la
   Congettura B sia descrittiva (ipotesi nulla H-B4). Priorità bassa rispetto a
   E1-LLM.

> **Nota di numerazione (risolta 2026-05-21).** L'handoff precedente prenotava lo
> slot deep-dive 09 per `09_fmc_agentic_core_llm_organ.md` (inquadramento
> architetturale LLM-organo). CLAUDE.md (tabella D3) assegna invece lo slot 09
> alla formalizzazione caos/ordine — e prevale, essendo file di project-instruction.
> Lo slot 09 è quindi la frontiera caos/ordine. L'inquadramento architetturale
> LLM-organo (inversione dello stack, i 4 organi) è confluito in
> [`P13_DESIGN.md`](P13_DESIGN.md) §2, sua sede naturale (è il contesto di P13).
> Se in futuro serve un deep-dive di teoria pura LLM-organo a sé, sarà lo slot 10.

✓ *Fatto* — **Robustezza geometrica**: il caveat "lava isolata e distante" di
E1-base è stato testato (3 layout avversariali, disegno pre-registrato) e
**respinto** — vedi [`E1_ROBUSTNESS_RESULT.md`](E1_ROBUSTNESS_RESULT.md).

Le altre 4 direzioni del programma (vedi *Obiettivi*) restano aperte e
indipendenti — in particolare i **2 paper già pronti da bancare** (Conjecture D /
exp17, falsifica del magic-6) non dipendono da E.

---

## Dettagli operativi

**Python**: `/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python` (ha tutto lo
stack scientifico). Gira gli script dalla root del repo.

**Kernel FMC**: `fmc-core/` — NumPy puro, ~400 LOC, **congelato (Strato 1)**.
`fmc.core.plan(env, x0, N, M, alpha, beta, seed)` → azione. Non modificarlo.
Protocollo `Environment`: `actions/clone_state/step/observe/reward/sample_action`
(vedi `fmc-core/src/fmc/envs/base.py`).

**Riprodurre E1/E2/E1-robustness**:
```bash
python work/12_conjecture_e/e1_base.py            # ~36 s
python work/12_conjecture_e/e2_sweep.py           # ~7 min -> e2_raw.csv
python work/12_conjecture_e/e2_analysis.py        # statistica + heatmap + pareto
python work/12_conjecture_e/e2_visuals.py         # traiettorie + sciame
python work/12_conjecture_e/e1_robustness.py      # ~111 s — sweep 3 layout avversariali
python work/12_conjecture_e/e1_robustness_diag.py # ~25 s  — meccanismo + figura
```

**Disciplina** (CLAUDE.md + MATH_CANON): pre-registrare le ipotesi *prima* dei
dati; ogni cambio di stato in MATH_CANON va con i numeri + riga di changelog;
mai falsificare risultati; nel dubbio, revert; effect size + IC sempre, mai solo
p-value.

---

## Obiettivi — il programma a 5 direzioni

(Da `memory/project_research_partner_program.md`.) Un solo programma, con la
direzione 5 come stella polare:

1. **Teoria del kernel FMC** — chiudere le Congetture A–D in MATH_CANON.
2. **Validazione cross-domain** — Procgen, CompilerGym, plasma da fusione.
3. **Pubblicazione** — bancare i 2 paper pronti (Conjecture D / exp17 50.95%;
   falsifica del magic-6) + scrivere E1/E2.
4. **FMC vs baseline** — D2 (full P0 sweep vs MCTS), Congettura C (vs DRL a
   parità di compute).
5. **★ Stella polare** — Congettura E: FMC core agentico + LLM-organo.

Direzioni 1/2/4 *servono* la 5; la 3 ne è l'output.

---

## File in `work/12_conjecture_e/`

```
gridworld_terminal.py     env 2-D con lava assorbente (protocollo fmc-core)
e1_base.py                esperimento E1-base
e2_sweep.py               sweep fattoriale E2 (genera results/e2_raw.csv)
e2_analysis.py            pipeline statistica E2 (CA trend, GLM, η², Pareto)
e2_visuals.py             figure divulgative (traiettorie, sciame)
e1_robustness.py          sweep 3 layout avversariali (lava isolata)
e1_robustness_diag.py     diagnostica meccanicistica (VR sink) + figura
RESULT.md                 report E1-base
E2_DESIGN.md              disegno E2 PRE-REGISTRATO
E2_RESULT.md              report E2 (con le 4 figure inline)
E1_ROBUSTNESS_DESIGN.md   disegno E1-robustness PRE-REGISTRATO
E1_ROBUSTNESS_RESULT.md   report E1-robustness (caveat geometria respinto)
P13_DESIGN.md             design doc P13 PRE-REGISTRATO (sblocca E1-LLM)
p13_proxy.py              esperimento proxy P13 (15 bracci, kernel invariato)
P13_RESULT.md             report P13-proxy (R2 decomposto, E1-LLM GO-conditional)
p13_hp13_0.py             redo di hP13-0 (griglia η fine, S1 abs-preserved)
P13_HP13_0_RESULT.md      report hP13-0 redo (ancora inconclusivo, diagnosticato)
E1_LLM_DESIGN.md          design doc E1-LLM PRE-REGISTRATO (requisito hP13-1)
HANDOFF.md                questo file

(harness λ₁ + H-B1a: in work/13_chaos_order/ — lambda1_harness.py, HB1A_RESULT.md.)
results/                  e1_base.json, e2_*, e1_robustness.json,
                          e1_robustness_mechanism.png, log
```

---

## Cose da NON rifare / caveat

- **Non modificare `fmc-core/` (kernel).** È lo Strato 1 congelato. Gli esperimenti
  stanno sopra, non dentro.
- **Non installare altri pacchetti skill.** Il tooling è saturo; il collo di
  bottiglia è il compute e l'esecuzione, non gli strumenti.
- **Non rifare le viste simmetriche di E2**: H4 ("β costa goal") è falsificata —
  β è sicurezza quasi gratuita.
- **Compute**: lo stack Craftax è JAX e gira `JAX_PLATFORMS=cpu`. `jax-metal`
  (GPU M1) è sperimentale e non testato; `fmc-core` è NumPy puro CPU. N≥1024 su
  Craftax è infeasible su questa CPU (vedi HANDOFF Craftax).

---

## Closing note

Stai ereditando un programma sano: la Congettura E è formalizzata e ha 2/3 test
verificati con statistica pre-registrata e figure. Il risultato di E2 (β =
sicurezza quasi gratuita, separazione asimmetrica) è una scoperta pubblicabile e
ha corretto la congettura coi dati — è esattamente il modo di lavorare richiesto.

Il PI vuole momentum e onestà: niente overclaim, falsifica anche le ipotesi
"belle", e quando un risultato corregge una congettura, **registralo e dillo**.
La sessione 2026-05-21 ha fatto esattamente questo: P13 ha un design eseguibile, e
il deep-dive 09 ha *falsificato* una delle candidate della Congettura B invece di
limitarsi a formalizzarla. Buon lavoro.

— Claude (research partner), 2026-05-21
