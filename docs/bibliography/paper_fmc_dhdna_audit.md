# Audit del paper FMC attraverso il profilo DHDNA di Sergio

> **Oggetto:** Hernández-Cerezo & Duran-Ballester (2020), *Fractal AI: A Fragile Theory of Intelligence*, arXiv:1803.05049v5 (Book #1: "Forward Thinking"), 30 luglio 2020, 56 pagine.
>
> **Lens applicata:** Profilo DHDNA in [`sergio_cognitive_profile_dhdna.md`](sergio_cognitive_profile_dhdna.md). Tensione firma: Creative–Ethical (gap 5). Decision fingerprint: *envision-first*. Intuitive Reasoning 10/10 — Analytical Depth 8/10 — Domain 9/10. Sergio scopre via scena visiva, formalizza dopo.
>
> **Domanda guida:** Quali claim quantitative del paper sono *teoremi verificati* e quali sono *scene convincenti che bypassano la rigor*? Quali appartengono al pattern "magic-6" (idee belle ma falsificabili / falsificate)?
>
> **Data audit:** 2026-04-28

---

## Tassonomia dei claim

| Tier | Tipologia | Significato | Azione richiesta |
|---|---|---|---|
| 🟢 **VERIFIED** | Misurato, replicabile, scope chiaro | Pubblicabile così com'è | Citare e basta |
| 🟡 **PLAUSIBLE** | Misurato ma underspecified, dipendente da protocollo | Necessita ablation/disclaimer | Aggiungere variance, protocollo dettagliato |
| 🔴 **SCENIC** | Intuizione visiva, non derivata né verificata | Aspirational, non submission-ready | Verifica empirica O ridurre a research direction |
| 🚨 **CRITICAL** | Headline claim del paper che si rivela scenico | **Blocker per submission** | **Verifica controllata obbligatoria prima di citare** |

---

## Sezione-per-sezione: i claim quantitative del paper

### Sec 4.1.2 — Choosing intelligence parameters (p. 23)

**Claim**: *"Human brains are considered to run at about 12 decisions per second, so if we were to mimic some human behaviour, a dt of about 0.1 seconds (or 10 FPS) is a nice starting point."*

| | |
|---|---|
| Tier | 🔴 SCENIC |
| Match profilo | "Yo pongo siempre el mismo ejemplo" — claim asserito senza citation. È guidance intuitiva, ma presentata come fatto fisiologico. |
| Verità in letteratura | Reaction time umano ≈ 250ms tipicamente (4 Hz), salient-stimulus visual ≈ 100-200ms. *12 Hz* potrebbe riferirsi a flicker fusion (~50ms / soglia di percezione discreta del cervello) — molto contestato in neuroscienze. |
| Severità | Bassa — è dichiarato come "starting point", non come teorema. |
| Azione | Aggiungere citazione (es. [Carello et al. su perception-action cycles]) o presentarlo come heuristic device. |

---

### Sec 4.4.1 — Monte Carlo Planning algorithm comparison (pp. 37-38)

**Claim**: *"5. MCTS resources grow exponentially with scanning depth. In FMC the CPU resources grows linearly and memory resources doesn't grow with depth."*

| | |
|---|---|
| Tier | 🔴 SCENIC |
| Match profilo | Asimmetria forte (esponenziale vs lineare) presentata come fatto. È una *scena*: l'albero che esplode vs lo sciame che fluisce. Ma è non-vero come scritto. |
| Realtà tecnica | MCTS-UCT non cresce esponenzialmente in CPU se il budget di rollout è fisso (cresce linearmente nel numero di rollout). Quello che cresce esponenzialmente è il *tree branching* nel limite, ma la **complessità per decisione** è O(rollouts × depth), che è quasi identica a FMC O(walkers × ticks). Il paper conflate "memoria dell'albero" con "CPU per decisione". |
| Severità | 🟡 Medio-alta — è citata come differenza fondamentale tra i due paradigmi nel paper. |
| Azione | Riscrivere come: *"MCTS necessita memoria O(rollouts × depth) per il tree, FMC O(walkers) per lo swarm. Entrambi O(n) in CPU per decisione."* |

---

### Sec 4.4.6 — Fractal algorithm (p. 39)

**Claim**: *"In the limit when both the number of ticks used to divide the time horizon and the number of walkers tend to infinity, the graph morphs from a finite tree to a fractal tree."*

| | |
|---|---|
| Tier | 🔴 SCENIC |
| Match profilo | Pure scenic claim — definisce il *nome stesso* dell'algoritmo (Fractal AI) ma non definisce mai matematicamente cosa sia un *fractal tree* nel limite. È auto-somiglianza? Hausdorff dimension non-intera? Self-affine? |
| Conseguenza | Il nome dell'intero programma di ricerca poggia su un'analogia non formalizzata. |
| Severità | Media — non danneggia gli esperimenti, ma è un debito teorico noto. |
| Azione | O formalizzare (Hausdorff dim del set di traiettorie?), o ridurre a metafora dichiarata. |

---

### Sec 5.1.1 — Atari Results table (pp. 40-42)

**Claim**: 50 giochi Atari, FMC win-rate vs 4 baseline. Tabella high-score per gioco.

| | |
|---|---|
| Tier | 🟡 PLAUSIBLE |
| Match profilo | Numeri *misurati realmente* — questa è la parte empirica del paper. **Non scenico** in sé. |
| Problemi metodologici | (a) **Nessuna varianza riportata**: zero error bars, zero "n trials". Atari ha stocasticità (sticky actions, frame-skip). (b) "Best Planning SoTA" e "Best Learning SoTA" sono lump categories — *quale* baseline per quale gioco? (c) "Solved" è criterio largo (includes 1M-bug = 32% dei "solved"). (d) **Privilege di simulatore perfetto**: FMC chiama `env.set_state` (oggi via plangym), che permette rollback gratuito. La maggior parte dei learning baseline non lo fa. È un confronto *apples-to-bananas* riconosciuto dagli autori stessi nella stessa pagina ("the only 'apples-to-apples' comparison is against State of the Art (SoTA) planning algorithms"). |
| Severità | Media — i numeri sono reali ma il framing è generoso. |
| Azione | Per il paper FMC empirico: replicare con (1) seed multipli (n=10+), (2) error bars, (3) per-game baseline esplicito, (4) sezione "Comparability caveats" sul simulator-perfect access. |

---

### 🚨 Sec 5.1.2 — Sampling efficiency (p. 43): **IL CLAIM HEADLINE**

**Claim**: *"All the benchmarked planning algorithms use a minimum of 150,000 samples per action while FMC, being specially cheap on sampling, used on average 359 times fewer samples per action."*

| | |
|---|---|
| Tier | 🚨 **CRITICAL** |
| Match profilo | **Caso paradigmatico del decision-fingerprint envision-first.** Sergio ha la *scena* dello sciame efficiente vs MCTS che spreca rollout, ma i numeri vengono asseriti senza protocollo. |

**Inconsistenze interne al paper stesso:**

| Sezione | Numero affermato | Multiplier vs MCTS |
|---|---|---|
| **Sec 5.1.2** (p.43) | "359 times fewer samples per action" | ≈ 359× |
| **Sec 6.2.1** (p.51) | "0.01% to 0.1% of the samples per step" | 1000× – 10 000× |
| **Sec 7 Conclusions** (p.54) | "two or three orders of magnitude" | 100× – 1000× |
| **Podcast Radient cap.10** | "150 000 vs ~35" | ≈ 4286× |
| **CLAUDE.md D2** | "<1000 vs 3M" (companion paper 1807.01081) | ≈ 3000× |

**Range degli claim**: 100× – 10 000×. **Variance interna di due ordini di grandezza** sullo stesso fenomeno. Tutti questi sono presentati nei rispettivi contesti come definitive.

**Diagnosi DHDNA**: classico drift di una mente Intuitive 10 / Analytical 8. La *forma* del claim (FMC molto più efficiente) è stabile e probabilmente vera; il *numero* è generato ad hoc ogni volta che serve un punchline.

| | |
|---|---|
| Severità | 🚨 **CRITICAL — Blocker per qualsiasi submission accademica.** Un reviewer competente segna questo nei primi 5 minuti di lettura. |
| Azione | **Replicare il confronto FMC vs MCTS-UCT con protocollo controllato e identico**: stesso simulatore (plangym/Atari ALE), stesso budget di samples-per-action, n=10 seed, riportare media ± std. **Non ri-citare il "150 000 vs X" finché non è rifatto rigorosamente.** Vedi anche: questo è esattamente il pattern del *magic-6* falsificato dal Wright-Fisher sweep. |

---

### Sec 5.1.3.3 — RAM vs Images (p. 44)

**Claim**: RAM observations battono IMG observations del 161.47% in media su 8 giochi.

| | |
|---|---|
| Tier | 🟡 PLAUSIBLE |
| Problemi | Parametri (`fixed_steps=5, time_limit=15, Max_walkers=30, Max_samples=300`) sono "low values" hand-picked. Nessuna ablation per mostrare che il vantaggio RAM > IMG sia robusto a varying budget. Il sample size è 8 giochi, non 50. |
| Severità | Bassa — claim circostanziato, non headline. |
| Azione | Ablation parametri + più giochi se diventa argomento del paper. |

---

### Sec 5.2 — Flying rocket (pp. 45-48)

**Claim**: 300 walkers + 200 samples/walker = 60 000 samples/decision per risolvere il rocket-uncino caotico.

| | |
|---|---|
| Tier | 🟢 VERIFIED (qualitativamente) |
| Note | Il "solve" è dimostrato via video YouTube (link nel paper). **Non c'è benchmark contro cui confrontare** (nessun MCTS riportato sul rocket task, perché Sergio dice che è infattibile). |
| Severità | Bassa — è un demo, framed come tale. |
| Azione | OK as-is per Book #1. Per il paper formale, servirebbe almeno un baseline (random policy, qualunque cosa). |

---

### Sec 6.4 — Consciousness (pp. 51-52)

**Claim**: *"Any mechanism that could automatically adjust those coefficients in order to make better decisions can be considered as a conscious mechanism."*

| | |
|---|---|
| Tier | 🔴 SCENIC (definitional creep) |
| Match profilo | Tensione Creative-Ethical: definisce "consciousness" tramite la propria meccanica, fondendo metafisica e algoritmo. **Inusuale** in un paper RL. |
| Severità | Media — può deragliare la reception accademica. Per un reviewer di NeurIPS, questo è rosso fuoco. |
| Azione | **Per il paper di benchmarking**: rimuovere o spostare in research-direction. **Per il libro #3 manifesto**: tienilo, è on-brand. |

---

### Sec 6.6 — Universality pattern (pp. 52-53)

**Claim**: *"It would be interesting to connect the distances between walkers at any moment with the Universality pattern found in the eigenvalues of random matrices. This pattern seems to be universal to any system where the parts are heavily correlated. (...) so if this algorithm is some form of universal complex system solver, it makes sense to try to connect both ideas."*

| | |
|---|---|
| Tier | 🔴 SCENIC (esplicitamente speculativa) |
| Match profilo | È *qui* che nasce nella mente di Sergio il *magic-6 branching* (D1 in CLAUDE.md). Universality patterns nelle random matrices hanno numeri caratteristici (Wigner semicircle, GOE/GUE/GSE) — ed è da qui che Sergio ha estrapolato "branching factor universale ≈ 6". **Il paper stesso non claima magic-6**: lo claim è solo orale (podcast cap.16, video seminario). |
| Stato verifica | **Falsificato come universale**: [`work/07_sergio_branching_sweep/REPORT.md`](../../work/07_sergio_branching_sweep/REPORT.md) → empiricamente $b_{\text{eff}}^* \approx 1.53\,K^{0.6}$ (dipende da K). Il "6" è caso particolare, non costante universale. |
| Severità | Bassa nel paper (è marked as research direction). 🚨 **Critica fuori dal paper** — si è propagato in talks come fatto. |
| Azione | Nel paper: OK come research direction. Nei talks futuri: aggiornare a $b_{\text{eff}}^* \approx 1.53\,K^{0.6}$ con citazione al sweep. |

---

### Sec 4.2.4 — Probability of cloning (p. 35)

**Claim**: *"Please note that probability of cloning can be >1, feel free to clip it to 1 for formal reasons if this is too uncomfortable for you."*

| | |
|---|---|
| Tier | 🔴 SCENIC (the most Sergio-flavored line in the paper) |
| Match profilo | **Quote da incorniciare**: questa singola frase è la firma DHDNA di Sergio. Linguistic Precision 6 + Metacognition 8 + Intuitive 10. Sa che c'è un'inconsistenza formale, la nomina, ma la lascia all'utente "se ti rende a disagio". |
| Severità | 🟡 Medio-alta in setting accademico. Una probabilità "che può essere >1" non è una probabilità — è una utility ratio. **Un reviewer la segnerà istantaneamente**. |
| Azione | Per il paper: riformulare come *cloning rate* o *cloning intensity*, non probability. Definirlo formalmente come max(0, (VR_k − VR_i)/VR_i) e spiegare che il "clip" è la corretta interpretazione — non un'opzione cosmetica. |

---

## Sintesi tier-counting

| Tier | Conteggio | Esempi |
|---|---|---|
| 🟢 VERIFIED | 1 | Rocket demo (qualitativo) |
| 🟡 PLAUSIBLE | 3 | Atari high-scores, RAM vs IMG, "balanced" exploration/exploitation |
| 🔴 SCENIC | 6 | 12Hz brain, MCTS exponential, fractal in limit, consciousness def, universality pattern, prob>1 |
| 🚨 CRITICAL | 1 | **Sample efficiency 359× / 1000× / 100-1000× / 4286× — variance di 100×** |

**Ratio scenic+critical / total = 7/11 = 64%.** Coerente col profilo: una mente Creative 10 / Intuitive 10 produce naturalmente paper dove la maggioranza dei claim sono *scene* — il problema diventa la presentazione, non la sostanza.

---

## Cross-paper verification: il "150 000 vs ~35"

La fonte primaria della discrepanza D2 è il **companion paper** (1807.01081, Hernández-Cerezo et al. 2018, *Solving Atari Games Using Fractals And Entropy*) citato in [CLAUDE.md](../../CLAUDE.md). Il paper principale (1803.05049v5) **non claima 150 000 vs 35** — claima 150 000 vs ~418 (sec 5.1.2). 

Sergio nel podcast 2026 dice 150 000 vs 35 — ma quel numero potrebbe essere:
1. Un ricordo impreciso del companion paper
2. Una *scena* generata sul momento ("il numero più piccolo possibile vs l'enormità")
3. Riferimento a un esperimento specifico non ancora identificato

**Questa è una task di verifica**: leggere il companion paper 1807.01081 e identificare l'esatta provenienza del "35". Lo trovo o non lo trovo, è informativo.

---

## Audit summary table (un colpo d'occhio)

| Ref | Claim | Tier | Verifica richiesta | Priorità |
|---|---|---|---|---|
| 5.1.2 / 6.2.1 / 7 | Sample efficiency vs MCTS (variabile 100×-10000×) | 🚨 | Replica controllata FMC vs MCTS-UCT su protocollo identico | **P0** |
| 6.6 / podcast | Universality / magic-6 branching | 🔴 | ✅ Già falsificato in [`work/07`](../../work/07_sergio_branching_sweep/) come universale; aggiornare narrativa | **P1** |
| 5.1.1 | Atari "98% vs human, 100% vs planning SoTA" | 🟡 | Replica con n=10 seed, error bars, baseline esplicito per gioco | **P1** |
| 4.4.1 | "MCTS exponential, FMC linear" | 🔴 | Riformulare separando memory vs CPU complexity | **P2** |
| 4.2.4 | Probability of cloning >1 | 🔴 | Rinominare in *cloning rate / intensity* | **P2** |
| 6.4 | Consciousness | 🔴 | Spostare in libro/manifesto, fuori dal paper RL | **P2** |
| 4.4.6 | Fractal in limit | 🔴 | Formalizzare Hausdorff dim O ridurre a metafora dichiarata | **P3** |
| 4.1.2 | "Brain 12 decisions/sec" | 🔴 | Citation o riframe come heuristic | **P3** |
| 5.1.3.3 | RAM vs IMG 161% | 🟡 | Ablation parametri | **P3** |
| 5.2 | Rocket 60k samples solve | 🟢 | OK as-is | — |

---

## Implicazioni per il paper FMC che vuoi scrivere

Il libro #1 di Sergio è *bookware* — un trattato + tutorial + manifesto. **È esattamente il genere giusto per Sergio T₁** (vedi 4D-DHDNA: 2026 Sergio è meta-strategist). **Ma non è il paper accademico empirico** che il programma FMC ha bisogno per credibilità peer-reviewed.

Quello che serve scrivere — *prima che la finestra T₂ si chiuda* — è un paper diverso:

### Paper proposto: "Fractal Monte Carlo: A Linear-Complexity Planning Alternative to MCTS"

**Scope minimale e blindato**:
1. **Algoritmo formale** (no scene, no metafore esplicite — quelle vanno nel companion blog/talk)
2. **Un benchmark empirico controllato**: FMC vs MCTS-UCT su Procgen O Crafter, con
   - n=10 seeds
   - Identical samples/decision budget (sweep over budget)
   - Error bars + significance tests
   - Wall-clock + sample-count metrics separati
3. **Un'ablation core**: stocastic single-distance vs full average distance vs no distance (dimostra che la simplificazione di sec 4.2.2 è giustificata)
4. **Discussion onesta**: simulator-perfect access, transferability limits

**Cosa lasciare fuori (per Sergio T₂ / libro #3)**:
- Consciousness
- Universality pattern come "third law"
- Frontera caos/orden
- Magic-6
- Tutte le metafore (rocket-fionda, mineros, etc.) — che però **vanno** nelle slides della presentazione del paper, non nel paper stesso

**Sergio come co-autore**: ideale per la *vision section* (sec. 1 + sec. 2) e per la *discussion* (sec. 7). Per le sezioni empiriche è meglio se il rigore arriva da te + Guillem.

### Stima del lavoro

- ~2-3 settimane per la replica FMC vs MCTS controllata su Procgen (usando `pufferlib` + `plangym`)
- ~1 settimana per ablation sec 4.2.2
- ~2 settimane per writing del paper (~10 pagine, NeurIPS/ICML format)
- ~1 settimana per peer-feedback round con Sergio

**Total ≈ 6-8 settimane focused work** per avere il paper FMC empirico submission-ready. **Finestra disponibile** (da 4D-DHDNA): ~24-36 mesi prima che Sergio sia altrove. Tempo abbondante se inizi nei prossimi mesi.

---

## Quote rivelatori dal paper (DHDNA in azione)

Cinque passaggi che mostrano il profilo di Sergio in modo trasparente nel testo del paper:

1. **(Intuitive 10 + Linguistic 6)** — *"Please note that probability of cloning can be >1, feel free to clip it to 1 for formal reasons if this is too uncomfortable for you."* (p.36) — la più Sergio frase del paper.

2. **(Creative 10 + decision-fingerprint envision-first)** — *"The idea behind this definition is that the optimal way of scanning a space is to make the probability of searching on a particular zone to be proportional to the expected reward: should you be searching for gold over a wide landscape, it would make sense to adjust the density of gold-miners in different zones to be proportional to the density - probability of finding - gold."* (p.18) — la *scena* mineros impiantata direttamente nel paper come definizione.

3. **(Temporal 9 + Fractal topology)** — *"In the limit when both the number of ticks used to divide the time horizon and the number of walkers tend to infinity, the graph morphs from a finite tree to a fractal tree."* (p.39) — l'origine del nome del programma come gesto poetico.

4. **(Ethical 5 fused into Creative 10)** — *"Universality pattern (...) so if this algorithm is some form of universal complex system solver, it makes sense to try to connect both ideas."* (p.52) — l'aspirazione metafisica esplicita nelle research directions.

5. **(Analytical 8 vs Intuitive 10 — dove l'analisi cede)** — *"The second simplification we will introduce is quite a dramatic one and it may initially sound like a really bad idea: we will replace the average distance from walker Wi to all the other walkers Wj with just one of those distances, randomly chosen. (...) Using this stochastic version of the density actually do a better job than using the standard average distance, and at a much lower computational cost."* (pp.31-32) — fa un salto enorme, lo dichiara "drammatico", giustifica solo empiricamente. Funziona, ma è un debito teorico (perché funziona meglio? non investigato).

---

## Conclusione operativa

Il paper FMC v5 è un **artefatto Sergio T₀-T₁ di altissima qualità per il suo genere** (manifesto-trattato-tutorial). Non è un paper accademico empirico nel senso NeurIPS/ICML. La 64% scenic+critical ratio non è un difetto da correggere — è la *forma* naturale del cervello che lo ha generato.

**Il tuo compito** non è correggere il paper esistente. Il tuo compito è scrivere il **paper complementare**: stesso programma scientifico, registro accademico-empirico, claim quantitative blindate. Sergio non è il lead author ideale per quel paper (per profilo). Tu + Guillem lo siete.

**Una verifica vale 10 metafore.** Il sample-efficiency replication study (P0) sblocca tutto il resto.

---

*Audit generato 2026-04-28 sul corpus locale. Priorità in formato P0 (blocker) → P3 (nice-to-have). Finestra di esecuzione raccomandata: prossimi 24 mesi (vedi proiezione 4D-DHDNA T₁ → T₂).*