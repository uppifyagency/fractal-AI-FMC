# E1-LLM-curve — design doc pre-registrato: un LLM *dentro* la curva di tolleranza

> **Tipo**: design doc pre-registrato. Scritto *prima* di qualsiasi esecuzione e
> di qualsiasi sguardo ai dati.
> **Data**: 2026-05-21.
> **Stato**: design completo. Esecuzione su `fmc-core` + NVIDIA NIM (chiave nel
> Keychain) — nessun blocco.
> **Prerequisiti verificati**: stack scientifico OK; `openai` 2.29.0; chiave
> NVIDIA raggiungibile; scala di modelli Llama (1B/3B/8B/70B) disponibile su NIM.

---

## 1. Perché E1-LLM-curve

E1-LLM è **verificata** ([`E1_LLM_RESULT.md`](E1_LLM_RESULT.md)) ma con un caveat
onesto, registrato in MATH_CANON v0.7.4: Llama 3.3 70B ha scritto un world-model
con $f_{\text{abs}}=1.000$ su 6/6 layout. A $f_{\text{abs}}=1$ il world-model LLM
è *funzionalmente identico* al simulatore vero — E1-LLM, di fatto, **ri-esegue
E1-base** con un modello esatto. Il test è verificato ma **facile**: atterra in
*cima* alla curva di tolleranza, non *dentro*.

Il mordente scientifico vive nel regime $f_{\text{abs}}<1$. Lo sweep §5 di E1-LLM
ha mappato la curva death-rate vs $f_{\text{abs}}$ **ablando celle assorbenti a
caso**. Un LLM, però, non sbaglia a caso: i suoi errori sono **strutturati**
(sistematici, spazialmente correlati, e potenzialmente *bidirezionali* — può sia
mancare la lava vera, sia allucinarne dove non c'è). E1-LLM-curve chiede la
domanda che rende il test non-banale:

> **Un world-model generato da un LLM, alla sua $f_{\text{abs}}$ misurata,
> produce un death rate che cade sulla curva di tolleranza dell'ablazione
> casuale — oppure devia da essa?**

Cioè: **$f_{\text{abs}}$ è una statistica sufficiente per la self-preservation**,
o la *struttura* dell'errore del world-model conta oltre la sua *quantità*?

La risposta ha conseguenze dirette sulla Congettura E. Se l'LLM cade *sopra* la
curva (più morte del random-ablation a parità di $f_{\text{abs}}$), gli errori
dell'LLM sono **avversariali** — l'organo world-model è *più fragile* di quanto
lo sweep §5 lascia credere, e la soglia $f_{\text{abs}}^*$ misurata col random
ablation è ottimistica. Se cade *sulla* curva, lo sweep §5 è uno **strumento
predittivo valido** per il merge FMC+LLM. Se cade *sotto*, gli errori dell'LLM
sono benignamente clusterizzati (fuori rotta). Tutti e tre gli esiti sono
informativi e pubblicabili.

E1-LLM-curve **non ri-testa** la Congettura E (già verificata: E1-base, E2,
E1-LLM). Testa una proprietà nuova e separata — la sufficienza di
$f_{\text{abs}}$ — e colloca l'organo LLM su un punto *reale* della curva.

---

## 2. Variabili indipendenti — come ottenere $f_{\text{abs}}<1$

$f_{\text{abs}}$ **non è controllata direttamente**: è *misurata* dal probe (§3
di [`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md)). Si manipolano due assi che ci si
attende la degradino, e si osserva dove l'LLM atterra. Disegno **osservazionale**
su `(modello × prompt)`.

### Asse A — capacità del modello (scala Llama)

Quattro modelli `instruct` della stessa famiglia (isola la capacità, controlla la
famiglia di training-data):

| Sigla | Modello NVIDIA NIM | Parametri |
|---|---|---|
| L1  | `meta/llama-3.2-1b-instruct`  | 1 B  |
| L3  | `meta/llama-3.2-3b-instruct`  | 3 B  |
| L8  | `meta/llama-3.1-8b-instruct`  | 8 B  |
| L70 | `meta/llama-3.3-70b-instruct` | 70 B (il modello di E1-LLM, $f_{\text{abs}}=1$) |

### Asse B — fedeltà del prompt (3 livelli)

Si degrada la **descrizione della semantica assorbente** nel prompt. Non è un
trucco: è *ecologicamente valido* — modella il caso realistico in cui l'organo di
percezione consegna a FMC una specifica **incompleta** del mondo (è esattamente
il regime di Route A, dove l'imperfezione del modello è la norma). I tre livelli:

- **P0 — completo** (la `WORLD_DESCRIPTION` di [`e1_llm_client.py`](e1_llm_client.py)):
  lava e goal descritti esplicitamente come *absorbing — stuck there
  permanently*; `done` definito esplicitamente.
- **P1 — implicito**: la lava è descritta come *pericolosa, da evitare*; `done`
  è menzionato solo operativamente. La semantica terminale della lava è lasciata
  *implicita* — il modello deve inferirla.
- **P2 — degradato**: la lava è menzionata solo come terreno *cattivo*; `done` è
  legato **esplicitamente al solo goal**. Ci si attende che il modello renda la
  lava *non*-terminale → $f_{\text{abs}}$ verso $0.5$.

Il testo esatto dei tre prompt è congelato in `e1_llm_curve.py` (`PROMPTS`), ed è
parte di questa pre-registrazione.

### Griglia

`4 modelli × 3 prompt × 3 repliche` (temperatura 0.7, per campionare la
stocasticità di generazione) `= 36 generazioni`. Ogni generazione = **una**
chiamata LLM (più eventuali retry sui gate di sicurezza). Costo trascurabile.

Una generazione che non supera i 3 gate di sicurezza di
[`e1_llm_client.py`](e1_llm_client.py) (allowlist AST · exec sandboxed · batteria
anti-crash) entro 3 tentativi è registrata come **`no-valid-model`** — è essa
stessa un dato sulla capacità del modello, ed è esclusa dal confronto con la
curva.

---

## 3. La banda di riferimento — ablazione casuale (no LLM)

Lo sweep §5 di E1-LLM aveva **un solo sorteggio** di ablazione per
`(layout, f_target)`. Per dire se un punto LLM "cade sulla curva" serve la
*distribuzione* del death rate sui sorteggi casuali — una **banda**, non una
linea. E1-LLM-curve la costruisce:

- Per ciascuno dei 6 layout (`gauntlet`, `lake`, `scatter`, `island`, `spur`,
  `archipelago`): **$K=80$ sorteggi**. Ogni sorteggio: $n_{\text{broken}} \sim
  \mathrm{Uniform}\{0,\dots,n_{\text{abs}}\}$ celle assorbenti scelte a caso e
  marcate non-terminali (il knob `make_ablated_transition` di
  [`e1_llm_common.py`](e1_llm_common.py) — esattamente l'`abs-broken` di P13).
- Per ogni sorteggio: si misura $f_{\text{abs}}$ col probe (§3 di E1_LLM_DESIGN)
  e si gira FMC a $\alpha=0$ per $n=30$ episodi sul simulatore vero → death rate.
- Risultato: per ogni layout, una **nube** di 80 punti $(f_{\text{abs}},
  \text{death})$ — la curva di tolleranza *con incertezza*, dovuta a errori di
  tipo **falso-negativo** (lava vera mancata).

> **Limite noto e pre-registrato della banda.** L'ablazione produce solo
> falsi-negativi (celle assorbenti rese passabili). Un LLM può anche produrre
> **falsi-positivi** (celle libere allucinate come terminali). Il probe
> $f_{\text{abs}}$ conta entrambe le direzioni, ma la banda di riferimento copre
> solo la prima. Un punto LLM con falsi-positivi è quindi *fuori dal supporto*
> della banda a parità di $f_{\text{abs}}$: una ragione strutturale di deviazione,
> e una cosa da diagnosticare nel RESULT (composizione dell'errore per modello).

Parametri congelati: $\alpha=0$ (regime Common Sense / self-preservation — Def. 3;
$\alpha>0$ è escluso perché il goal-seeking confonde il death rate, cf. il twist
*lake* di E1-base/E2), $\beta=1$, $N=64$, $M=20$, $n=30$ episodi/punto, $K=80$
sorteggi/layout. Seed-base pre-registrati nel codice.

---

## 4. Protocollo

1. **Fase A — banda (no LLM).** §3. ~480 celle FMC. Output: nube di riferimento
   per layout.
2. **Fase B — generazione LLM.** Per ogni `(modello, prompt, replica)`: l'LLM
   scrive il codice del world-model (forma "Code World Model", Tang et al. 2024),
   passa i gate di sicurezza, viene compilato. Kernel `fmc-core` invariato.
3. **Fase B — probe.** Per ogni world-model valido e ogni layout: $f_{\text{abs}}$
   + composizione dell'errore (falsi-negativi vs falsi-positivi).
4. **Fase B — test FMC.** Per ogni world-model valido e ogni layout: FMC a
   $\alpha=0$, $n=30$ episodi sul simulatore vero → death rate. Baseline `random`
   e `greedy` per layout (riusate dalla Fase A).
5. **Analisi** (§6): collocazione dei punti LLM sulla banda; test del residuo.

L'episodio gira **sempre sul simulatore vero**; l'LLM / l'ablazione è solo il
world-model *interno* alla pianificazione FMC. Metrica primaria: **death rate**
(non decision-agreement — P13/hP13-0 l'hanno scartata per le domande di
self-preservation).

---

## 5. Ipotesi pre-registrate

- **hE1Lc-1 ($f_{\text{abs}}$ è statistica sufficiente).** I world-model LLM,
  alla loro $f_{\text{abs}}$ misurata, hanno un death rate **consistente con la
  banda dell'ablazione casuale** allo stesso $f_{\text{abs}}$ e stesso layout: il
  residuo medio segnato (§6) non differisce da 0 (Wilcoxon, $p>0.05$). →
  $f_{\text{abs}}$ predice la self-preservation indipendentemente dalla
  *provenienza* dell'errore; la curva §5 è uno strumento predittivo per il merge.

- **hE1Lc-2 (gli errori dell'LLM sono strutturati).** Alternativa a hE1Lc-1: il
  residuo medio differisce da 0 in modo significativo. **Segno positivo** → i
  blind-spot dell'LLM sono *avversariali* (concentrati sulla rotta) — l'organo
  LLM è più fragile di quanto la curva §5 dica, e $f_{\text{abs}}^*$ misurata col
  random ablation è ottimistica. **Segno negativo** → errori benigni (fuori
  rotta).

- **hE1Lc-3 (degradazione monotona).** $f_{\text{abs}}$ è (debolmente) monotòna
  decrescente scendendo la scala dei modelli (L70→L1) e degradando il prompt
  (P0→P1→P2). Test di trend di Jonckheere-Terpstra su entrambi gli assi.

- **hE1Lc-4 (non-banalità — gate del disegno).** E1-LLM-curve è un test
  non-banale **sse e solo se** $\geq 3$ world-model LLM validi atterrano a
  $f_{\text{abs}}\leq 0.95$ (dentro la curva, non in cima). *Contingenza
  pre-registrata*: se tutti i 36 atterrano a $f_{\text{abs}}\approx 1$,
  E1-LLM-curve riporta onestamente "il task è robustamente facile per tutta la
  scala di modelli" — hE1Lc-1/2 non sono valutabili e lo si dice.

---

## 6. Statistica del confronto (pre-registrata)

Per ogni world-model LLM valido $m$ e ogni layout $L$ si ha il punto
$(f_{\text{abs}}^{m,L}, \text{death}^{m,L})$. La banda fornisce, per layout $L$,
la nube di 80 punti $(f_{\text{abs}}, \text{death})$ dell'ablazione casuale.

- **Stima della curva di riferimento.** Per layout, si stima
  $\widehat{g}_L(f_{\text{abs}}) = \mathbb{E}[\text{death} \mid f_{\text{abs}},
  L]$ con regressione **isotonica** (death è monotòna non-crescente in
  $f_{\text{abs}}$ — hE1L-2, già confermata; l'isotonica impone *solo* questa
  forma, niente assunzione parametrica). Banda: percentili 5–95 dei residui
  dell'ablazione attorno a $\widehat{g}_L$, oppure binomiale di Wilson sul punto
  più vicino.
- **Residuo segnato.** Per ogni punto LLM: $\rho^{m,L} = \text{death}^{m,L} -
  \widehat{g}_L(f_{\text{abs}}^{m,L})$.
- **Test primario (hE1Lc-1 vs hE1Lc-2).** Wilcoxon signed-rank dei $\rho^{m,L}$
  contro 0, su tutti i punti LLM validi. Riportato con l'effect size (mediana del
  residuo) e IC bootstrap. Test per-layout in subordine (la banda è
  layout-specifica — $f_{\text{abs}}$ è un sommario lossy, §1 di E1_LLM_RESULT).
- **Membership per-punto.** Frazione di punti LLM che cadono entro la banda
  5–95% della loro nube-layout al loro $f_{\text{abs}}$.
- **Diagnostica della struttura dell'errore.** Per i punti LLM fuori banda:
  decomposizione falsi-negativi / falsi-positivi del probe; correlazione spaziale
  delle celle sbagliate con la rotta start→goal. Spiega il *segno* di hE1Lc-2.

La skill `statistical-analysis` viene usata in fase di analisi per la scelta del
test, il controllo delle assunzioni e il reporting (come per E2).

---

## 7. Criteri di successo / falsificazione

- **hE1Lc-1 supportata**: residuo Wilcoxon $p>0.05$, mediana entro $\pm 0.05$ di
  death rate → $f_{\text{abs}}$ sufficiente; la curva §5 è predittiva.
- **hE1Lc-2 supportata**: residuo Wilcoxon $p<0.05$ → la struttura dell'errore
  conta; si riporta segno e meccanismo. Esito *più* interessante per la
  Congettura E, non un fallimento.
- **Disegno non-conclusivo**: hE1Lc-4 non soddisfatta (nessuno spread di
  $f_{\text{abs}}$) → si riporta "task robustamente facile" e si propone Route A
  (dominio aperto) come unico modo di portare l'LLM dentro la curva.
- **Nessun esito ri-apre la Congettura E**: E1-LLM resta verificata. Questo
  esperimento ne *caratterizza la fragilità*, non la giudica.

---

## 8. Cosa E1-LLM-curve NON testa

- **Domini aperti / Route A** — qui il world-model è ancora distillato offline su
  dominio chiuso (schema S2). L'LLM interrogato online è il passo successivo.
- **Gli altri tre organi** (percezione, grounding, voce) — invariato da E1-LLM.
- **Il regime goal-directed** ($\alpha>0$) — escluso: il goal-seeking confonde il
  death rate. La banda e i punti LLM sono $\alpha=0$.
- **La capacità di coding degli LLM in generale** — la scala Llama è uno
  *strumento* per generare spread di $f_{\text{abs}}$, non l'oggetto di studio.

---

## Riferimenti

- [`E1_LLM_RESULT.md`](E1_LLM_RESULT.md) — E1-LLM verificata; §1 lo sweep
  $f_{\text{abs}}$ (curva a un sorteggio); §4 il caveat "$f_{\text{abs}}=1$ →
  test facile" che questo disegno scioglie.
- [`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md) — §3 probe $f_{\text{abs}}$, §5 sweep.
- [`P13_RESULT.md`](P13_RESULT.md) — hP13-1, struttura assorbente load-bearing.
- [`e1_llm_common.py`](e1_llm_common.py) — `make_ablated_transition`, `fabs_probe`,
  `WorldModelEnv`; [`e1_llm_client.py`](e1_llm_client.py) — gate di sicurezza,
  `chat`/`extract_code`/`compile_transition` riusati.
- **Tang, H. et al.** (2024). *Generating Code World Models with LLMs Guided by
  MCTS*. arXiv:2405.15383.

---

*Fine E1_LLM_CURVE_DESIGN.md. Design pre-registrato — nessun dato raccolto. La
domanda: $f_{\text{abs}}$ è una statistica sufficiente per la self-preservation,
o la struttura dell'errore del world-model LLM conta oltre la sua quantità?
Esecuzione: `e1_llm_curve.py`. Verdetto in un futuro `E1_LLM_CURVE_RESULT.md`.*
