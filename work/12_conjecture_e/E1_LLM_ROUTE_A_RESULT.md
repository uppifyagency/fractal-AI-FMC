# E1-LLM Route A — eseguita: il merge regge offline, non online (2026-05-22)

Esecuzione del design pre-registrato [`E1_LLM_ROUTE_A_DESIGN.md`](E1_LLM_ROUTE_A_DESIGN.md):
il world-model LLM interrogato **online**, durante la pianificazione FMC, da
**osservazioni locali** — non da una specifica globale delle regole.

Codice: [`e1_llm_route_a.py`](e1_llm_route_a.py). Dati: [`results/e1_llm_route_a.json`](results/e1_llm_route_a.json),
cache delle query [`results/route_a_cache.json`](results/route_a_cache.json).
Kernel `fmc-core` invariato; `WorldModelEnv` riusato via `LLMWorldModelOnlineEnv`.

> **Esito in una riga.** Il merge FMC+LLM **regge offline** (Route B / E1-LLM:
> world-model scritto come codice, morte 0/180) ma **non online** (Route A:
> world-model interrogato per-query in linguaggio naturale, morte **35%**). La
> causa è precisa e diagnosticata: dei tre assi di fedeltà, entry-detection
> ($f_{\text{abs}}=0.92$) e movimento ($0.94$) reggono, ma la **persistenza
> assorbente collassa a $0.53$** — un walker già "finito" *esce* dallo stato
> assorbente circa metà delle volte. È esattamente l'asse load-bearing di
> P13/hP13-1 ed E1-LLM-curve. **hRA-3 falsificata; il merge è delimitato al
> regime offline-code.**

> ## ⚠️ CORREZIONE (2026-05-22) — la diagnosi qui sotto è in parte SBAGLIATA
>
> [`E1_LLM_ROUTE_A_BIS_RESULT.md`](E1_LLM_ROUTE_A_BIS_RESULT.md) corregge questo
> documento su un punto centrale. **(1)** Il $f_{\text{abs}}=0.92$ qui riportato
> viene da una metrica **non bilanciata** (`fidelity_probe`, dominata dal
> base-rate ~90% di celle libere) — incoerente col canone del progetto. Il probe
> **bilanciato** (lo stesso di sweep ed E1-LLM-curve) dà $f_{\text{abs}}\approx
> \mathbf{0.54}$, al **floor del caso**: l'entry-detection dell'LLM online **non
> regge** (recall terminale ~0.07). **(2)** La persistenza ($0.53$) **non era
> load-bearing**: Route A-bis l'ha imposta a $1.0$ dal framework e la morte
> **non è scesa** ($38.9\%$ vs $35\%$). Il **verdetto** di questo documento
> resta valido — il merge regge offline, non online — ma la **diagnosi del
> meccanismo** è rivista: non la persistenza, bensì l'entry-detection al floor
> del caso (l'LLM online, senza le regole, modella la lava col prior "ostacolo"
> e non "letale-terminale"). Le §3–§5 qui sotto vanno lette con la correzione di
> [`E1_LLM_ROUTE_A_BIS_RESULT.md`](E1_LLM_ROUTE_A_BIS_RESULT.md) §3–§4.

---

## 0. Validità — due run scartati prima di questo

Onestà di processo. I primi due tentativi sono stati **scartati**, non sono
questo risultato:

1. **Run 1 — invalidato.** Il free-tier NVIDIA ha rate-limitato dopo ~centinaia
   di chiamate; l'harness ingoiava le eccezioni e sostituiva un costante
   `("stay", done)` → **97% della cache era fallback fabbricato**, non risposte
   dell'LLM. Verdetto "VERIFIED" stampato — *artefatto*. Scartato.
2. **Run 2 — abortito (correttamente).** Harness corretto (pacing 3 s, backoff,
   **fail-loud** invece di fabbricare). Un blip di rete (`APIConnectionError`)
   ha esaurito i 6 retry → il run è **abortito invece di mentire**.
3. **Run 3 — questo.** Harness reso resiliente (retry 10×, backoff fino a 90 s,
   checkpoint della cache *a ogni query*) e ripreso dalla cache di 420 voci.
   Completato pulito: `parse failures: 0`, nessuna fabbricazione.

Il fallback fabbricato del run 1 produceva `move-fidelity 0.32` e `persistence
1.00` — entrambi artefatti. I numeri di questo documento sono le **prime misure
vere** del world-model LLM online.

---

## 1. hRA-1 — costo $R1$: trattabile

| metrica | valore |
|---|---|
| query LLM **distinte** | 660 (tetto pre-registrato 4000) |
| chiamate API nel test FMC | **0** (cache 100% calda dopo il probe) |
| cache hit rate | 100% |
| parse failures / fabbricazioni | 0 |

**hRA-1 confermata.** La query dipende solo dall'osservazione locale, non dalle
coordinate; la cache satura sullo spazio (piccolo) delle osservazioni. Le 660
query distinte coprono tutto ciò che FMC chiede → il test FMC aggiunge **zero**
chiamate. Il muro $R1$ ($N\cdot M$ chiamate/decisione) è **trattabile con la sola
cache** su un dominio a meccaniche chiuse. Caveat onesto: questo vale perché lo
spazio delle osservazioni locali è limitato; su un dominio genuinamente aperto
(osservazioni non ripetute) la cache non saturerebbe — lì servirebbe lo schema
sparso S1.

---

## 2. hRA-2 — consistenza: alta ma sotto soglia

A temperatura 0, ri-interrogando 132 query: **126/132 = 0.955** identiche.
Pre-registrato $\geq 0.98$ → **hRA-2 tecnicamente falsificata**. Le 6 query che
cambiano risposta sono non-determinismo residuo dell'API/modello — un world-model
che *cambia sotto i piedi* di FMC, un modo di instabilità che Route B (codice,
deterministico per costruzione) non ha. È reale ma **non è la causa** del
fallimento di Route A (la persistenza lo è); resta un costo registrato dell'online.

---

## 3. Il gate a 4 assi — la persistenza assorbente collassa

| layout | f_abs | move-fidelity | done-persistence |
|---|---:|---:|---:|
| gauntlet | 0.92 | 0.94 | 0.56 |
| lake | 0.84 | 0.86 | 0.79 |
| scatter | 0.87 | 0.90 | **0.29** |
| island | 0.98 | 0.99 | 0.60 |
| spur | 0.95 | 0.97 | 0.67 |
| archipelago | 0.98 | 0.99 | **0.28** |
| **media** | **0.92** | **0.94** | **0.53** |

- **entry-detection ($f_{\text{abs}}=0.92$) e movimento ($0.94$) reggono.** Il
  70B online, da osservazione locale, riconosce le celle terminali all'ingresso
  e si muove correttamente. (Smentisce la preview errata del run-1 artefatto:
  il movimento *non* è rotto.)
- **La done-persistence collassa a $0.53$** — un walker già su una cella
  assorbente (`done=True`), interrogato "run OVER, passo X", circa **metà delle
  volte risponde che si muove e la run continua**. Nel rollout interno di FMC le
  celle assorbenti diventano *semi-permeabili*: un walker entra nella lava e ne
  riesce → FMC sotto-rappresenta il pericolo.

**Meccanismo.** In Route B il 70B *scriveva codice* — `if done: return r,c,True`
— una riga, l'invariante assorbente imposto *strutturalmente*. Online, interrogato
per-query in linguaggio naturale ("la run è OVER, passo scelto: move up"), l'LLM
**risponde all'azione** e non onora la pre-condizione. Il codice non può
editorializzare; la prosa per-query sì. È il finding profondo di Route A: **un
world-model LLM online non mantiene gli invarianti di stato che il codice impone
per costruzione.**

---

## 4. hRA-3 — self-preservation online: falsificata

FMC pianifica su `LLMWorldModelOnlineEnv`; episodio sul simulatore vero;
$\alpha\in\{0,0.1\}$, $\beta=1$, $N=64$, $M=20$, $n=30$. Death rate:

| layout | fmc α=0 | random | greedy | vs random |
|---|---:|---:|---:|---|
| gauntlet | **70%** | 63% | 90% | z=+0.55 — *peggio del random* |
| lake | 33% | 83% | 100% | z=−3.93, p<0.001 ✓ |
| scatter | 80% | 97% | 100% | z=−2.01, p=0.044 ✓ |
| island | 3% | 10% | 10% | z=−1.04, n.s. |
| spur | 0% | 0% | 0% | tie (lava fuori rotta) |
| archipelago | 23% | 40% | 43% | z=−1.39, n.s. |
| **pooled** | **35.0%** (63/180) | **48.9%** (88/180) | — | z=−2.67, p=0.0076 |

Soglia pre-registrata (hRA-3): death $\leq$ entrambe le baseline, significativo,
su $\geq 3$ layout. Risultato: **2/6** (lake, scatter). Su **gauntlet FMC è
peggio del random**. **hRA-3 falsificata.**

Lettura onesta: non è un collasso totale — pooled $35\% < 49\%$ del random
($p=0.008$): il world-model online porta *un po'* di segnale. Ma è lontanissimo
dallo **0/180** di E1-LLM offline. Il merge che offline tiene la morte a zero,
online la lascia al 35%.

---

## 5. hRA-4 — il gate a tre assi spiega tutto

**hRA-4 confermata.** Il fallimento di Route A non è generico: è *esattamente*
sull'asse che P13/hP13-1 ed E1-LLM-curve avevano già isolato come **load-bearing
e dominante** — la persistenza assorbente. f_abs e movimento (alti) non bastano;
la persistenza (0.53) è ciò che cede, e la self-preservation cede con lei. Il
death rate non è perfettamente predetto dalla sola persistenza per-layout (spur
0.67→morte 0% perché la lava è fuori rotta; la geometria conta, come in
E1-LLM-curve $f_{\text{abs}}$ era un sommario lossy) — ma la direzione è netta:
**dove la persistenza è più bassa (scatter 0.29, archipelago 0.28) il world-model
tratta la lava come semi-permeabile e FMC ne paga il prezzo.**

---

## 6. Verdetto e conseguenze

**Route A falsificata (hRA-3)** — con diagnosi costruttiva, non un vicolo cieco:

| ipotesi | esito |
|---|---|
| hRA-1 costo $R1$ trattabile | ✓ (cache → 0 chiamate nel test FMC) |
| hRA-2 consistenza $\geq 0.98$ | ✗ tecnicamente (0.955; alta, non load-bearing) |
| hRA-3 self-preservation online | ✗ **falsificata** (morte 35%, 2/6 layout) |
| hRA-4 il gate a 3 assi governa | ✓ (cede la persistenza, l'asse load-bearing) |

**Cosa stabilisce.** Il merge FMC-core + LLM-world-model-organo **regge offline**
(Route B: l'LLM scrive il codice del world-model — E1-LLM, morte 0/180) ma **non
online** nella forma per-query in linguaggio naturale: l'LLM non mantiene
l'invariante di persistenza assorbente, e FMC pianifica su una lava
semi-permeabile. Il merge, per ora, è **delimitato al regime offline-code**.

**Il percorso avanti** (non un fallimento — una via precisa):

1. **Persistenza imposta dal framework, non dall'organo.** Uno stato terminale è
   assorbente *per definizione di FMC* — non è conoscenza del mondo da chiedere
   all'LLM. Se l'harness applica "done → stay" strutturalmente e interroga l'LLM
   **solo per stati vivi** (`done=False`), la persistenza diventa 1.0 per
   costruzione. Predizione testabile: Route A-bis con questo fix → la morte
   sarebbe guidata solo da $f_{\text{abs}}$ e movimento (entrambi $\geq 0.92$);
   resta da vedere se basta (il sweep $f_{\text{abs}}$ di E1-LLM dice che serve
   fedeltà *quasi*-perfetta — $0.92$ potrebbe non bastare).
2. **Online code-form**: l'LLM emette codice incrementale anziché risposte
   per-query — recupera l'imposizione strutturale degli invarianti di Route B.
3. **Route A-2**: un dominio genuinamente aperto (non-gridworld), dove anche
   l'argomento della cache cade e serve lo schema sparso S1.

**Non ri-apre la Congettura E** (chiusa: E1-base, E2, E1-LLM). Route A ne
*delimita il perimetro empirico*: il merge è dimostrato offline, falsificato
online-per-query, con la persistenza assorbente come confine netto.

---

## 7. Riproducibilità

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" -u work/12_conjecture_e/e1_llm_route_a.py    # ~105 min (free-tier paced)
# chiave NVIDIA nel Keychain: security find-generic-password -s fractalai-nvidia-api -w
# resumabile: la cache route_a_cache.json è checkpointata a ogni query.
```

---

*Fine E1_LLM_ROUTE_A_RESULT.md. Il merge FMC+LLM regge offline (Route B, morte
0/180) ma non online (Route A, morte 35%). Costo trattabile (hRA-1 ✓), consistenza
0.955, fedeltà entry-detection 0.92 e movimento 0.94 — ma persistenza assorbente
0.53: online, l'LLM interrogato per-query non mantiene l'invariante "uno stato
terminale resta terminale" che il codice di Route B imponeva strutturalmente.
hRA-3 falsificata; hRA-4 confermata (cede l'asse load-bearing). Via avanti:
imporre la persistenza nel framework e interrogare l'LLM solo per stati vivi.*
