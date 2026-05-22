# E1-LLM Route A-bis — eseguita: la persistenza non era il problema (2026-05-22)

Esecuzione del design pre-registrato [`E1_LLM_ROUTE_A_BIS_DESIGN.md`](E1_LLM_ROUTE_A_BIS_DESIGN.md):
imporre la persistenza assorbente nel **framework** (l'harness applica "done →
stay", l'LLM è interrogato solo per stati vivi) e ri-chiedere se la
self-preservation online si recupera.

Codice: [`e1_llm_route_a_bis.py`](e1_llm_route_a_bis.py). Dati: [`results/e1_llm_route_a_bis.json`](results/e1_llm_route_a_bis.json).
Kernel `fmc-core` invariato; cache di Route A riusata → **0 nuove chiamate API**.

> **Esito in una riga.** Imporre la persistenza dal framework **non recupera
> niente** — morte $38.9\%$ vs $35.0\%$ di Route A (hRAb-2 *e* hRAb-3
> falsificate). E così facendo Route A-bis **smaschera un errore di Route A**:
> il suo $f_{\text{abs}}=0.92$ era una metrica **non bilanciata**, dominata dal
> base-rate delle celle libere. Il probe **bilanciato** (lo stesso del sweep e
> di E1-LLM-curve) dà $f_{\text{abs}}\approx\mathbf{0.54}$ su tutti i 6 layout —
> al **floor del caso**. La diagnosi vera di Route A non è la persistenza: è che
> **l'LLM online non costruisce affatto un modello assorbente fedele** —
> l'entry-detection è a livello di caso. §3 corregge Route A.

---

## 1. Il delta e il risultato

Un solo cambiamento rispetto a Route A: se `s.done` è `True` l'harness
restituisce `(r,c,True)` senza interrogare l'LLM (persistenza per costruzione);
l'LLM è interrogato solo per stati vivi. Tutto il resto identico. La cache di
Route A (voci `done=0`) copre tutte le query → **0 nuove chiamate API**, ~2 min.

**Death rate (fmc α=0), Route A-bis vs Route A vs baseline:**

| layout | A-bis | Route A | random | greedy |
|---|---:|---:|---:|---:|
| gauntlet | 53% | 70% | 67% | 97% |
| lake | 70% | 33% | 67% | 100% |
| scatter | 87% | 80% | 97% | 93% |
| island | 3% | 3% | 3% | 23% |
| spur | 0% | 0% | 3% | 0% |
| archipelago | 20% | 23% | 23% | 50% |
| **pooled** | **38.9%** (70/180) | **35.0%** | **43.3%** | — |

- **hRAb-2 (recupero direzionale): FALSIFICATA.** $38.9\% > 35.0\%$ — la morte
  *non* scende; è anzi marginalmente più alta (rumore). Imporre la persistenza
  non aiuta.
- **hRAb-3 (recupero pieno): FALSIFICATA.** $0/6$ layout significativi; pooled vs
  random $z=-0.86$, $p=0.39$ — Route A-bis non è nemmeno significativamente
  meglio del random.

**La persistenza assorbente non era il fattore dominante.** Imposta a $1.0$ per
costruzione, il merge online resta rotto esattamente come prima.

---

## 2. Perché — il probe bilanciato

Route A-bis misura $f_{\text{abs}}$ col probe **bilanciato** ([`e1_llm_common.fabs_probe`](e1_llm_common.py),
50% sonde che atterrano su lava/goal + 50% su libere) — lo stesso del sweep §5 di
E1-LLM e di E1-LLM-curve:

| layout | $f_{\text{abs}}$ (bilanciato) | move-fidelity | persistenza |
|---|---:|---:|---:|
| gauntlet | 0.54 | 0.94 | 1.00 (imposta) |
| lake | 0.56 | 0.86 | 1.00 |
| scatter | 0.52 | 0.90 | 1.00 |
| island | 0.54 | 0.99 | 1.00 |
| spur | 0.53 | 0.97 | 1.00 |
| archipelago | 0.54 | 0.99 | 1.00 |
| **media** | **0.538** | **0.94** | **1.000** |

$f_{\text{abs}}\approx 0.54$ è praticamente il **floor del caso** del probe
bilanciato (0.5 = nessuna conoscenza assorbente — metà sonde indovinate per il
solo base-rate). Il `terminal_recall` è ~0.07–0.10: **l'LLM online, quando un
walker entra in una cella terminale, la riconosce come tale solo ~1 volta su
10.** Il movimento ($0.94$) regge; l'entry-detection no.

---

## 3. Correzione di E1_LLM_ROUTE_A_RESULT.md

Route A-bis impone una correzione onesta a [`E1_LLM_ROUTE_A_RESULT.md`](E1_LLM_ROUTE_A_RESULT.md),
documento già committato.

**Cosa Route A diceva (sbagliato):** *"$f_{\text{abs}}=0.92$ e movimento $0.94$
reggono; la persistenza assorbente collassa a $0.53$ — è l'asse load-bearing."*

**Cosa è realmente vero:**

1. **Il $f_{\text{abs}}=0.92$ di Route A era una metrica non-bilanciata.** Il
   suo `fidelity_probe` calcolava $f_{\text{abs}}$ come frazione di *tutte* le
   $(cella,azione)$ con `ndone` corretto — dominata dal base-rate (~90% delle
   coppie atterra su celle libere, banali). Il probe **bilanciato** — quello che
   tutto il resto del progetto usa (sweep, E1-LLM-curve) — dà $f_{\text{abs}}
   \approx 0.54$. È un mio errore nello scrivere `e1_llm_route_a.py`: una metrica
   incoerente con il canone del progetto.
2. **L'entry-detection NON regge** — è al floor del caso. L'LLM online non
   riconosce la lava come terminale all'ingresso (recall ~0.07).
3. **La persistenza ($0.53$) non era load-bearing.** Era un numero reale, ma
   imporla a $1.0$ (Route A-bis) non cambia nulla — perché l'LLM raramente
   *flagga* l'ingresso terminale in primo luogo, quindi pochi walker diventano
   `done` nel modello e la persistenza è in gran parte muta.

**Cosa di Route A resta valido:** il **verdetto** — il merge FMC+LLM regge
offline (Route B, $0/180$) ma **non online** (morte ~35–39%, ~livello random) —
è corretto e confermato da Route A-bis. Cambia solo la *diagnosi del meccanismo*:
non la persistenza, ma l'**entry-detection** al floor del caso.

---

## 4. La diagnosi corretta — e perché online ≪ offline

Il contrasto vero, ora pulito: **stesso modello (Llama 3.3 70B)**,

- **offline / code-form** (Route B, E1-LLM): $f_{\text{abs}}$ bilanciato
  $=\mathbf{1.000}$ — world-model assorbente perfetto;
- **online / per-query da osservazione locale** (Route A): $f_{\text{abs}}$
  bilanciato $\approx\mathbf{0.54}$ — al floor del caso.

Perché un crollo così? In Route B l'LLM riceveva le **regole** ("lava e goal sono
assorbenti — stuck there permanently") e le *trascriveva* in codice → corretto.
In Route A l'LLM **non riceve le regole**: da un'osservazione locale ("sei su
ground, lava a destra") deve *inferire* se entrare nella lava termina la run.
Gli smoke test lo mostrano nitido: l'LLM inferisce "goal-entry → run finita"
(goal = obiettivo, conoscenza generale) ma **non** "lava-entry → run finita" —
tratta la lava col **prior sbagliato**: un *ostacolo da evitare*, non una tile
*letale-terminale*. Il prior dell'LLM sulla lava ("è pericolosa, non ci vai")
**confligge con la regola di questo mondo** ("ci entri e la run finisce"), e
senza la regola enunciata vince il prior. Code-form funziona perché la regola
era data; l'inferenza online da percezione locale fallisce perché l'LLM porta la
sua semantica, non quella del mondo.

**hRAb-4.** Il sweep $f_{\text{abs}}$ predice, a $f_{\text{abs}}=0.54$, morte
~$64\%$ (ablazione casuale). Route A-bis osserva $38.9\%$ — *meno* della
predizione. Gli errori dell'LLM a $f_{\text{abs}}=0.54$ sono quindi un po' meno
avversariali dell'ablazione casuale a parità di fedeltà (alcuni errori cadono su
lava fuori-rotta, innocui). Ma $38.9\%$ resta ~livello random: non un recupero,
solo un fallimento un po' meno netto del peggior caso.

---

## 5. Verdetto e conseguenze

| ipotesi | esito |
|---|---|
| hRAb-2 recupero direzionale | ✗ falsificata (38.9% ≥ 35.0%) |
| hRAb-3 recupero pieno | ✗ falsificata (0/6 sig) |
| hRAb-4 il sweep predice il residuo | parziale (A-bis meno letale della predizione) |

**Netto.** La persistenza assorbente **non** era il blocco del merge online.
Imporla dal framework non recupera niente. Il blocco vero è che **l'LLM online,
interrogato per-query da osservazione locale senza le regole, non costruisce un
modello assorbente migliore del caso** ($f_{\text{abs}}\approx 0.54$) — perché
applica il proprio prior sulla lava invece della regola del mondo.

**Conseguenza per la Congettura E.** Il merge FMC-core + LLM-world-model-organo
regge **offline** (l'LLM trascrive regole date in codice) ma **non online** da
percezione locale — e il confine non è un invariante da imporre meglio, è la
**capacità dell'LLM di inferire le dinamiche terminali del mondo specifico senza
che gli siano dette**. Vie avanti possibili (non più "imporre la persistenza"):
(a) dare all'LLM la *regola* esplicita su lava/goal — ma allora è Route B
travestita; (b) un dominio dove i prior dell'LLM *coincidono* con le regole;
(c) un LLM-organo di percezione che etichetti le tile con la loro semantica
*operativa* (questa-tile-termina-la-run) prima di passarle al world-model. Tutte
fuori dallo scope di Route A.

**Onestà di processo.** Route A aveva una metrica $f_{\text{abs}}$ incoerente col
canone del progetto; ha prodotto una diagnosi sbagliata ("persistenza
load-bearing"). Route A-bis — nato per *testare* quella diagnosi — l'ha
falsificata e, col probe bilanciato corretto, ha rivelato il vero meccanismo. È
il valore di eseguire l'ipotesi invece di assumerla: un esperimento pre-registrato
per confermare una diagnosi l'ha invece corretta.

---

## 6. Riproducibilità

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" -u work/12_conjecture_e/e1_llm_route_a_bis.py    # ~2 min, cache-warm, 0 API
```

---

*Fine E1_LLM_ROUTE_A_BIS_RESULT.md. Imporre la persistenza dal framework non
recupera la self-preservation online (morte 38.9% ≈ Route A 35%, hRAb-2/3
falsificate). Il probe bilanciato rivela $f_{\text{abs}}\approx 0.54$ — al floor
del caso: il $0.92$ di Route A era una metrica non-bilanciata. Diagnosi corretta:
il merge online fallisce perché l'LLM, da osservazione locale e senza le regole,
non riconosce la lava come terminale (recall ~0.07) — applica il prior "ostacolo"
invece della regola "letale-terminale". Il merge FMC+LLM regge offline (regole
trascritte in codice), non online (inferenza da percezione locale).*
