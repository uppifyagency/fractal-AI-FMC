# E1-LLM Route A-ter — eseguita: rinominare non recupera l'entry-detection (2026-05-24)

Esecuzione del design pre-registrato [`E1_LLM_ROUTE_A_TER_DESIGN.md`](E1_LLM_ROUTE_A_TER_DESIGN.md):
rinominare la tile terminale di morte da `lava` a `pit` per distinguere le due
sotto-cause diagnostiche di Route A-bis — (i) mismatch *semantico* del prior LLM
("lava = ostacolo da evitare" invece di "tile letale-terminale"), vs (ii)
confound *strutturale* "saggezza vs predizione" (l'LLM rifiuta di predire che un
agente *entri* in una tile pericolosa indipendentemente dal nome).

Codice: [`e1_llm_route_a_ter.py`](e1_llm_route_a_ter.py). Dati: [`results/e1_llm_route_a_ter.json`](results/e1_llm_route_a_ter.json).
Kernel `fmc-core` invariato. Nuovo nome-tile → cache di Route A non riusabile →
**704 nuove chiamate API** (free-tier rate-limited a 3s/call con backoff →
wall-time effettivo ~43 h).

> **Esito in una riga.** Rinominare la tile terminale `lava` → `pit` (un nome il
> cui prior LLM "ci si cade dentro" coincide con la regola del mondo) **non
> recupera l'entry-detection**: il $f_{\text{abs}}$ bilanciato sale da $0.54$
> (Route A) a $\mathbf{0.59}$ — ancora ben sotto la soglia pre-registrata $0.80$;
> il death rate online resta a $\mathbf{39.4\%}$, $0/6$ layout significativi
> (hRAt-1 ✗, hRAt-2 ✗). Sotto-causa (ii) supportata: **l'LLM, interrogato
> per-query da osservazione locale, rifiuta di predire che un agente entri in una
> tile pericolosa indipendentemente dal nome** — un confound saggezza-vs-predizione
> strutturale al world-model online. La via del semplice rename è chiusa.

---

## 1. Risultato — fedeltà a 3 assi (probe bilanciato)

| layout | $f_{\text{abs}}$ | terminal recall | move-fidelity | done-persistence |
|---|---:|---:|---:|---:|
| gauntlet | 0.62 | 0.24 | 0.94 | 0.56 |
| lake | 0.60 | 0.19 | 0.88 | 0.79 |
| scatter | 0.57 | 0.14 | 0.90 | 0.29 |
| island | 0.60 | 0.20 | 0.99 | 0.60 |
| spur | 0.55 | 0.09 | 0.97 | 0.69 |
| archipelago | 0.64 | 0.28 | 0.99 | 0.28 |
| **media** | **0.59** | **0.19** | **0.94** | **0.54** |

Confronto con Route A (lava, stessa metrica bilanciata): $f_{\text{abs}}=0.54$.

- **hRAt-1 (recupero entry-detection, soglia $f_{\text{abs}}\geq 0.80$): FALSIFICATA.**
  Media $0.59$ — un guadagno di $+0.05$ rispetto a `lava`, ben lontano dalla
  soglia. Il `terminal_recall` resta a $0.19$ (vs $\sim 0.07$ di Route A) —
  marginalmente meglio, ma sempre dominato dai falsi negativi: l'LLM continua a
  *non flaggare* l'ingresso terminale ~$80\%$ delle volte.
- **hRAt-4 (movimento invariato): ✓.** $0.94$ — controllo di sanità superato; il
  movimento non dipendeva dal nome della tile.
- **hRAb-1 (persistenza assorbente — non imposta qui per misurarla): $0.54$**, in
  linea con Route A. Confounder atteso a $f_{\text{abs}}$ basso (pochi `done`
  flaggati → la persistenza è in gran parte muta).

---

## 2. Death rate (fmc α=0) vs baseline

| layout | $\text{fmc}_{\alpha=0}$ | random | greedy | $z$ vs random | $p$ |
|---|---:|---:|---:|---:|---:|
| gauntlet | 67% | 77% | 93% | −0.86 | 0.39 |
| lake | 57% | 77% | 100% | −1.64 | 0.10 |
| scatter | 87% | 97% | 100% | −1.40 | 0.16 |
| island | 7% | 3% | 10% | +0.59 | 0.55 |
| spur | 0% | 3% | 0% | −1.01 | 0.31 |
| archipelago | 20% | 23% | 50% | −0.31 | 0.75 |
| **pooled** | **39.4%** (71/180) | **46.7%** | — | **−1.38** | **0.17** |

- **hRAt-2 (recupero self-preservation, criterio hRA-3: death $\leq$ baseline,
  significativo, $\geq 3$ layout): FALSIFICATA.** $0/6$ significativi al $\alpha=0.05$.
  La $\text{fmc}_{\alpha=0}$ è direzionalmente sotto random in $5/6$ layout, ma il
  margine è piccolo e mai significativo — *nessun* recupero. Pooled $39.4\%$ vs
  Route A $35.0\%$: equivalenti nel rumore.

---

## 3. Verdetto e diagnosi — sotto-causa (ii) supportata

Le due sotto-cause pre-registrate erano:

- **(i) Mismatch semantico** (predizione di hRAt-1+hRAt-2): la parola "lava"
  specificamente evocava il prior "muro/evita"; con un nome il cui prior
  coincide con la regola del mondo ("pit" — ci si cade dentro), l'entry-detection
  si recupera ($f_{\text{abs}}\geq 0.80$, death $\to$ regime offline).
- **(ii) Confound strutturale "saggezza vs predizione"** (predizione di hRAt-3):
  l'LLM rifiuta di predire che un agente *entri* in una tile pericolosa
  *qualunque* sia il nome — confonde "cosa succede SE fa l'azione" (world-model)
  con "cosa CONVERREBBE fare" (giudizio dell'agente). Rinominare non aiuta.

**I dati sostengono (ii).** Il rename ha prodotto un guadagno marginale
($+0.05$ in $f_{\text{abs}}$, $+0.12$ in terminal recall), ma né l'entry-detection
si avvicina alla soglia pre-registrata né il death rate si avvicina al regime
offline. L'LLM continua sistematicamente a non flaggare l'ingresso in una tile
pericolosa indipendentemente dal nome — `pit` o `lava`, il pattern è il
medesimo: predice il movimento dell'agente verso il pericolo *senza* la
conclusione che la run termina.

Questo è il confound (ii) in forma operativa: il world-model LLM per-query, da
osservazione locale, **mescola la dinamica del mondo con il giudizio normativo
dell'agente** — risponde "non ci andrebbe" (saggezza) invece di "ci va, e la run
finisce" (predizione condizionata sull'azione data). Il prompt di Route A-ter
era identico a Route A nella struttura, *eccetto* il nome — quindi è il
*compito* (predire l'ingresso in pericolo), non il *vocabolario*, a rompere.

---

## 4. Conseguenze — il confine online del merge è strutturale, non semantico

**Per la Congettura E.** Il merge FMC-core + LLM-world-model-organo:

- regge **offline** (Route B, regole date → codice): $f_{\text{abs}}=1.0$,
  death $0/180$ — confermato da E1-LLM;
- fallisce **online** per-query da percezione locale: **il confine non è
  semantico** (Route A-ter lo prova — rinominare non aiuta), **è strutturale**
  al world-model online quando lo si interroga per-(stato,azione) da osservazione
  locale senza regole esplicite.

**Vie costruttive sopravvissute** (dopo Route A → A-bis → A-ter):

- **(a) Regola esplicita all'LLM su tile pericolose** → è Route B travestita,
  perde l'interesse "open-domain";
- **(b) Dominio dove i prior dell'LLM coincidono con le regole** — non
  gridworld, vero open-domain (es. coding/proof, dove la semantica delle azioni
  è già nel prior);
- **(c) Organo di percezione che etichetti le tile operativamente** (questa-tile-
  termina-la-run) prima del world-model — separa la *percezione* dalla
  *predizione*, aggira il confound saggezza/predizione.

Le tre vie sono progetti nuovi, fuori dallo scope di Route A. Route A come
*thread* è concluso con i suoi tre varianti pre-registrate esaurite: A (test
originale), A-bis (test della prima diagnosi — falsificata, persistenza non
load-bearing), A-ter (test della seconda diagnosi — sotto-causa (i) falsificata,
sotto-causa (ii) supportata).

---

## 5. Verdetto in tabella

| ipotesi | esito |
|---|---|
| hRAt-1 ($f_{\text{abs}}\geq 0.80$ con `pit`) | ✗ falsificata ($0.59 < 0.80$) |
| hRAt-2 (death recupera) | ✗ falsificata ($0/6$ sig, $39.4\% \approx 35.0\%$) |
| hRAt-3 (confound saggezza/predizione strutturale) | ✓ supportata (sotto-causa (ii)) |
| hRAt-4 (movimento invariato) | ✓ confermata ($0.94$) |

---

## 6. Onestà di processo

- Lo smoke test pre-run aveva osservato il pattern in cinque chiamate: "agente
  su ground, pit a destra, azione 'right' → ('stay', False)". L'LLM aveva già
  rifiutato di predire l'ingresso nella pit; il run completo a $n=30$ x $6$
  layout x $\sim 700$ query lo ha confermato statisticamente.
- Il design pre-registrato (E1_LLM_ROUTE_A_TER_DESIGN §4) aveva un criterio di
  lettura simmetrico: hRAt-3 era contemplata come esito *alternativo* fin
  dall'inizio. Il verdetto non è una sorpresa, è una distinzione testata fra due
  ipotesi specificate ex-ante.
- Wall-time $\sim 43$ h dovuto al rate-limiting NVIDIA free-tier (pacing 3s +
  backoff esponenziale). L'harness fail-loud non ha fabbricato alcuna risposta —
  $704$ chiamate API, $0$ fallback fittizi.

---

## 7. Riproducibilità

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
export NVIDIA_API_KEY=$(security find-generic-password -s fractalai-nvidia-api -w)
"$PY" -u work/12_conjecture_e/e1_llm_route_a_ter.py
# wall-time effettivo ~43 h (free-tier rate-limited); 704 API calls
```

---

*Fine E1_LLM_ROUTE_A_TER_RESULT.md. Rinominare la tile terminale `lava` → `pit`
(un nome il cui prior LLM coincide con la regola del mondo) non recupera
l'entry-detection: $f_{\text{abs}}$ bilanciato sale da $0.54$ a $0.59$ — ben
sotto la soglia pre-registrata $0.80$ — e la self-preservation non si recupera
($0/6$ layout significativi, $39.4\% \approx 35.0\%$ di Route A). hRAt-1 e
hRAt-2 falsificate, hRAt-3 supportata: l'LLM rifiuta di predire l'ingresso in
una tile pericolosa indipendentemente dal nome — un confound saggezza-vs-predizione
strutturale al world-model online. Il confine "offline regge / online fallisce"
del merge FMC+LLM è strutturale, non semantico. Route A è concluso (tre
varianti pre-registrate esaurite). Vie costruttive sopravvissute fuori dallo
scope: dominio open dove i prior LLM coincidono con le regole, o organo di
percezione che etichetti le tile operativamente prima del world-model.*
