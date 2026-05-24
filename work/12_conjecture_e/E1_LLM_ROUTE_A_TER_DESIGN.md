# E1-LLM Route A-ter — design pre-registrato: test di allineamento del prior

> **Tipo**: design doc pre-registrato. Scritto *prima* di qualsiasi esecuzione e
> di qualsiasi sguardo ai dati.
> **Data**: 2026-05-22.
> **Estende**: [`E1_LLM_ROUTE_A_RESULT.md`](E1_LLM_ROUTE_A_RESULT.md) e
> [`E1_LLM_ROUTE_A_BIS_RESULT.md`](E1_LLM_ROUTE_A_BIS_RESULT.md). Specifica
> **solo il delta** rispetto a Route A.

---

## 1. Perché

Route A-bis ha corretto la diagnosi di Route A: il merge FMC+LLM fallisce online
**all'entry-detection** — il $f_{\text{abs}}$ bilanciato dell'LLM online è
$\approx 0.54$, al floor del caso (terminal recall ~0.07). La causa proposta:
l'LLM, interrogato per-query da osservazione locale **senza le regole**, modella
la tile `lava` col **prior sbagliato** — "ostacolo da evitare" (un muro da cui si
rimbalza) invece della regola di questo mondo, "tile letale-terminale su cui si
*entra* e la run finisce".

Questa è una diagnosi, non ancora un fatto dimostrato. Route A-ter la **testa
direttamente** e distingue due sotto-cause:

- **(i) Mismatch semantico.** È la parola "lava" specificamente a evocare il
  prior "muro/evita". Con una tile il cui prior coincide con la regola
  ("ci entri → la run finisce"), l'entry-detection si recupera.
- **(ii) Confound più profondo.** L'LLM si rifiuta di predire che un agente
  *entri* in una tile pericolosa, *qualunque* sia il nome — confonde il compito
  del world-model ("cosa succede *se* fa l'azione") con il giudizio dell'agente
  ("cosa *converrebbe* fare"). Allora rinominare non aiuta.

Distinguere (i) da (ii) è il valore di Route A-ter: se (i), il merge online
**funziona** col vocabolario giusto — un positivo condizionale; se (ii), il
world-model LLM per-query ha un limite strutturale "saggezza vs predizione".

---

## 2. Il delta — esattamente una cosa

Identico a Route A in **tutto** (6 layout, `meta/llama-3.3-70b-instruct`, prompt
finalizzato, $\alpha\in\{0,0.1\}$, $\beta=1$, $N=64$, $M=20$, $n=30$, episodi sul
simulatore vero, death rate, kernel `fmc-core` invariato, harness indurito
pacing/backoff/fail-loud/checkpoint) **tranne il nome della tile terminale di
morte**:

- **Route A**: la tile di codice 1 è descritta all'LLM come **`lava`**.
- **Route A-ter**: la tile di codice 1 è descritta come **`pit`** (una buca).
  Il prior LLM di "pit": *ci si cade dentro* — si *entra* (movimento normale) e
  l'esito è terminale. Coincide, per ipotesi, con la regola del mondo.

`ground` e `goal` invariati. La meccanica del gridworld è invariata: la tile 1
resta assorbente-terminale, esattamente come in E1/Route A — cambia **solo come
l'LLM la sente nominare**. Il `probe di fedeltà` usa il probe **bilanciato**
([`fabs_probe`](e1_llm_common.py)) — la metrica corretta del canone, non quella
non-bilanciata di Route A.

**Costo.** Nome-tile nuovo → osservazioni locali nuove → la cache di Route A
**non** si riusa: ~660 query LLM nuove, run paced ~100 min (free-tier). Harness
robusto (resumibile, checkpoint per-query).

---

## 3. Ipotesi pre-registrate

- **hRAt-1 (recupero dell'entry-detection — sotto-causa (i)).** Con la tile
  terminale chiamata `pit`, il $f_{\text{abs}}$ bilanciato dell'LLM online
  **risale nettamente** sopra il floor del caso — soglia pre-registrata
  $f_{\text{abs}} \geq 0.80$ (media sui 6 layout), contro lo $0.54$ di Route A.
  → il fallimento di Route A era **semantico**: il prior su "lava".

- **hRAt-2 (recupero della self-preservation).** Se hRAt-1 vale, il death rate
  online di Route A-ter scende verso il regime offline: death $\leq$ entrambe le
  baseline, significativo, su $\geq 3$ layout (criterio hRA-3). → **il merge
  FMC+LLM online funziona** quando il vocabolario delle tile è allineato ai prior
  dell'LLM.

- **hRAt-3 (il confound profondo — sotto-causa (ii)).** Forma alternativa: se
  $f_{\text{abs}}$ **non** risale con `pit` (~resta al floor), allora l'LLM
  rifiuta di predire l'ingresso in una tile pericolosa indipendentemente dal
  nome. Il world-model LLM per-query confonde predizione e saggezza — un limite
  strutturale del merge online che nessun vocabolario corregge.

- **hRAt-4 (movimento invariato).** La move-fidelity resta ~0.94 (il movimento
  non dipendeva dal nome della tile) — controllo di sanità.

---

## 4. Criteri di lettura

- **hRAt-1 ∧ hRAt-2** → Route A-ter **positivo**: il merge online *funziona* a
  vocabolario allineato; il confine offline/online di Route A si sposta da "il
  merge online fallisce" a "il merge online fallisce se i prior dell'LLM sulle
  tile sono disallineati dalle regole — e questo è ispezionabile e correggibile".
- **hRAt-3** → Route A-ter **negativo profondo**: il merge online ha un confound
  saggezza/predizione strutturale; serve un organo di percezione che traduca le
  tile in semantica operativa *prima* del world-model (la via (c) di
  E1_LLM_ROUTE_A_BIS §5).
- **hRAt-1 ∧ ¬hRAt-2** → entry-detection recuperata ma la sopravvivenza no: il
  collo di bottiglia è altrove (movimento, o $f_{\text{abs}}$ comunque sotto il
  gradino del sweep).
- **Nessun esito ri-apre la Congettura E** — chiusa. Route A-ter raffina il
  confine online del merge da "fallisce" a una condizione precisa e azionabile.

---

## 5. Cosa Route A-ter NON cambia

- Resta dominio gridworld-meccaniche, osservazione locale, online (per-query).
- Non tocca movimento, FMC, layout, modello, prompt-struttura — *solo* il nome
  della tile 1.
- Non è Route A-2 (dominio genuinamente aperto non-gridworld).
- Un solo nome alternativo (`pit`) è testato; se risultasse ambiguo, è esso
  stesso un dato a favore di (ii).

---

*Fine E1_LLM_ROUTE_A_TER_DESIGN.md. Design pre-registrato — nessun dato raccolto.
Un solo delta rispetto a Route A: la tile terminale di morte è chiamata `pit`
invece di `lava` — un nome il cui prior LLM ("ci si cade dentro") coincide con la
regola del mondo ("tile assorbente-terminale su cui si entra"). hRAt-1: il
$f_{\text{abs}}$ online risale sopra 0.80? hRAt-2: la self-preservation si
recupera? hRAt-3: o l'LLM rifiuta comunque di predire l'ingresso in un pericolo
(confound saggezza/predizione)? Esecuzione: `e1_llm_route_a_ter.py`. Verdetto in
`E1_LLM_ROUTE_A_TER_RESULT.md`.*
