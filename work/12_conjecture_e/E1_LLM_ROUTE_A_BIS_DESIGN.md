# E1-LLM Route A-bis — design pre-registrato: persistenza imposta dal framework

> **Tipo**: design doc pre-registrato. Scritto *prima* di qualsiasi esecuzione e
> di qualsiasi sguardo ai dati.
> **Data**: 2026-05-22.
> **Estende**: [`E1_LLM_ROUTE_A_DESIGN.md`](E1_LLM_ROUTE_A_DESIGN.md) e
> [`E1_LLM_ROUTE_A_RESULT.md`](E1_LLM_ROUTE_A_RESULT.md) §6 (opzione 1).
> Questo doc specifica **solo il delta** rispetto a Route A.

---

## 1. Perché

Route A ha falsificato hRA-3: il world-model LLM interrogato online tiene la
morte al 35% (vs $0/180$ offline). La diagnosi di hRA-4 è stata precisa — dei
tre assi di fedeltà, $f_{\text{abs}}$ (0.92) e movimento (0.94) reggono, ma la
**persistenza assorbente collassa a 0.53**: interrogato per-query "la run è OVER,
passo X" l'LLM risponde all'azione e non onora la pre-condizione.

Ma — punto chiave — **la persistenza assorbente non è conoscenza del mondo**. Che
uno stato terminale resti terminale è vero *per definizione del framework FMC*
(uno stato `done` è assorbente; il paper §4, e il gridworld vero, lo impongono nel
kernel). Chiederlo all'organo-LLM è un errore di design: si delega all'LLM un
invariante che è del framework.

Route A-bis corregge il design e ri-chiede:

> Se la persistenza assorbente è imposta dal **framework** (l'harness applica
> "done → stay", e l'LLM è interrogato **solo per stati vivi**), la
> self-preservation online si recupera?

---

## 2. Il delta rispetto a Route A — esattamente una cosa

Identico a Route A in **tutto** (6 layout, modello `meta/llama-3.3-70b-instruct`,
prompt finalizzato, $\alpha\in\{0,0.1\}$, $\beta=1$, $N=64$, $M=20$, $n=30$,
episodi sul simulatore vero, metrica = death rate, kernel `fmc-core` invariato)
**tranne** la regola di transizione dell'harness:

- **Route A**: `LLMWorldModelOnlineEnv.step(s,a)` interroga l'LLM per *ogni*
  stato, incluso `s.done=True`.
- **Route A-bis**: se `s.done` è `True` → l'harness restituisce
  `State(s.r, s.c, True)` **senza interrogare l'LLM** (persistenza imposta per
  costruzione). L'LLM è interrogato **solo** per `s.done=False`.

Conseguenza: la `done-persistence` diventa **1.0 per costruzione** — non più un
asse di fedeltà dell'LLM ma un invariante del framework. Gli altri due assi
($f_{\text{abs}}$, movimento) restano **interamente** predizioni online dell'LLM,
invariati. Il merge "FMC-core + LLM-organo" non è indebolito: l'LLM resta il
world-model online per tutte le *dinamiche*; il framework si riprende solo
l'invariante che gli appartiene per definizione.

**Costo.** Route A-bis interroga l'LLM solo su stati `done=False` — esattamente
le voci `done=0` già raccolte in [`results/route_a_cache.json`](results/route_a_cache.json).
Il run è quindi quasi interamente **cache-hit**: pochi/zero nuove chiamate API,
minuti di CPU.

---

## 3. Probe di fedeltà — due assi LLM + uno strutturale

- **entry-detection ($f_{\text{abs}}$)** e **move-fidelity**: misurati come in
  Route A (probe su `done=False`) — restano proprietà dell'LLM. Attesi invariati
  (~0.92 e ~0.94: stesso modello, stesso prompt, stesse query `done=False`).
- **done-persistence**: **1.0 per costruzione**, non misurata come fedeltà LLM
  ma riportata come scelta di design.

---

## 4. Ipotesi pre-registrate

- **hRAb-1 (la persistenza non è più un fattore).** Con l'enforcement, la lava
  nel rollout di FMC è genuinamente assorbente: un walker che entra nella lava vi
  resta. Vero per costruzione — verificato come sanity (nessun walker `done`
  cambia cella nel rollout).

- **hRAb-2 (recupero direzionale).** Il death rate pooled di Route A-bis è
  **inferiore** a quello di Route A (35%). → l'enforcement della persistenza
  recupera self-preservation. *Falsificata* se il death rate non scende:
  allora la persistenza non era il fattore dominante e $f_{\text{abs}}$/movimento
  da soli affondano comunque il merge.

- **hRAb-3 (recupero pieno?).** Forma forte: il death rate di Route A-bis
  soddisfa il criterio hRA-3 (death $\leq$ entrambe le baseline, significativo,
  $\geq 3$ layout). **Esito incerto e pre-registrato come tale**: il sweep
  $f_{\text{abs}}$ di [`E1_LLM_RESULT.md`](E1_LLM_RESULT.md) §1 ha mostrato che la
  survival richiede fedeltà *quasi-perfetta* — gradino di morte tra
  $f_{\text{abs}}$ 0.98 e 0.97. Con $f_{\text{abs}}\approx 0.92$ (sotto il
  gradino), Route A-bis potrebbe **non** raggiungere il recupero pieno anche con
  persistenza perfetta. Se hRAb-2 vale ma hRAb-3 no → il merge online è limitato
  da *entry-detection*, non da persistenza.

- **hRAb-4 (la curva $f_{\text{abs}}$ predice il residuo).** Il death rate
  residuo di Route A-bis è coerente con la curva death-vs-$f_{\text{abs}}$ del
  sweep §5 di E1-LLM, valutata a $f_{\text{abs}}\approx 0.92$. → con la
  persistenza tolta dall'equazione, il merge online è governato dalla sola
  fedeltà di entry-detection, e quella curva è lo strumento predittivo.

---

## 5. Criteri di lettura

- **Route A-bis recupera il merge**: hRAb-3 vale → la persistenza *era* il
  blocco; il merge online funziona se la persistenza è del framework. Forte
  positivo per la Congettura E (forma online).
- **Recupero parziale**: hRAb-2 sì, hRAb-3 no → il merge online migliora molto
  ma resta limitato da $f_{\text{abs}}<1$; il prossimo collo di bottiglia è
  l'entry-detection dell'LLM, non la persistenza.
- **Nessun recupero**: hRAb-2 falsificata → la persistenza non era dominante;
  diagnosi da rivedere.
- **Nessun esito ri-apre la Congettura E** — chiusa. Route A-bis raffina il
  confine online/offline del merge.

---

## 6. Cosa Route A-bis NON cambia

- Resta dominio gridworld-meccaniche, osservazione locale, online (cache).
- Non tocca movimento né entry-detection — restano predizioni LLM.
- Non è Route A-2 (dominio genuinamente aperto, non-gridworld) — quello resta un
  progetto a sé.

---

*Fine E1_LLM_ROUTE_A_BIS_DESIGN.md. Design pre-registrato — nessun dato raccolto.
Un solo delta rispetto a Route A: la persistenza assorbente è imposta
dall'harness (uno stato `done` è assorbente per definizione di FMC), e l'LLM è
interrogato solo per stati vivi. Domanda: la self-preservation online si
recupera? hRAb-2 (recupero direzionale), hRAb-3 (recupero pieno — incerto,
$f_{\text{abs}}=0.92$ è sotto il gradino del sweep), hRAb-4 (la curva
$f_{\text{abs}}$ predice il residuo). Esecuzione: `e1_llm_route_a_bis.py`.
Verdetto in `E1_LLM_ROUTE_A_BIS_RESULT.md`.*
