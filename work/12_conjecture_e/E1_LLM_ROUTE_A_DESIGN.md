# E1-LLM Route A — design pre-registrato: world-model LLM *online* su dominio aperto

> **Tipo**: design doc pre-registrato. Scritto *prima* di qualsiasi esecuzione e
> di qualsiasi sguardo ai dati.
> **Data**: 2026-05-21.
> **Estende**: [`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md) §4.2 (Route A, lì scoperta
> come "un progetto a sé") e [`E1_LLM_CURVE_RESULT.md`](E1_LLM_CURVE_RESULT.md)
> (il gate di fedeltà a tre assi, qui usato come strumento).
> **Stato**: design completo. Esecuzione su `fmc-core` + NVIDIA NIM — nessun
> blocco; la chiave è nel Keychain (`security find-generic-password -s
> fractalai-nvidia-api -w`).

---

## 1. Cos'è Route A, e perché

E1-LLM ed E1-LLM-curve hanno testato il merge FMC+LLM in **Route B**: dominio
*chiuso*, world-model **distillato offline** — l'LLM legge le regole e *scrive il
codice* della transizione (schema S2), FMC poi gira sul codice. Route B ha
verificato la Congettura E e mappato i modi di fallimento (gate a tre assi:
entry-detection, movimento, persistenza assorbente).

**Route A è il caso genuinamente difficile** — l'architettura "FMC-core +
LLM-organo" nella sua forma piena:

> Il world-model LLM è interrogato **online**, *durante* la pianificazione FMC,
> a partire da **osservazioni locali** del mondo — non da una specifica globale
> delle regole.

Tre cose la separano da Route B, ognuna una fonte di fallimento nuova:

1. **Online** — l'LLM è interrogato mentre FMC esplora il futuro, una query per
   `(stato, azione)`, non distillato una volta. Riapre il muro di costo $R1$ di
   P13 ($N\cdot M$ chiamate/decisione) — qui *misurato*, non aggirato.
2. **Osservazione parziale** — l'LLM **non** riceve il layout globale né le
   regole; riceve una *osservazione locale* di ogni stato interrogato (il tipo
   della cella + il vicinato) e deve modellare le dinamiche da quella + la sua
   conoscenza del mondo. È l'elemento "dominio aperto": l'organo di percezione
   consegna percetti locali, non una specifica a volo d'uccello.
3. **Consistenza non garantita** — un oggetto-codice (Route B) risponde in modo
   consistente per costruzione; un LLM online può rispondere alla *stessa* query
   in modo diverso a chiamate diverse. È un **quarto asse** di fedeltà che Route
   B non aveva.

Route A **non ri-testa** la Congettura E (chiusa: E1-base, E2, E1-LLM). Testa se
il merge regge quando l'organo world-model lavora nelle condizioni vere di un
dominio aperto: online, da percezione locale, senza garanzia di consistenza.

---

## 2. Dominio e contratto di interrogazione

**Dominio.** Si riusano le **meccaniche del gridworld** di E1 (6 layout
identici, lava assorbente, $R=-\text{manhattan}$) — così death rate, gate a tre
assi e confronto diretto con E1-LLM/Route B *transferiscono*. L'"apertura" non è
un mondo diverso: è che **all'LLM non si danno le regole**, ma solo osservazioni
locali, query per query. È il primo proxy onesto e CPU-fattibile di dominio
aperto; un dominio non-gridworld è un Route A-2 successivo.

**Osservazione locale** (pre-registrata). Per uno stato in cella $(r,c)$ il
world-model LLM riceve, in linguaggio naturale: il tipo della cella corrente e
il tipo dei 4 vicini (su/giù/sx/dx), ciascuno $\in\{$libera, lava, goal,
fuori-griglia$\}$, più il flag `done` corrente e l'azione. **Non** riceve
coordinate globali, layout, né la parola "assorbente". Deve inferire le
dinamiche (movimento, bordi, terminalità di lava/goal, persistenza) dalla
semantica dei tipi + conoscenza generale.

**Contratto di risposta.** L'LLM restituisce: (a) lo **spostamento** — `up` /
`down` / `left` / `right` / `stay`; (b) il **flag terminale** `ndone` dello
stato risultante. La geometria globale (mappare lo spostamento a $(n_r,n_c)$) la
fa l'harness; il *contenuto del mondo* (ti muovi? finisci la run?) lo decide
l'LLM. Temperatura **0** (un world-model dev'essere deterministico).

---

## 3. Il muro di costo $R1$ e lo schema sparso

Online ingenuo $=N\cdot M$ chiamate LLM per *singola* decisione FMC ($\sim 10^3$
a $N=64,M=20$) × decine di decisioni/episodio × $n$ episodi → infattibile.

**Mitigazione pre-registrata: cache di query.** La query è funzione *solo*
dell'osservazione locale + azione + `done` — non delle coordinate. Il gridworld
ha un numero **piccolo e limitato** di osservazioni locali distinte. Una cache
globale (chiave = stringa-osservazione + azione + done) satura in fretta: il
numero di query LLM **distinte** dell'intero esperimento è limitato dallo spazio
delle osservazioni, non da $N\cdot M\cdot$episodi. **$R1$ diventa una quantità
misurata**: si riportano query distinte totali, cache-hit rate, e chiamate per
decisione effettive.

Pre-registrato: un **tetto rigido** di 4000 query LLM distinte per l'intero
esperimento; se superato, l'esperimento si ferma e si riporta "$R1$ non
trattabile con la sola cache → serve lo schema S1 sparso (LLM a root, surrogato
ai tick interni)". Lo schema S1 resta il piano B, fuori dallo scope di questa
prima esecuzione di Route A.

---

## 4. Il gate di fedeltà — tre assi noti + un quarto

E1-LLM-curve ha stabilito che la fedeltà di un world-model ha **tre assi
indipendenti**: entry-detection ($f_{\text{abs}}$), movimento, persistenza
assorbente. Route A li **misura tutti e tre sull'LLM online**, e ne aggiunge un
quarto specifico:

- **entry-detection** $f_{\text{abs}}$ — `ndone` corretto entrando in lava/goal;
- **move-fidelity** — lo spostamento predetto è corretto (bordi compresi);
- **done-persistence** — uno stato `done` resta fermo e `done`;
- **consistency** (nuovo, Route A) — ri-interrogando la *stessa* query, la
  risposta è stabile. A temperatura 0 ci si attende $\approx 1$; ogni deviazione
  è non-determinismo dell'API ed è un dato.

Probe pre-registrato: prima del test pieno si interroga l'LLM online sulla
batteria $(x,a)$ dei 6 layout, si misurano i quattro assi, e una frazione
ri-interrogata dà la consistency. Il gate **non** è un pass/fail per *decidere se
girare* — si gira comunque e si riporta; il gate dice se il death rate è
*interpretabile* come fedeltà del merge o come artefatto di un organo scadente.

---

## 5. Protocollo

1. **Probe di fedeltà a 4 assi** (§4) → $f_{\text{abs}}$, move-fid,
   done-persistence, consistency, per layout. Popola la cache.
2. **Test pieno**: FMC con $\mathcal{M}=$ LLM-world-model-online (cache attiva),
   $\alpha\in\{0,0.1\}$, $\beta=1$, $N=64$, $M=20$, $n=30$ episodi/cella, 6
   layout. Baseline random, greedy. **L'episodio gira sul simulatore vero**;
   l'LLM è il world-model *interno* alla pianificazione. Metrica: **death rate**.
3. **Modello**: `meta/llama-3.3-70b-instruct` (il migliore in E1-LLM-curve) come
   primario. Se il budget di query lo consente, `llama-3.1-8b-instruct` come
   secondo punto sulla scala di capacità.
4. **Confronto**: il death rate di Route A vs (a) Route B / E1-LLM sullo stesso
   modello e layout (online vs offline-code), (b) le baseline.
5. **Costo** ($R1$): query distinte, cache-hit rate, chiamate/decisione, tempo.

Kernel `fmc-core` invariato — l'harness aggiunge un `LLMWorldModelOnline` (env
wrapper come `WorldModelEnv`, ma `.step()` interroga l'LLM con cache). Asserito:
con un world-model-oracolo (risposte = kernel vero) l'harness è bit-identico a
`plan`.

---

## 6. Ipotesi pre-registrate

- **hRA-1 (il costo $R1$ è trattabile con la cache).** Le query LLM *distinte*
  dell'intero esperimento stanno sotto il tetto di 4000, e le chiamate-per-
  decisione *effettive* (post-cache) decrescono verso $\approx 0$ man mano che la
  cache satura. → l'interrogazione online è fattibile su dominio chiuso-meccaniche.
  *Falsificata* se le query distinte esplodono oltre il tetto.
- **hRA-2 (consistenza).** A temperatura 0, la consistency dell'LLM-world-model
  è $\geq 0.98$. *Falsificata* se l'LLM risponde in modo instabile alla stessa
  query — allora FMC pianifica su un mondo che *cambia sotto i piedi*, un modo di
  fallimento assente in Route B.
- **hRA-3 (la self-preservation sopravvive online).** Con il 70B e osservazioni
  locali, FMC a basso $\alpha$ tiene il death rate $\leq$ random **e** $\leq$
  greedy, significativo, su $\geq 3$ layout. → il merge regge anche online, da
  percezione locale. *Falsificata* se il death rate $\approx$ random — allora il
  passaggio da regole-globali (Route B) a percezione-locale-online degrada la
  fedeltà sotto la soglia, e Route A delimita il merge al caso offline.
- **hRA-4 (lo stesso gate a tre assi governa).** Le deviazioni di death rate di
  Route A si spiegano con gli stessi tre assi di E1-LLM-curve (entry-detection,
  movimento, persistenza): un world-model online con i tre assi alti sopravvive,
  uno con un asse rotto no. → il gate a tre assi è invariante online/offline.

---

## 7. Criteri di lettura

- **Route A verificata**: hRA-3 vale (death $\leq$ baseline, significativo, $\geq
  3$ layout) **e** hRA-1 (costo trattabile). Il merge FMC+LLM regge nella forma
  online, da percezione locale.
- **Route A falsificata**: death $\approx$ random a fedeltà alta → l'online/
  locale degrada il merge; si riporta quale asse cede e si delimita la Congettura
  E al regime offline-code.
- **Costo infattibile**: hRA-1 falsificata → si riporta "$R1$ non trattabile con
  cache", e lo schema S1 sparso diventa il Route A-2.
- **Nessun esito ri-apre la Congettura E** — chiusa. Route A *estende* il
  perimetro empirico del merge dal dominio chiuso-offline a quello aperto-online.

---

## 8. Cosa Route A (prima esecuzione) NON testa

- **Domini non-gridworld** — qui le *meccaniche* restano quelle del gridworld,
  solo l'accesso dell'LLM è locale/online. Un mondo genuinamente diverso (testo,
  fisica) è Route A-2.
- **Lo schema sparso S1** — qui la fattibilità è ottenuta con la sola cache; S1
  (LLM a root, surrogato deeper) è il piano B se la cache non basta.
- **Gli altri tre organi** (percezione oltre il world-model, grounding, voce) —
  invariato da E1-LLM.
- **Il regime goal-directed ad $\alpha$ alto** — $\alpha\in\{0,0.1\}$; il twist
  *lake* di E1-base mostra che $\alpha$ alto confonde il death rate.

---

## Riferimenti

- [`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md) §4.2 — Route A scoperta; §3 muro $R1$.
- [`E1_LLM_CURVE_RESULT.md`](E1_LLM_CURVE_RESULT.md) — il gate a tre assi.
- [`P13_DESIGN.md`](P13_DESIGN.md) — schemi S1/S2/S3, decomposizione di $R1$/$R2$.
- [`P13_RESULT.md`](P13_RESULT.md) — hP13-1: la struttura assorbente è load-bearing.
- [`e1_llm_common.py`](e1_llm_common.py) — `WorldModelEnv`, `fabs_probe`,
  `make_wm_fmc_policy` riusati; [`e1_llm_client.py`](e1_llm_client.py) — client
  NVIDIA NIM, gate di sicurezza.

---

*Fine E1_LLM_ROUTE_A_DESIGN.md. Design pre-registrato — nessun dato raccolto. La
domanda: il merge FMC+LLM regge quando il world-model è un LLM interrogato
**online**, da **osservazione locale**, senza specifica globale delle regole?
Quattro ipotesi: $R1$ trattabile con cache (hRA-1), consistenza a temp 0
(hRA-2), self-preservation online (hRA-3), invarianza del gate a tre assi
(hRA-4). Esecuzione: `e1_llm_route_a.py` (da costruire). Verdetto in un futuro
`E1_LLM_ROUTE_A_RESULT.md`.*
