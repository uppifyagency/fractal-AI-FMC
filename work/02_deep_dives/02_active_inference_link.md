# Deep Dive 02 — Fractal AI ↔ Active Inference: la fondazione di principio del merge FMC+LLM

> *"Il core FMC è l'agente; l'LLM è un organo."* — Congettura E. Questo deep dive
> mostra **perché** quella inversione dello stack è principiata e non arbitraria:
> è, esattamente, l'architettura dell'Active Inference. L'LLM è il **modello
> generativo**; FMC è il **motore di inferenza** che minimizza la Expected Free
> Energy.

> **Stato**: scritto, 2026-05-21 (era outline). Fondazione teorica del merge
> FMC+LLM (direzione D5 del programma research-partner). Teoria pura — nessun
> esperimento. Collegamenti: [deep dive 05](05_smc_particle_filter_view.md) (FMC
> come SMC/Feynman-Kac, prerequisito di §3), [deep dive 09](09_chaos_order_frontier_formalization.md)
> (la frontiera caos/ordine), MATH_CANON Congettura E, Teorema 2, Def. 3.

---

## 0. Tesi

> **FMC è un motore di inferenza Active Inference.** Massimizza, via Sequential
> Monte Carlo, un funzionale di orizzonte a due termini — uno *pragmatico*
> (goal-seeking) e uno *epistemico* (mantenere aperti i futuri) — che è l'analogo
> della Expected Free Energy (EFE) di Friston. I due termini sono i due esponenti
> $\alpha, \beta$ del virtual reward (Def. 3).

Da questa tesi discende la fondazione del **merge FMC+LLM** (Congettura E):

- L'Active Inference richiede *due* componenti: un **modello generativo**
  $p(o,s)$ del mondo, e un **motore di inferenza** che lo usa per minimizzare la
  free energy.
- FMC, da solo, è il motore — ma non possiede il modello generativo su domini
  aperti (su domini chiusi glielo dà il simulatore).
- L'LLM, da solo, è un modello generativo del mondo — ma non è un motore di
  inferenza che pianifica minimizzando free energy.
- **Il merge = Active Inference con un modello generativo LLM e un motore di
  inferenza SMC (FMC).** L'inversione dello stack di Congettura E ("FMC agente,
  LLM organo") è la tesi di Friston che *il modello generativo è un organo
  dell'inferenza, non l'agente* — l'agente *è* il processo di inferenza.

Questo deep dive rende la tesi rigorosa, ne segna i limiti onesti, e ne estrae
predizioni falsificabili.

---

## 1. Il Free Energy Principle in due minuti

Friston (2010) postula che ogni sistema che persiste minimizza la propria
**free energy variazionale** — un limite superiore alla sorpresa
$-\log p(o)$ delle osservazioni $o$. Due regimi:

- **Percezione** — minimizzare la *variational free energy* $F[q]$ rispetto alle
  credenze $q(s)$ sugli stati nascosti $s$:
  $$
  F[q] = \underbrace{D_{\mathrm{KL}}\!\big[q(s)\,\|\,p(s\mid o)\big]}_{\geq 0}
         \;-\; \log p(o).
  $$
  Minimizzare $F$ rende $q(s)$ la migliore approssimazione del posterior
  $p(s\mid o)$.

- **Azione / pianificazione** — scegliere una policy $\pi$ che minimizzi la
  **Expected Free Energy** $G(\pi)$, la free energy *attesa nel futuro* sotto
  quella policy. Per un orizzonte di tick $\tau$:
  $$
  G(\pi) = \sum_\tau G(\pi,\tau), \qquad
  G(\pi,\tau) = \underbrace{-\,\mathbb{E}_q\!\big[\log p(o_\tau\mid C)\big]}_{\text{costo pragmatico}}
  \;-\; \underbrace{\mathbb{E}_q\!\big[D_{\mathrm{KL}}[\,q(s_\tau\mid o_\tau,\pi)\,\|\,q(s_\tau\mid\pi)\,]\big]}_{\text{valore epistemico}}.
  $$

L'agente **minimizza $G$** — equivalentemente, **massimizza** la somma di:

- **valore pragmatico** $\mathbb{E}_q[\log p(o\mid C)]$ — quanto gli esiti attesi
  somigliano agli esiti *preferiti* $C$ (il "goal");
- **valore epistemico** $\mathbb{E}_q[D_{\mathrm{KL}}[\cdots]]$ — quanta
  *informazione* l'agente si aspetta di guadagnare sugli stati nascosti
  (esplorazione, riduzione di incertezza).

Tutto il resto di questo documento è la traduzione di queste due righe in FMC.

---

## 2. La mappa formale FMC ↔ Active Inference

| Active Inference (Friston) | Fractal AI | Note |
|---|---|---|
| Modello generativo $p(o,s)$ | Kernel $\mathcal{M}$ + stato iniziale $x_0$ | §6: nel merge, $\mathcal{M}$ = LLM |
| Credenze posteriori $q(s)$ | Distribuzione dei walker sullo swarm $\mathbf{W}_t$ | $q$ rappresentata da *particelle*, non parametrica |
| Supporto di $q(s)$ | Cono causale $X_H(x_0,\tau)$ | gli stati che $q$ può pesare |
| Variational free energy $F[q]$ | (implicita — FMC non fa percezione esplicita) | vedi §7 |
| Expected free energy $G(\pi)$ | $-\sum_t \log \mathrm{VR}_t$ (funzionale di orizzonte) | §3 |
| Valore pragmatico | $\widehat{R}^\alpha$ (termine di reward) | §4 — identità stretta |
| Valore epistemico | $\widehat{D}^\beta$ (termine di distanza) | §4 — *famiglia*, non identità |
| Precisione (su policy / EFE) | esponenti $\alpha,\beta$ | §5 |
| Markov blanket | l'interfaccia `observe` (sensoriale) + $a^*$ (attiva) | **non** il cono causale |
| Inferenza variazionale (gradient descent) | resampling SMC dello swarm (cloning) | §3 — stesso obiettivo, solver diverso |
| Apprendimento del modello generativo | *assente in FMC* | §7 — il merge lo ripara via LLM |

> **Correzione rispetto all'outline precedente.** Il Markov blanket di Friston è
> il *confine statistico* dell'agente (stati sensoriali + attivi che separano
> interno da esterno), **non** il cono causale. Il cono $X_H$ è il *supporto della
> credenza* $q$. Lo swarm $\mathbf{W}$ sono gli stati *interni* (le credenze); il
> simulatore è l'*esterno*; il blanket è `observe()` (sensoriale) e $a^*$ (attivo).

---

## 3. Il ponte rigoroso: VR è un potenziale di Feynman-Kac

Il punto tecnico delicato. Non si afferma "$\mathrm{VR} = -G$" termine a termine —
le due quantità non hanno la stessa forma algebrica. EFE è una **somma** di due
termini; VR (Def. 3) è un **prodotto** $\widehat{R}^\alpha\cdot\widehat{D}^\beta$.
Il ponte passa per il logaritmo e per la struttura SMC.

Il deep dive [05](05_smc_particle_filter_view.md) stabilisce che FMC è un sistema
di particelle di **Feynman-Kac**: la $\mathrm{VR}_t$ è il *potenziale* $G_t$ del
tick $t$, e il peso di un cammino di orizzonte $M$ è $\prod_{t=1}^{M}\mathrm{VR}_t$.
Prendendo il logaritmo, il funzionale che FMC ottimizza lungo l'orizzonte è
**additivo**:

$$
\mathcal{J}(\text{cammino}) \;=\; \sum_{t=1}^{M}\log\mathrm{VR}_t
\;=\; \alpha\sum_{t=1}^{M}\log\widehat{R}_t \;+\; \beta\sum_{t=1}^{M}\log\widehat{D}_t .
$$

> **Proposizione 02.1 (struttura EFE di FMC).** Il funzionale di orizzonte di FMC
> $\mathcal{J}$ è una somma pesata di un termine **pragmatico** ($\sum\log\widehat{R}$,
> peso $\alpha$) e di un termine **epistemico/esploratorio** ($\sum\log\widehat{D}$,
> peso $\beta$). È la struttura a due termini della Expected Free Energy:
> massimizzare $\mathcal{J}$ è l'analogo SMC di minimizzare $G(\pi)$. ∎ (sketch)

La forma del ponte:

- EFE è additiva in linear-space; FMC è additiva in **log-space** dopo il prodotto
  di potenziali Feynman-Kac. Il logaritmo è esattamente la trasformazione che
  manda il resampling moltiplicativo SMC nell'obiettivo additivo di Friston.
- La trasformazione `relativize` (Def. 2) è ciò che rende $R$ e $d$ — di segno e
  scala arbitrari — quantità positive e confrontabili di cui ha senso il prodotto.
  È il prezzo che FMC paga per non avere un modello probabilistico esplicito: al
  posto delle densità di Friston, normalizza per z-score.

La mappa quindi **non** è "VR è meno l'EFE": è "FMC ottimizza un funzionale di
orizzonte con la stessa struttura a due termini pesati dell'EFE, via SMC invece
che via discesa variazionale". È un'analogia *strutturale rigorosa*, ed è tutto
ciò che serve per la fondazione del merge.

---

## 4. I due termini, con onestà

### 4.1 $\widehat{R}^\alpha$ = valore pragmatico — identità stretta

Il termine di reward è il valore pragmatico senza riserve. $R$ codifica la
preferenza (l'esito desiderato $C$); $\widehat{R}^\alpha$ è la pressione verso gli
stati ad alta reward. Il Teorema 2 di MATH_CANON lo conferma da un'altra
direzione: la distribuzione invariante dello swarm è $\pi^*\propto R^\alpha$, una
Gibbs/Boltzmann con $\alpha$ = temperatura inversa. In AIF, massimizzare il valore
pragmatico *è* campionare proporzionalmente a $\exp(\log p(o\mid C))$. Stessa cosa.

### 4.2 $\widehat{D}^\beta$ = il termine esploratorio — *famiglia*, non identità

Qui serve precisione, ed è il punto dove l'outline precedente era troppo
disinvolto ("epistemic ≡ $d^\beta$"). Il valore epistemico di AIF è un
**guadagno di informazione** — una KL tra posterior con e senza l'osservazione.
Il termine $\widehat{D}^\beta$ di FMC è la **distanza a coppie tra walker**: tiene
lo swarm *sparso*. Non sono la stessa quantità.

Ma appartengono alla stessa **famiglia**. Tre nozioni, da tre discipline, dello
stesso impulso "tieni aperti molti futuri":

1. **Valore epistemico** (Active Inference) — guadagno di informazione atteso.
2. **Empowerment** (Salge et al. 2013) — la capacità di canale $\max I(A_t;S_{t+n})$
   tra azioni e stati futuri: *quanti futuri distinguibili* l'agente può
   raggticamente raggiungere.
3. **Forza entropica causale** (Wissner-Gross & Freer 2013) — $F=T_c\nabla_X S_c$,
   il gradiente dell'entropia dei cammini causali futuri.

MATH_CANON registra già due ancore canoniche: l'$\alpha=0$ di FMC (VR $=\widehat{D}^\beta$
puro) è "l'equivalente formale del Common Sense" e *l'equivalente formale
dell'empowerment* (Salge 2013); e la Eq. 11 di Wissner-Gross è "il limite
continuo di FMC con $\alpha=0$".

> **Proposizione 02.2 (la $\beta$ è la testa esploratoria, in forma empowerment).**
> A $\alpha=0$, $\mathrm{VR}=\widehat{D}^\beta$: FMC massimizza la dispersione
> dello swarm = la diversità degli stati futuri che le particelle rappresentano.
> Questo è uno **stimatore a particelle** della diversità dei futuri raggiungibili
> — la forma *empowerment / forza-entropica-causale* dell'impulso esploratorio.
> **Non** è la KL-information-gain letterale di AIF: è un proxy di swarm-diversità
> della stessa famiglia. ∎ (sketch)

La lettura onesta: $\widehat{D}^\beta$ è il membro *empowerment-flavoured*,
particle-based, della famiglia esploratoria — più grezzo del valore epistemico
KL, ma gradient-free e calcolabile su un simulatore black-box (è la ragione per
cui FMC funziona dove AIF discreta non arriva — §6).

---

## 5. $\alpha$ e $\beta$ sono le precisioni di AIF, rese esplicite

In Active Inference la EFE ha i suoi due termini con peso nominalmente unitario;
la modulazione del tradeoff pragmatico/epistemico passa per i **parametri di
precisione** (la precisione $\gamma$ sulle policy, le precisioni sui termini).
FMC fa la stessa cosa, ma con due esponenti scalari espliciti:

- $\alpha$ — il Teorema 2 lo identifica con la **temperatura inversa**: è la
  precisione sulla componente pragmatica. $\alpha\!\to\!\infty$ = greedy (precisione
  infinita sul goal); $\alpha\!\to\!0$ = nessuna pressione di goal.
- $\beta$ — la precisione sulla componente esploratoria.

Il contributo del progetto, qui, non è "abbiamo scoperto le precisioni" — AIF le
ha. È che FMC le rende **due esponenti scalari su un peso di resampling**, e la
Congettura E2 ne ha **caratterizzato empiricamente la fenomenologia** (sweep
$6\alpha\times4\beta$, 4320 episodi):

- $\alpha$ è un trade-off reale — più precisione pragmatica → più goal *e* più
  morte (option-collapse);
- $\beta$ è sicurezza **quasi gratuita** — più precisione esploratoria dimezza la
  morte senza costare goal (H4 di E2 falsificata).

Tradotto in linguaggio AIF: **alzare la precisione epistemica è quasi-gratis in
termini di valore pragmatico realizzato.** È un'affermazione concreta e
falsificabile sulla geometria del tradeoff EFE che la letteratura AIF, che tiene
le precisioni perlopiù implicite, non formula in questa forma. E lega a
[deep dive 09](09_chaos_order_frontier_formalization.md): la banda $(\alpha^*,\beta^*)$
è la frontiera caos/ordine — il punto di bilanciamento pragmatico/epistemico.

---

## 6. Il merge come Active Inference completa

Ora la fondazione del merge. L'Active Inference ha bisogno di due cose; il merge
le fornisce con la divisione del lavoro giusta.

**(a) Il modello generativo = l'LLM.** AIF non può inferire senza un $p(o,s)$. FMC
su domini aperti non ce l'ha. Un LLM *è* un modello generativo del mondo, appreso.
Nel merge l'LLM fornisce il kernel $\mathcal{M}$ (l'organo "modello del mondo" di
Congettura E). Punto sottile e importante: l'apprendimento del modello generativo
— la parte di AIF che FMC non ha (§7) — nel merge è **già avvenuto**, nel
pre-training dell'LLM. Il merge eredita gratis la componente di learning di AIF.

**(b) Il motore di inferenza = FMC.** AIF standard (discrete-state) minimizza $G$
via discesa variazionale: richiede il modello in forma esplicita e, di norma,
differenziabile. Un LLM-world-model **non è differenziabile in modo utile** per il
planning — non si fa backprop attraverso un rollout di world-model LLM. FMC è un
solver **SMC, gradient-free**: gli serve solo poter *campionare* $\mathcal{M}(x,a)$,
non derivarlo. 

> **Questa è la ragione di principio per cui FMC è il motore giusto per un modello
> generativo LLM.** Non una scelta di comodo: un modello generativo
> non-differenziabile *richiede* un solver di inferenza gradient-free, e SMC è
> esattamente quello. L'inversione dello stack di Congettura E — FMC il motore,
> LLM l'organo-modello — è la fattorizzazione che l'Active Inference impone quando
> il modello generativo è un LLM.

**(c) La pulsione è intrinseca, non aggiunta.** In AIF l'agente non ha bisogno di
una "reward di sopravvivenza": evita gli stati sorprendenti perché *sono*
free-energy alta. La Congettura E1 (self-preservation emergente) è la lettura FMC
di questo: uno stato assorbente/di morte ha zero futuro → zero valore epistemico,
zero empowerment → il termine $\widehat{D}^\beta$ lo evita per costruzione. E1 non
è reward engineering; è la componente epistemica dell'EFE che fa il suo lavoro.
Il *converso locale del Teorema 3* (E1-robustness: una cella assorbente è un pozzo
di VR) è, in linguaggio AIF, "uno stato senza futuri è uno stato a valore
epistemico nullo, e l'inferenza lo de-pesa da sola".

---

## 7. Cosa FMC non eredita da AIF — e cosa il merge ripara

Onestà sui limiti della mappa.

| Componente AIF | FMC da solo | FMC+LLM (il merge) |
|---|---|---|
| Inferenza di stato (percezione, $F[q]$) | assente — FMC riceve $x_0$ già dato | l'organo *percezione* LLM la fornisce |
| Pianificazione (minimizzazione di $G$) | ✅ è il cuore di FMC | ✅ invariato |
| Apprendimento del modello generativo | **assente** — $\mathcal{M}$ è fisso | ✅ ottenuto nel pre-training dell'LLM |
| Aggiornamento online del modello | assente | parziale — solo se l'LLM è in-context-adattato |

Due limiti reali restano e vanno detti:

1. **FMC non fa percezione esplicita.** Non minimizza $F[q]$ (la riga vuota nella
   tabella di §2). Riceve lo stato $x_0$ come dato. Nel merge è l'organo percezione
   a colmare — ed è esattamente perché quell'organo non è "gratis" (richiede una
   metrica $d$ canonica) che il deep dive / il design P13 lo trattano come un
   rischio, non un dettaglio.
2. **L'aggiornamento online del modello** resta debole: l'LLM è appreso ma
   *statico* a inference-time (salvo adattamento in-context). Un agente AIF pieno
   aggiorna il proprio modello generativo con l'esperienza. Il merge, allo stato,
   è AIF con **percezione + planning + un modello generativo appreso-ma-congelato**
   — non con il loop di apprendimento continuo. È un merge *quasi* completo, e
   questa riga dice esattamente quanto manca.

---

## 8. Predizioni falsificabili

Una mappa teorica vale solo se rischia qualcosa. Tre predizioni che seguono dalla
tesi "FMC = motore EFE".

- **AIF-1.** A $\beta=0$ FMC è un agente AIF *puramente pragmatico* (nessun termine
  epistemico): dovrebbe collassare in convergenza prematura. — *Stato*:
  **confermata direzionalmente** — Teorema 3 (collasso a $\beta=0$) ed E2
  ($\beta=0\to79\%$ morte). Coerente.
- **AIF-2.** Il termine $\widehat{D}^\beta$, se è davvero della famiglia epistemica,
  dovrebbe spingere lo swarm verso stati *information-rich* — stati da cui si
  diramano molti futuri distinguibili. Test: misurare, lungo un rollout FMC, se le
  regioni preferite dal termine $\beta$ hanno empowerment empirico più alto della
  media. — *Stato*: **non testato**. Esperimento economico su `fmc-core`.
- **AIF-3.** Se $\alpha,\beta$ sono precisioni, la frontiera $\lambda_1\approx0$ di
  [deep dive 09](09_chaos_order_frontier_formalization.md) (banda di bilanciamento
  pragmatico/epistemico) dovrebbe coincidere con la banda di Pareto di E2. — *Stato*:
  **predizione condivisa con dd09 §5.4**; test di coerenza interna del canone.

Falsificazione della tesi di §0: se a $\beta=0$ FMC *non* collassasse (AIF-1
fallisce), o se il termine $\beta$ non correlasse con alcuna misura di
diversità-di-futuri (AIF-2 fallisce), la lettura "FMC = motore EFE" sarebbe
un'analogia estetica, non strutturale.

---

## 9. Riferimenti

### Active Inference / Free Energy Principle

- **Friston, K.** (2010). *The free-energy principle: a unified brain theory?*
  Nat. Rev. Neurosci. 11(2):127–138.
- **Friston, K., FitzGerald, T., Rigoli, F., et al.** (2017). *Active inference: a
  process theory*. Neural Computation 29(1):1–49. — decomposizione dell'EFE.
- **Da Costa, L., Parr, T., Sajid, N., et al.** (2020). *Active inference on
  discrete state-spaces: a synthesis*. arXiv:2001.07203.
- **Sajid, N., Ball, P. J., Parr, T., Friston, K. J.** (2021). *Active inference:
  demystified and compared*. Neural Computation 33(3):674–712.
- **Parr, T., Pezzulo, G., Friston, K.** (2022). *Active Inference: The Free Energy
  Principle in Mind, Brain, and Behavior*. MIT Press.

### Famiglia esploratoria (empowerment, entropia causale)

- **Salge, C., Glackin, C., Polani, D.** (2013). *Empowerment — an Introduction*.
  arXiv:1310.1863. — capacità di canale azione→futuro; equivalente formale del
  Common Sense $\alpha=0$.
- **Wissner-Gross, A. D., Freer, C. E.** (2013). *Causal Entropic Forces*. Phys.
  Rev. Lett. 110(16). — $F=T_c\nabla_X S_c$; limite continuo di FMC a $\alpha=0$.

### Repo

- [`docs/MATH_CANON.md`](../../docs/MATH_CANON.md) — Def. 2 (relativize), Def. 3
  (virtual reward), Teorema 2 (Gibbs, $\alpha$ = temperatura inversa), Congettura E.
- [`05_smc_particle_filter_view.md`](05_smc_particle_filter_view.md) — FMC come
  Feynman-Kac/SMC; prerequisito di §3 (VR = potenziale, $\prod\mathrm{VR}$ = peso
  di cammino).
- [`09_chaos_order_frontier_formalization.md`](09_chaos_order_frontier_formalization.md)
  — la frontiera $\lambda_1\approx0$ = banda di bilanciamento pragmatico/epistemico.
- [`work/12_conjecture_e/`](../12_conjecture_e/) — E1 (self-preservation emergente),
  E2 ($\alpha$/$\beta$ come precisioni, caratterizzazione del tradeoff),
  `P13_DESIGN.md` (l'organo modello-del-mondo LLM).

---

*Fine deep dive 02. Status: scritto (era outline). ~470 righe. Tesi: FMC è un
motore di inferenza Active Inference; il merge FMC+LLM è Active Inference con
modello generativo LLM e solver SMC — e quella fattorizzazione è imposta, non
scelta, dal fatto che un LLM-world-model è non-differenziabile.*
