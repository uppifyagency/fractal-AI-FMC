# W1A — Mappatura critica del core matematico di FMC

> **Ruolo**: research associate + scettico/falsificatore.
> **Data**: 2026-07-09.
> **Fonti primarie lette per intero**: `docs/MATH_CANON.md` (855 righe), `work/02_deep_dives/01_cloning_mathematics.md`, `work/02_deep_dives/05_smc_particle_filter_view.md`, `work/02_deep_dives/04_relativize_axiomatics.md`.
> **Convenzione**: distinguo sempre **dimostrato** (D) / **asserito** (A) / **congetturato** (C). Cito `file:riga`.
> **Verdetto in una riga**: il *setup* e le *definizioni* sono in gran parte ben posti; i *tre teoremi centrali sono tutti allo stato di sketch o asserzione*, e due di essi (Th.1 e Th.2) hanno buchi non riconosciuti nel canone — uno dei quali è un errore aritmetico load-bearing.

---

## 1. Definizioni canoniche — contenuto e buona posizione

### Def. 1 — Walker swarm e iterazione (`MATH_CANON.md:72-110`)

**Contenuto essenziale.** Swarm = $N$ coppie (stato, etichetta) $\in (E\times A)^N$. Iterazione $\mathbf W_{t+1}=\mathcal S_t\circ\mathcal C_t(\mathbf W_t)$. L'etichetta $\ell^{(i)}$ è l'azione applicata **solo al tick 0**, poi persiste come marker fino al cloning. Decisione finale = moda del marginale di etichetta (`MATH_CANON.md:107`).

**Buona posizione: parziale. Due problemi.**

1. **Ambiguità di ordinamento (contraddizione interna).** La formula boxed `MATH_CANON.md:93` e `01_cloning_mathematics.md:75` scrivono $\mathbf W_{t+1}=\mathcal S_t\circ\mathcal C_t(\mathbf W_t)$ — **prima clone, poi step**. Ma lo pseudocodice SMC (`05_smc_particle_filter_view.md:485-513`) esegue PREDICT → WEIGHT → RESAMPLE, cioè **prima step, poi clone** ($\mathcal C\circ\mathcal S$). Le due convenzioni **non sono la stessa cosa ai tick di bordo**: con l'ordine boxed, al tick 0 dopo l'init tutti i walker sono in $x_0$ *identici* → il vettore reward è costante ($z=0$, $\widehat R=1$ ovunque) e tutte le distanze sono $0$ → la VR è degenere e il primo clone è un no-op mal definito. L'algoritmo reale (`run_swarm`) simula-prima; la formula boxed del canone è quindi o mal ordinata o richiede un caveat esplicito "per $t\ge 1$". **Non è un cavillo**: chi implementa dalla Def. 1 e non dal codice ottiene un tick di sfasamento.

2. **Tie-breaking non specificato** nell'$\arg\max$ di `MATH_CANON.md:107`. Con etichette a pari conteggio la decisione è indefinita. Minore, ma è una definizione, non un'euristica.

**Stato**: definizione operativamente chiara ma con una contraddizione di composizione (A vs codice) non risolta.

### Def. 2 — Relativize (`MATH_CANON.md:112-137`)

**Contenuto essenziale.** z-score $z^{(i)}=(r^{(i)}-\mu)/(\sigma+\varepsilon)$, poi mappa piecewise $\widehat R=\exp(z)$ per $z\le 0$, $1+\log(1+z)$ per $z>0$.

**Le sei proprietà elencate (`MATH_CANON.md:128-135`)**: positività (D), continuità in 0 (D, banale), differenziabilità $C^1$ in 0 (D, banale), compressione $O(\log z)$ a $+\infty$ (D), "decay sub-esponenziale" a $-\infty$ (**imprecisione**: $e^z$ per $z\to-\infty$ decade *esattamente esponenzialmente*, non "sub-esponenzialmente"; l'etichetta è sbagliata anche se l'intento — non azzerarsi mai esatto — è corretto), invarianza affine (D, dal z-score).

**Il fatto matematicamente più importante e sotto-sfruttato**: **l'invarianza affine** ($\widehat R(a\mathbf r+b)=\widehat R(\mathbf r)$ per $a>0$) implica che FMC è **scale-free nella reward** e che *conta solo la forma della distribuzione della reward sulla popolazione corrente di walker*. Conseguenza dura, verificabile e non enfatizzata: **qualunque reward shaping puramente affine è invisibile a FMC**. Questo è il fondamento matematico (non citato come tale) del perché la Congettura D richiede shaping *moltiplicativo/tier* e non additivo-globale.

**Assunzione implicita non dichiarata: relativize è mean-field.** $\mu,\sigma$ sono statistiche sull'*intera popolazione* di walker al tick $t$. Quindi il peso di un walker **dipende da tutti gli altri walker**. Questo ha conseguenze gravi sulla Def. 3 e sul Teorema 1 (vedi §3.1): la VR **non è un potenziale di stato** $G(x)$, è un funzionale della misura empirica.

**Il "buco aperto" della unicità (`MATH_CANON.md:137`, dd04 intero)**: il canone afferma che relativize è "l'unica funzione che soddisfa cinque assiomi ragionevoli". **Questo è asserito e, come formulato, falso.** `04_relativize_axiomatics.md` è esplicitamente un *outline* con "(da dimostrare)" (`04:48`). Peggio: gli assiomi A1–A5 (`04:13-17`) fissano al massimo una **classe di equivalenza asintotica** (comportamento a $\pm\infty$ + monotonia + positività), non una funzione puntuale. Innumerevoli $C^1$ soddisfano A1–A5 senza coincidere con la mappa del paper (es. qualunque interpolazione liscia con gli stessi asintoti). Inoltre il *matching $C^1$ in 0* (proprietà 3) è un vincolo **aggiuntivo** non presente in A1–A5. Quindi l'insieme di assiomi è **sotto-determinato**. Il teorema di `04:39-46` ("a meno di errore $O(\epsilon)$") è hand-wavy: mescola convergenza asintotica e uguaglianza puntuale.

**Stato**: definizione ben posta e utile; la sua *unicità/naturalezza* è **asserita, non dimostrata, e sovra-affermata** (P8, `MATH_CANON.md:709`).

### Def. 3 — Virtual reward (`MATH_CANON.md:139-158`)

**Contenuto.** $\mathrm{VR}^{(i)}=(\widehat R^{(i)})^\alpha(\widehat D^{(i)})^\beta$, con $\widehat D^{(i)}=\widehat R(d(W^{(i)},W^{(j(i))}))$ — relativize applicata al vettore delle **distanze a un singolo partner casuale**.

**Buona posizione: sì, con un caveat statistico.** Il termine distanza usa **un solo partner** $j(i)$, non la densità locale. Quindi $\widehat D^{(i)}$ è uno **stimatore a 1 campione** di $\mathbb E_j[d(x_i,x_j)]$. L'identificazione "$\mathrm{VR}\propto\rho(x)^{-\beta}$" (`05:187`, `MATH_CANON.md:283`) vale **solo in aspettazione / grande $N$**, non per la quantità realmente calcolata. Il canone tratta $\rho^{-\beta}$ come se fosse la definizione; in realtà è il limite di un estimatore rumoroso. Minore ma rilevante per l'onestà del Teorema 2 (dove $\rho$ entra "per auto-consistenza").

### Def. 4 — Cloning kernel (`MATH_CANON.md:160-188`) — **contiene un errore load-bearing**

**Contenuto.** Rate (non probabilità): $1$ se $\mathrm{VR}_i=0$; $0$ se $\mathrm{VR}_k\le\mathrm{VR}_i$; $(\mathrm{VR}_k-\mathrm{VR}_i)/\mathrm{VR}_i$ altrimenti. Probabilità effettiva $=\min(\rho,1)$. Clone copia stato **e** etichetta.

**Cosa è ben fatto (D).** La correzione terminologica rate-vs-probabilità (`MATH_CANON.md:164`) è rigorosa e onesta: la quantità può superare 1 e il clip è la corretta lettura probabilistica. Ottimo.

**L'errore (asserito e sbagliato).** `MATH_CANON.md:186` afferma che, una volta clippato, il rate dà *"la probabilità di accettazione standard $P_{\mathrm{MH}}=\min(\mathrm{VR}^{(k)}/\mathrm{VR}^{(i)},1)$"*. **Falso.** Il rate è $\rho=\mathrm{VR}_k/\mathrm{VR}_i-1$, quindi la probabilità effettiva è
$$P_{\text{FMC}}=\min\!\big(\mathrm{VR}_k/\mathrm{VR}_i-1,\,1\big),\quad\text{non}\quad \min(\mathrm{VR}_k/\mathrm{VR}_i,1).$$
Con $r:=\mathrm{VR}_k/\mathrm{VR}_i$: FMC dà $\min(r-1,1)$, MH dà $\min(r,1)=1$ per ogni $r\ge1$. Coincidono **solo per $r\ge 2$**. Nel regime $1<r<2$ FMC clona **strettamente meno** di MH (es. $r=1.5$: FMC $0.5$, MH $1.0$).

**Perché è load-bearing.** La regola FMC non è né Metropolis ($\min(r,1)$) né Barker ($r/(1+r)$): è una **rampa traslata** che *ignora i miglioramenti marginali* ($P\to 0$ per $r\to 1^+$) — una forma di isteresi/soglia implicita. Questo è un tratto caratteristico genuino di FMC, **mis-descritto** come MH. E — cruciale — la derivazione di Gibbs del Teorema 2 assume proprio l'accettazione MH. Con la regola traslata il conto di equilibrio cambia (§3.2).

### Def. 5 — ESS (`MATH_CANON.md:190-205`)

$\mathrm{ESS}=(\sum\mathrm{VR})^2/\sum\mathrm{VR}^2\in[1,N]$. Standard, ben posta (D). Coerente con `01:112`. Nota: FMC vanilla resampla a ogni tick e **non usa** l'ESS come gate; è quindi diagnostica, non parte del kernel. Corretto e dichiarato.

### Def. 6 — Effective branching (`MATH_CANON.md:207-223`)

$b_{\text{eff}}=\exp(H(\{p_a\}))\in[1,K]$, perplessità delle etichette *sopravvissute*. Ben posta (D), definizione operativa pulita, misurata sulle label a fine planning (`MATH_CANON.md:223`). È la definizione più solida e meno ambigua del documento; tutto il lavoro di falsificazione della Congettura A vi si appoggia correttamente.

---

## 2. Riepilogo stato definizioni

| Def | Ben posta? | Problema principale |
|---|---|---|
| 1 Swarm/iterazione | Parziale | Ordine $\mathcal S\circ\mathcal C$ vs codice ($\mathcal C\circ\mathcal S$); degenere a $t=0$; tie-break assente |
| 2 Relativize | Sì (mappa) | Unicità **asserita e sovra-affermata**; natura mean-field non dichiarata; "sub-esponenziale" impreciso |
| 3 Virtual reward | Sì | $\widehat D$ è stimatore a 1 campione, non densità |
| 4 Cloning kernel | Sì (regola) | **Errore**: la regola clippata NON è MH $\min(r,1)$; è $\min(r-1,1)$ |
| 5 ESS | Sì | — (diagnostica, non nel kernel vanilla) |
| 6 Branching | Sì | — (la più pulita) |

---

## 3. I tre teoremi — enunciato, ipotesi, stato, anello debole

### Teorema 1 — Convergenza in $L^p$ (`MATH_CANON.md:229-251`; dd05 §3.1)

**Enunciato.** $E$ compatto, $\mathcal M_t$ Feller, $G_t=\mathrm{VR}_t$ limitato e $>0$: $\|\hat\eta_t^N(\varphi)-\eta_t(\varphi)\|_{L^p}\le c_t\|\varphi\|_\infty/\sqrt N$.

**Stato della prova: ASSERITO per citazione** (Del Moral 2004, Th. 7.4.4). Lo "sketch" (`MATH_CANON.md:241-247`) sono 3 bullet: (1) identificazione FMC↔Feynman-Kac; (2) bound di varianza pairwise ≤ multinomial "per Jensen"; (3) propagazione additiva.

**Anello più debole — GRAVE e non riconosciuto.** L'identificazione Feynman-Kac (passo 1) richiede che il potenziale sia un **potenziale di stato fisso** $G_t:E\to\mathbb R_+$. Ma $G_t=\mathrm{VR}$:
- **dipende dalla misura empirica** (relativize usa $\mu,\sigma$ sulla popolazione — Def. 2 mean-field), e
- **è stocastico** (termine distanza a partner casuale — Def. 3).

La teoria standard di Del Moral **non copre** potenziali mean-field: il flusso limite $\eta_t$ **non è nemmeno ben definito** quando $G$ dipende da $\eta_t$ stessa (è un punto fisso auto-referenziale, tipo McKean-Vlasov). Il canone tratta $\eta_t$ come "la distribuzione Feynman-Kac asintotica" senza affrontare questo. Serve una teoria di **propagation of chaos** per sistemi a campo medio, con costanti diverse. Il tasso $O(1/\sqrt N)$ resta *plausibile* ma è **analogia, non teorema**.

Passo 2 ("pairwise ≤ multinomial per Jensen"): **nessuna disuguaglianza scritta**. Il Lemma di dd05 (`05:134-148`) dimostra solo che i *limiti* coincidono, non un ordinamento di varianza a $N$ finito. Il resampling a 1-partner è più vicino a un "Bernoulli/Bertoin resampling" e può avere varianza *superiore* al sistematico. Claim non supportato.

Passo 3 / caveat $c_t$ (`MATH_CANON.md:251`): onesto ($c_t$ può esplodere $O(t)$–$O(t^2)$), ma rende il bound vacuo su orizzonti lunghi (Montezuma).

**Verdetto Th.1**: A. Il punto di rottura è **il potenziale mean-field**, ignorato del tutto. P1 (`MATH_CANON.md:702`) resta non verificato empiricamente — coerente con "asserito".

### Teorema 2 — Detailed balance / Gibbs (`MATH_CANON.md:253-301`; dd01 §4)

**Enunciato.** La distribuzione invariante del marginale a singolo walker, ristretta al cono, è $\pi^*(x)\propto R(x)^\alpha\rho(x)^{-\beta}$.

**Stato della prova: SKETCH, e con più falle non riconosciute.** È il teorema **più citato** (fonda Th.5, l'interpretazione "$\alpha$ = temperatura inversa", e le Congetture A/B/E) ed è **il più problematico**.

**Anello debole 1 — la regola non è MH (eredita l'errore di Def. 4).** Tutta la derivazione di $\pi\propto R^\alpha$ presuppone accettazione $\min(\mathrm{VR}_k/\mathrm{VR}_i,1)$. La regola vera è $\min(\mathrm{VR}_k/\mathrm{VR}_i-1,1)$. Con la regola vera il conto di bilancio cambia.

**Anello debole 2 — il cloning-only NON ha stazionaria di Gibbs.** Verifica diretta del bilancio dettagliato con la regola vera e proposta simmetrica: per $w_y>w_x$ (regime $1<w_y/w_x<2$), $a(x\to y)=w_y/w_x-1>0$ ma $a(y\to x)=\max(0,w_x/w_y-1)=0$. Bilancio dettagliato $\pi(x)a(x\to y)=\pi(y)a(y\to x)=0$ forza $\pi(x)=0$ — assurdo. Il cloning è un **ratchet monotòno in salita** (flusso in discesa nullo): la sua unica stazionaria è la **massa puntuale sul massimo di VR**, non una Gibbs sparsa. dd01 (`01:141`) "risolve" scrivendo a destra un rate **negativo** $\frac{R(x)^\alpha-R(y)^\alpha}{R(y)^\alpha}<0$ — una **probabilità negativa**: il conto di dd01 è *matematicamente incoerente*. La versione MATH_CANON (`MATH_CANON.md:269-283`) è più prudente (dichiara $\Pr[y\to x]=0$) ma poi hand-wava ("il calcolo porta a") esattamente il passaggio dove sta tutto il lavoro.

**Anello debole 3 — la Gibbs, se esiste, viene da $\mathcal S$, che FMC non implementa reversibile.** Poiché il cloning non è reversibile, la $\pi^*\propto R^\alpha$ potrebbe emergere solo se il **perturbatore $\mathcal S$** fosse un mover reversibile Metropolis-corretto rispetto a $R^\alpha$. Il canone lo *assume* ("perturbatore reversibile", `MATH_CANON.md:257`, `01:120`) ma **FMC reale non lo fa**: $\mathcal S$ = applica azione random + step del simulatore (deterministico su Atari), **senza correzione di Metropolis e senza reversibilità** rispetto ad alcun target. L'ipotesi che regge il teorema **non è soddisfatta dall'algoritmo**.

**Anello debole 4 — contraddizione empirica.** Se $\pi^*\propto R^\alpha$ con $\alpha=1$ finito, la distribuzione dovrebbe essere **sparsa** (temperatura finita) e $b_{\text{eff}}$ dovrebbe stabilizzarsi $>1$. Invece i dati (`MATH_CANON.md:388-404`) mostrano $b_{\text{eff}}\to 1$ (palmera) **per ogni $\alpha>0$** a $M$ grande. Il canone chiama questo "conseguenza del Teorema 2 (Gibbs concentrata sui massimi)" — ma la concentrazione totale è il limite $\alpha\to\infty$, non $\alpha=1$. Quindi l'equilibrio osservato è **collasso al miglior walker (Teorema 3)**, non una Gibbs a temperatura $\alpha$. **L'interpretazione "$\alpha$ = temperatura inversa" non è confermata a livello stazionario** (solo, forse, transitorio).

**Verdetto Th.2**: A/sketch con **errore di identificazione (MH)** + **ipotesi di reversibilità non soddisfatta** + **conto dd01 con probabilità negativa** + **tensione empirica**. È il buco più serio del core perché è la base più citata. La mappa fisica-statistica (`MATH_CANON.md:293-300`) è un'**analogia**, spacciata per teorema.

### Teorema 3 — Lemma anti-collasso (`MATH_CANON.md:302-327`; dd01 §5)

**Enunciato.** Con $\beta=0$ e perturbazione debole: $\mathrm{Var}_t\le\mathrm{Var}_0\gamma^t$, $\gamma\in(0,1)$.

**Stato: SKETCH plausibile ma con stima errata di $\gamma$.**

**Anello debole.** La stima $\gamma\approx 1-\mathbb E[(\mathrm{VR}_{\max}-\mathrm{VR}_i)/\mathrm{VR}_i]$ (`MATH_CANON.md:312`) è **mal posta**: $(\mathrm{VR}_{\max}-\mathrm{VR}_i)/\mathrm{VR}_i$ è illimitato, quindi $1-\mathbb E[\cdot]$ può essere **negativo**, fuori da $(0,1)$. $\gamma$ è *nominato*, non derivato. La direzione qualitativa (collasso geometrico senza distanza) è ragionevole ed empiricamente supportata (`MATH_CANON.md:314`), ma il bound quantitativo non regge.

**Incoerenza cross-documento sui tempi di collasso.** Th.3 afferma collasso in $\Theta(\log N)$ tick (`MATH_CANON.md:312`); la mappatura Wright-Fisher della Congettura A afferma tempo di fissazione $O(N)$ (`MATH_CANON.md:424`). Sono **regimi diversi** (selezione forte $\alpha>0$ → $\log N$; drift neutrale $\alpha=0$ → $O(N)$) ma il canone **non riconcilia** esplicitamente le due scale; un lettore le trova in contraddizione.

**Caveat empirico onesto e importante (`MATH_CANON.md:316-324`).** $\beta>1$ *riduce* $b_{\text{eff}}$ invece di aumentarlo: oltre una soglia il termine distanza torna a essere un **selettore**, non un repulsore. Questo è un ottimo pezzo di autofalsificazione, ma **mina la dicotomia pulita "$\alpha$=exploit / $\beta$=explore"** su cui poggia pesantemente la Congettura E ("$\beta$ = self-preservation", `MATH_CANON.md:599,614`). La E2 trova $\beta$ monotòno protettivo fino a $2.0$; il rocket sweep trova $\beta=5\to b_{\text{eff}}=1.89$. Le due letture di $\beta$ sono **regime-dipendenti** e il canone non le concilia.

**Verdetto Th.3**: A/sketch; direzione corretta, costante $\gamma$ mal posta, tempi non riconciliati con WF.

---

## 4. La vera forza matematica del core (2-3 proprietà)

Distinguo ciò che è **realmente speciale e (quasi) dimostrato** da ciò che è marketing.

1. **Scale-freeness della reward via invarianza affine di relativize (Def. 2, D).** È l'unica proprietà del core **dimostrata banalmente e senza buchi**, e ha conseguenze operative dure: FMC non richiede tuning della scala assoluta della reward; auto-normalizza sulla popolazione a ogni tick; qualunque shaping affine-globale è invisibile. Distingue FMC da CEM/MCTS (che richiedono scaling) e *spiega* matematicamente perché lo shaping efficace (Cong. D) deve essere non-affine/tier. Questa è la headline honest.

2. **Resampling pairwise embarrassingly-parallel (Def. 4, D come struttura).** Ogni walker si confronta con **un** partner: $O(N)$ confronti locali, **nessuna normalizzazione globale dei pesi** prima di ricampionare. È un vantaggio architetturale reale su SMC sistematico/multinomiale (che richiede normalizzazione globale) e sull'espansione sequenziale di MCTS. L'equivalenza al multinomiale nel limite (dd05 Lemma) è uno sketch ragionevole. NB: la regola *specifica* (rampa traslata $\min(r-1,1)$, §Def.4) è essa stessa un tratto distintivo — ignora i miglioramenti marginali — ma è **mis-descritta** come MH.

3. **Decisione per marginalizzazione delle etichette (auxiliary state, dd05 §6.2).** Trasformare un filtro distribuzionale in un **decisore discreto** via bincount delle `initial_decision` sopravvissute, **senza risolvere un problema di controllo/Bellman esplicito**. È l'innovazione concettuale propria di FMC rispetto a SMC (che è filtro, non planner). Ben identificata in `05:354-356`.

(Bonus, ma teoricamente fragile) Il termine di diversità $d^\beta$ come esplorazione *intrinseca al peso* invece che affidata al mixing del kernel (SMC) o a UCB (MCTS): posizionamento originale, ma la sua teoria "anti-collasso" regge solo per $\beta\in(0,\text{soglia})$ (Th.3 caveat).

**In sintesi**: il pezzo di matematica *solido* è (1); la struttura *architetturale* solida è (2)+(3). La "termodinamica dello swarm" (Th.2) è la parte **più venduta e meno dimostrata**.

---

## 5. Gap matematici concreti, ordinati per gravità

| # | Gravità | Gap | Dove | Tipo |
|---|---|---|---|---|
| G1 | **GRAVE** | Teorema 2: la regola di clone non è MH (Def.4 err.), il cloning-only ha stazionaria a massa puntuale (non Gibbs), la Gibbs richiederebbe $\mathcal S$ reversibile che FMC non implementa; dd01 usa una probabilità negativa; empiria ($b_{\text{eff}}\to1\ \forall\alpha>0$) contraddice la Gibbs a temperatura finita | `MATH_CANON.md:186,257,269-283`; `01:141` | Errore + ipotesi non soddisfatta + overclaim |
| G2 | **GRAVE** | Teorema 1: il potenziale $G=\mathrm{VR}$ è **mean-field** (relativize su $\mu,\sigma$) e stocastico; la Feynman-Kac standard (Del Moral) non si applica e $\eta_t$ non è ben definita; il bound di varianza pairwise≤multinomial è asserito senza prova | `MATH_CANON.md:233-247`; `05:197-207` | Identificazione non valida / asserzione |
| G3 | MEDIO | Unicità di relativize (P8): A1–A5 fissano una classe asintotica, non una funzione; il matching $C^1$ è extra-assiomatico; dd04 è "outline (da dimostrare)" | `MATH_CANON.md:137,709`; `04` intero | Asserito, falso come "unicità puntuale" |
| G4 | MEDIO | Teorema 3: $\gamma$ mal posto (può uscire da $(0,1)$); tempi $\log N$ vs WF $O(N)$ non riconciliati; caveat $\beta>$soglia mina la dicotomia $\alpha/\beta$ usata da Cong. E | `MATH_CANON.md:312,316-324,424` | Stima errata + incoerenza |
| G5 | MINORE | Def.1: ordine $\mathcal S\circ\mathcal C$ vs codice; degenere a $t=0$; tie-break assente | `MATH_CANON.md:93,107` | Ambiguità/contraddizione |
| G6 | MINORE | Def.3: $\widehat D$ è stimatore a 1 campione, non $\rho^{-\beta}$; identità solo in aspettazione | `MATH_CANON.md:146,283`; `05:187` | Imprecisione |
| G7 | MINORE | Th.5 (dd01 §6) eredita Th.2+Th.3: "teorema mancante" ancora mancante | `01:176-191` | Asserito |

**Nota trasversale**: il canone è *onesto* nel marcare molte cose come sketch/aperte (Appendice B `MATH_CANON.md:842-850` elenca "dimostrazione completa Th.1", "Th.3 con bound su $\gamma$", "unicità relativize" come backlog). Ma **G1 e l'errore MH di Def.4 non sono segnalati** — sono presentati come stabiliti. Questo è il difetto di rigore più serio: non l'incompletezza (dichiarata) ma l'**overclaim non dichiarato** su Th.2.

---

## 6. I 2-3 risultati da consolidare/dimostrare per un paper

1. **Stazionaria corretta del kernel FMC reale (sostituisce Th.2).** Due strade oneste: (a) dimostrare che il cloning-only converge a **massa puntuale (fissazione)** e caratterizzare la distribuzione *transitoria* (è ciò che i dati mostrano); oppure (b) derivare la stazionaria vera di $\mathcal S\circ\mathcal C$ con l'accettazione FMC (rampa traslata $\min(r-1,1)$) e $\mathcal S$ non-reversibile. **Ancoraggio rigoroso già disponibile**: la mappatura **Wright-Fisher/Moran** nel caso neutrale $\alpha=0$ è *empiricamente confermata* ($q=-0.948$ vs $-1$ teorico, `MATH_CANON.md:810`) e dà un teorema **dimostrabile** (dinamica neutrale FMC = processo di Moran, tempo di fissazione $O(N)$). Questo è il singolo risultato di più alto valore: chiude G1 con onestà e produce matematica vera.

2. **Bound di convergenza mean-field per FMC (ripara Th.1/P1).** Enunciare e dimostrare un risultato di **propagation of chaos** che gestisca esplicitamente il potenziale a campo medio (relativize) e il resampling pairwise — anche solo per il caso neutrale $\alpha=\beta=0$, o un CLT alla McKean-Vlasov. È il modo per rendere rigoroso l'embedding SMC (dd05), che è il miglior argomento di *legittimazione accademica* del progetto, senza il buco del potenziale di stato.

3. **Caratterizzazione di relativize come *classe di equivalenza asintotica* (ripara P8/G3).** Riformulare il teorema di dd04 nella forma *onesta e dimostrabile*: "sotto A1–A5, ogni $\widehat R$ è unica **a meno di equivalenza asintotica**" (dimostrabile), abbandonando la pretesa di unicità puntuale (falsa). Aggiungere l'invarianza affine come **proprietà headline dimostrata** e derivarne il corollario operativo (invisibilità dello shaping affine → necessità dello shaping tier di Cong. D). Modesto, self-contained (~1-2 settimane come da stima `MATH_CANON.md:726`), e chiude un buco nominato.

**Priorità**: #1 (risolve il buco più citato e ha già l'aggancio WF confermato) > #2 (legittima l'intero framing SMC) > #3 (basso costo, chiude un nome). Prima di tutto, però, va corretto l'**errore aritmetico di Def. 4:186** (la regola clippata non è MH) — è una riga, ma propaga in Th.2 e in tutta la lettura "termodinamica".

---

*Fine W1A. Le forze del core sono reali ma concentrate nella scala-freeness (Def.2) e nell'architettura (Def.4/decisione-per-etichetta); la "termodinamica" (Th.2) è la parte più fragile e sovra-affermata del documento.*
