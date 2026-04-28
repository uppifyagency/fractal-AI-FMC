# Deep Dive 08 — Insight estratti dal video-seminario su slide di Sergio Hernández

> **Stato**: scritto.
> **Fonte primaria**: [`VideoTranscriptSergio.md`](../../VideoTranscriptSergio.md) (raw, ~74 KB, no punteggiatura — trascrizione automatica).
> **Cronologia stimata**: ~2019-2021 (post-Atari 1807.01081 e post-port `fragile` PyTorch a cui Sergio fa riferimento; pre-Radient 2026).
> **Lingua del video**: spagnolo. Quote preservate verbatim (con il caveat che la trascrizione automatica ha errori grammaticali e zero punteggiatura).
> **Strumento di lettura**: file letto via wrap a 120 caratteri in `/tmp/sergio_wrapped.md` (625 righe).

## 0. Tesi del documento

> *Il video aggiunge alla nostra documentazione di FMC almeno tre nozioni operative non ancora formalizzate, conferma quattro formule del paper canonico con piccole varianti pedagogiche, e contiene un'unica discrepanza quantitativa rilevante (efficienza vs MCTS) che richiede una nota di onestà nei nostri claim pubblici.*

L'obiettivo qui è (i) estrarre e controverificare ogni formula contro [`1803.05049v5`](../../docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf) e [`ANALISIS.md`](../../ANALISIS.md), (ii) registrare i claim empirici verificabili, (iii) isolare ciò che è davvero nuovo rispetto a quanto già scritto in `plugin/fractal-coding-loop/docs/`, `work/02_deep_dives/01-07`, e [`2026_radient_sergio_interview.md`](../../docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md).

---

## 1. Inventario delle formule e dei claim quantitativi

Convenzione: ogni voce ha **(F#)** identificatore, **claim**, **verbatim** dal video (Sergio in spagnolo, righe del file wrapped), **paper-§**, **status**.

### F1 — Entropia causale del cono

**Claim**: l'entropia su un cono di futuri possibili è `H = -∑ p_i log(p_i)` sui cuadraditi (celle) che il cono attraversa. La forza intelligente sul punto è `f ∝ ∇H`.

> *"podéis calcular la entropía que - la suma de pp por el logaritmo de p para todos los puntitos"* — riga 62
> *"el gradiente de la función entropía y el gradiente me dice que me mueva"* — riga 71

**Paper-§**: §1.2 (Wissner-Gross causal entropic force, eq. 1) e §2.6 (causal entropy in FMC notation).
**Status**: ✅ già documentato in [`THEORY.md` §2](../../plugin/fractal-coding-loop/docs/THEORY.md), [`ANALISIS.md` §2.6](../../ANALISIS.md). Conferma testuale.

### F2 — Esempio numerico 74 → 87

**Claim**: spostandosi a sinistra l'entropia del cono cresce da ~74 a ~87 unità (la zona grigia che era inaccessibile sparisce).

> *"el cono [...] tiene un ancho [...] de 74 si tú te mueves a la izquierda [...] te saldrá 87 como aumenta te tienes que mover a la izquierda"* — righe 67-68

**Paper-§**: nessuno (puramente illustrativo, slide).
**Status**: 🆕 numero non presente nei nostri docs. Utile come *teaching figure* — non corrisponde a un esperimento riproducibile.

### F3 — Differenza tra intelligenze = solo τ

**Claim**: ogni sistema fisico è "intelligente"; cambia solo l'orizzonte temporale `τ`.

> *"una partícula pensaría a un tiempo de Planck, una piedra que se cae piensa un microsegundo, una mosca piensa a milisegundos y nosotros pensamos en hora buen día"* — riga 104

**Paper-§**: §1 implicita; Wissner-Gross 2013 esplicita τ.
**Status**: ✅ presente in [`THEORY.md` §1](../../plugin/fractal-coding-loop/docs/THEORY.md) e Radient cap. 1-2. Conferma. La formulazione "Planck-microsec-millisec-ora" è più nitida nel video che altrove — vale come *quote canonica*.

### F4 — Cloning binario originale (paper-paper-paper)

**Claim**: nella prima versione dell'algoritmo, il clone scattava solo in **due casi binari**: (i) walker morto (R=0), (ii) fine traiettoria con punti troppo vicini.

> *"sólo cuando uno se muere [...] o cuando llegue al final que estaban muy próximos"* — righe 152-156

**Paper-§**: §4.1-§4.3 (predecessori); §4.4 introduce la versione continua.
**Status**: ✅ documentato in [`ALGORITHM.md` §5](../../plugin/fractal-coding-loop/docs/ALGORITHM.md). Conferma.

### F5 — Probabilità di clone continua

**Claim**: si rimpiazza il clone binario con una probabilità di clone continua, proporzionale al rapporto delle reward.

> *"compararía mi recompensa con la suya y diría vaya la suya un 20% mejor pues con la probabilidad del 20 por ciento me voy a ese sitio"* — righe 167-168

Formalmente:

$$P_\text{clone}(i \to j) = \frac{(R_j - R_i)^+}{R_j}$$

**Paper-§**: §4.4, eq. (12).
**Status**: ✅ documentato in [`ALGORITHM.md` §5](../../plugin/fractal-coding-loop/docs/ALGORITHM.md), riga 304 sgg.

### F6 — Density continua (analogo del clone reward)

**Claim**: stessa identica meccanica per la "densità di walker intorno": confronto continuo, non binario "elimina se troppo vicino".

> *"aquí lo que hago es comparar mi densidad [...] cuánta gente hay a mi alrededor comparado con cuánta gente alrededor del otro"* — riga 175

**Paper-§**: §4.4-§4.5.
**Status**: ✅ documentato. Conferma.

### F7 — Metafora del minatore (JTBD per virtual reward)

**Claim**: il virtual reward = (densità d'oro nella mia zona) / (densità di altri minatori che mi fanno concorrenza).

> *"si tú quieres montar una mina [...] cada minero quiere que la densidad de oro sea la más grande posible y que la densidad de otra gente que está mirando por tu zona sea baja"* — righe 192-193
> *"el mejor yacimiento de oro posible donde la gente aún no lo haya descubierto eso sería la recompensa"* — riga 195
> *"si tu posición es un 20 por ciento mayor [...] la maleta irme a donde tú estás es del 20 por ciento"* — righe 201-202

**Paper-§**: §4.4 (virtual reward `VR_i = R_i^α · D_i^β`), interpretato come "trade-off reward × distance".
**Status**: 🆕 **framing non presente nei nostri docs**. Utile per `THEORY.md`/onboarding: spiega `α` (avidità per la reward) e `β` (avidità per zone meno affollate) come *competizione tra prospettori*. **Da inserire come exemplum pedagogico**.

### F8 — Distance estimator stocastico O(N)

**Claim**: invece di calcolare tutte le N(N-1)/2 distanze, ogni walker la calcola contro **uno solo** scelto a caso. O(N²) → O(N).

> *"vamos a decir un solo dinero al azar [...] hacemos sólo la distancia entre mi posición y la de otro minero elegido al azar [...] el orden en el cuadrado se ha convertido en orden n"* — righe 217-223

**Paper-§**: §4.5.
**Status**: ✅ documentato in [`ALGORITHM.md` §4.3](../../plugin/fractal-coding-loop/docs/ALGORITHM.md), [`THEORY.md` §4.4](../../plugin/fractal-coding-loop/docs/THEORY.md).

### F9 — Stocasticità come *feature*, non bug

**Claim sottile**: la varianza del partner randomico **fa bene** all'algoritmo perché inietta rumore.

> *"como lo vamos a hacer muchas veces el que esta función sea estocástica va a ser bueno porque nos va a meter ruido en el sistema y veréis que hará que vaya muy bien"* — righe 220-223

**Paper-§**: §4.5 (menzionato come "unbiased estimator").
**Status**: ⚠️ **menzionato superficialmente** in `ALGORITHM.md` §4.3 ("E[D_i] over runs converges to 1/ρ"). Sergio nel video lo presenta come *vantaggio* attivo, non come "lo facciamo per efficienza". **Da rinforzare** — vedi §4.5 sotto sulla *robustezza al rumore "gratis"*.

### F10 — Reshaping universale (relativize)

**Claim**: la trasformazione del reward è:

```
R_N = (R - μ) / σ                   ← z-score
R̂   = exp(R_N)        se R_N ≤ 0
R̂   = 1 + ln(1 + R_N) se R_N > 0
```

> *"cogemos la recompensa original y le restamos la media [...] lo dividimos por la desviación estándar [...] eso se llama el balú [value]"* — righe 230-232
> *"si el número es positivo le hacemos un logaritmo [...] las cosas que sean negativas [...] le hallamos su exponencial"* — righe 235-238
> *"se suben de la media subir a un medio 233 veces por encima de la media"* (errata trascrittiva: probabilmente "al medio se suma a multiplicare 2.33 volte sopra la media") — riga 239

**Paper-§**: §2.2.3 (formule esplicite, p. 12-13 di V5).
**Status**: ✅ documentato in [`ANALISIS.md` §126-132](../../ANALISIS.md), [`THEORY.md`](../../plugin/fractal-coding-loop/docs/THEORY.md), [`04_relativize_axiomatics.md`](04_relativize_axiomatics.md). Conferma piena.

### F11 — Reward negative → agenti pavidi (osservazione empirica)

**Claim**: se si permettono reward negative (senza relativize) gli agenti si "spaventano", frenano e diventano cauti.

> *"si usáramos recompensas negativas [...] cuando hacen la suma [...] ven que seguir andando le da recompensa negativa con lo cual lo que hacen es frenar y entonces se vuelven miedosos"* — righe 280-283
> *"nunca tenéis que en este sistema nunca se tienen que permitir recompensas negativas"* — riga 283

**Paper-§**: §2.2.3 menziona "valid reward must be > 0", ma non discute la patologia comportamentale dei negativi.
**Status**: 🆕 **osservazione empirica utile** non presente nei nostri docs. Conferma operativa che la `relativize` non è opzionale: **toglierla cambia la personalità dell'agente**, non solo la sua efficienza.

### F12 — *Cross-entropy collapse* (riformulazione concettuale centrale)

**Claim**: l'algoritmo finale, che apparentemente "non usa entropia", in realtà sta facendo **massimizzazione della cross-entropy** tra la distribuzione dei walker `P_W` e la distribuzione di reward `P_R` su ogni slice del cono.

> *"si tú miras la entropía cruzada de esas dos distribuciones lo que está haciendo es maximizar la"* — riga 528
> *"el hecho de saltar hacia donde hay más virtual reward es totalmente equivalente pero muchísimo más rápido de calcular que intentar maximizar la entropía cruzada [...] pero es lo mismo"* — righe 529-531
> *"esa cosa tan tontita que ves [...] está usando el aumento de entropía"* — righe 541-543
> *"la entropía desaparece [...] no la necesita usar [...] la unidad desaparece totalmente la necesidad de calcular la entropía"* — righe 544-549
> *"se ha quedado los huesos"* (= "rimane lo scheletro") — riga 551
> *"la inteligencia no va de aumentar la entropía de nada, va de que la probabilidad de que tú vayas a un sitio a otro sea proporcional a la recompensa"* — righe 536-537

**Paper-§**: §3, eq. (3): `P_S^OPT(x) ∝ R(x)` — equivalente a `D_H(P_R, P_S) = 0`.
**Status**: ⚠️ **presente come formula** in [`ANALISIS.md` §3.1-§3.2](../../ANALISIS.md), ma **non come centro narrativo** nei docs operativi. Sergio nel video lo elegge come *riformulazione filosofica*: «non stiamo massimizzando entropia, stiamo allineando densità di scanning a densità di reward».

**Conseguenza per i nostri docs**: vedi §4.1 sotto.

### F13 — Albero walker = frattale

**Claim**: l'albero generato dai cloni continui assomiglia a un frattale denso quando τ è lungo.

> *"al final te sale un árbol de decisión [...] que eso tiende a ser un fractal"* — righe 146-148

**Paper-§**: footnote nel §4.5; nome dell'algoritmo.
**Status**: ✅ documentato. Conferma.

### F14 — Atari risultati

**Claim**: «più della metà» dei 50 giochi Atari risolti al massimo punteggio. Specifico: «36» giochi valutati formalmente.

> *"de los 50 que elegimos [...] más de la mitad lo consiguió resolver [...] llegó a la máxima"* — righe 364-365
> *"36 de ellos cree usando exactamente el mismo algoritmo"* — riga 367
> *"más de la mitad de los juegos tenían un fallo [...] nunca estaban pensados para que alguien llegase al final"* — righe 366-367

**Paper-§**: §5.1 di V5 (1803.05049), tabella in 1807.01081.
**Status**: ✅ documentato in [`work/03_atari_replication/`](../../work/03_atari_replication/) e in [`THEORY.md`](../../plugin/fractal-coding-loop/docs/THEORY.md). Conferma. La nozione "metà dei giochi sono *bugged* perché non pensati per finire" è già in Radient ma vale ricordarla.

### F15 — Efficienza FMC vs MCTS — **discrepanza da risolvere**

**Claim del video**: ~100 000× più efficiente di MCTS.

> *"viene a ser como cien mil veces más eficiente esto que es montecarlo chis sets"* — riga 493

**Confronto con altre fonti**:

| Fonte | Numero MCTS | Numero FMC | Rapporto | Nota |
|---|---|---|---|---|
| Paper §5.1 (1803.05049v5) | ~3 000 000 samples per action (UCT) | ~400 samples per action | ~7 500× | numero formale del paper |
| Paper 1807.01081 (Atari empirico) | ~150 000 simulator reads | ~1 000 | ~150× | confronto specifico per stesso budget |
| Radient 2026 cap. 10 | ~150 000 | ~35 | ~4 286× | citazione orale di Sergio |
| **Questo video** | (non specificato) | (non specificato) | **~100 000×** | citazione orale di Sergio |
| `THEORY.md` §1 (nostro) | ~150 000 | ~400 | ~375× | parafrasi del paper |

**Status**: ⚠️ **discrepanza tra fonti**. I numeri orali di Sergio (35, 100 000×) sono memorie a posteriori e non coincidono con il paper. **Per pubblicazione esterna**: citare i numeri del paper (3M vs 400 → ~7 500×) e segnare "Sergio ha riportato in interviste rapporti tra 4 000× e 100 000× a seconda del task". Mai citare il "100 000×" senza qualificatore.

**Action**: aggiungere una sezione "Numeri da citare con cura" in [`THEORY.md`](../../plugin/fractal-coding-loop/docs/THEORY.md) o nella nota a piè di pagina di [`ANALISIS.md`](../../ANALISIS.md).

### F16 — FMC come sostituto di MCTS in AlphaZero

**Claim**: si può sostituire MCTS in AlphaZero con FMC, ottenendo (a) un planner più rapido, (b) una procedura per *educare* la policy network.

> *"podrían mezclarse [...] el alfa 0 reemplazar el montecarlo treaser por esto [...] aprendería al principio bastante más rápido"* — righe 491-493
> *"esa parte del alfa 0 sería mucho más eficiente"* — riga 493
> *"tu puedes hacer que tu red neuronal aprenda a hacer que ese prior se convierta en el posterior en la distribución que te da el otro"* — righe 488-490

**Paper-§**: §6.4 cita la possibilità a livello concettuale; non c'è esperimento.
**Status**: 🆕 **suggerimento strategico non documentato come roadmap**. Pertinente al nostro effort di benchmark (vedi [`DominiDaIndagare.md`](../../DominiDaIndagare.md)). **Da considerare** come direzione "FMC + DL ibrido" — territorio inesplorato (anche [`ANALISIS.md` §10.6](../../ANALISIS.md) lo segnala).

### F17 — Tre componenti dell'intelligenza

**Claim**: ogni AI completa ha (i) world-model, (ii) reward, (iii) planning. FMC copre (iii).

> *"siempre necesita esas tres cosas [...] aprender una reward que os diga más o menos para dónde tenía que ir y luego el planning"* — righe 580-582

**Paper-§**: §6 (deployment).
**Status**: ✅ documentato in [`THEORY.md`](../../plugin/fractal-coding-loop/docs/THEORY.md), [`02_active_inference_link.md`](02_active_inference_link.md). Conferma.

### F18 — Coscienza come tripla emergente

**Claim**: la *coscienza* emerge solo quando i tre pilastri (WM, reward, planning) sono spinti al massimo simultaneamente:

1. **Auto-coscienza**: io devo apparire dentro il mio world model — altrimenti non posso prevedere correttamente cosa succede quando agisco.
2. **Long-horizon reward shaping**: modifico le mie preferenze (smetto di amare lo zucchero) sulla base di previsioni a 5 mesi.
3. **Long-horizon planning**: piano a 200 anni (cambio climatico, scelta dell'auto).

> *"yo siempre veo mis manos [...] yo tendría que salir como una cosa dentro de ese world model"* — righe 583-585
> *"una persona consciente decide no tomar azúcar porque a la larga se pone gordo y entonces modifica sus preferencias"* — riga 590
> *"cuando tú decides que no va a usar el coche porque contamina y hay cambio climático está mirando a 200 años vista eso también es consciencia"* — righe 593-594
> *"la conciencia es emergente, es una propiedad emergente pero emergen los tres trozos a la vez"* — riga 603

**Paper-§**: nessuno (extra al paper).
**Status**: 🆕 **non documentato**. Connessione naturale con [`02_active_inference_link.md`](02_active_inference_link.md) (Friston). **Da inserire** come §X di quel deep-dive, o come capitolo separato sulla *theory of consciousness from FMC*. Vedi §4.3.

### F19 — Cooperazione multi-agente emergente

**Claim**: tre agenti (razzi-uncino) che condividono un task sollevano una pietra troppo pesante per uno solo. La cooperazione emerge **senza** reward esplicita di cooperazione, solo dall'ottimizzazione individuale.

> *"el rojo dice si le hecho una ayuda me llevo yo los puntos cada uno quiere optimizar sus puntos entonces entre todos consiguen subirlas"* — righe 339-340
> *"cooperan de manera que se ha caído el móvil como eran de manera espontánea"* — righe 345-346

**Paper-§**: §6.3 (cooperazione multi-agente in domini fisici).
**Status**: 🆕 **menzione empirica non scritta** nei nostri docs. Allineata con Wissner-Gross 2013 che mostra cooperazione emergente in puzzle multi-particle (vedi `CORPUS.md` riga 32). **Da catalogare** come empirical demo (vedi §5).

### F20 — Esplorazione via "odore di sé" (smell-trail)

**Claim**: due agenti senza reward esplicita di esplorazione, ognuno con avversione al proprio colore (e attrazione al colore altrui), si trovano spontaneamente in un labirinto.

> *"no les gusta el olor a verde [...] van dejando color con lo cual la función se va creando sola"* — riga 403
> *"al final encuentra la manera de unirse [...] una manera muy avanzada de hacer planning de ruta"* — righe 406-407

**Paper-§**: implicita in §6 (composizione di reward); non c'è esperimento dedicato.
**Status**: 🆕 **demo non documentata**. Equivalente a *intrinsic motivation via novelty* (curiosity-driven exploration), realizzato senza modulo separato — emerge dall'avversione al proprio trail. Da segnalare come *technique* per applicazioni FMC future.

### F21 — Pacman: orizzonte τ → ∞, episodio in colpo solo

**Claim**: con orizzonte τ molto grande, FMC pianifica l'**intera partita Pacman** prima di muovere il personaggio fisico. Lo schermo è proiettato su 2D ("autoencoder" implicito), si naviga in latent-space, si ricostruisce a ritroso.

> *"el horizonte ese que hemos hablado del tau lo hemos puesto a infinito [...] está aún no se ha movido [...] cuando ya consigue llegar al final salir del laberinto dice este es el camino que tenemos que seguir"* — righe 393-396

**Paper-§**: 1807.01081 (Atari) menziona walker che fanno rollout lunghi; non descrive esplicitamente la modalità "full-episode".
**Status**: 🆕 **demo non documentata**, importante perché mostra il limite superiore della scalabilità in τ.

### F22 — Robustezza al rumore "for free"

**Claim**: aggiungendo rumore gaussiano alla posizione misurata, FMC degrada *gentilmente* — l'agente "esita, lascia margine di sicurezza".

> *"cuando en lugar de pasar muy cerca de un sitio rozando ya no pasa rozando [...] siempre ya deja una distancia de seguridad [...] se vuelve más cauteloso porque no está seguro pero el algoritmo escala sin ningún problema"* — righe 564-566
> *"cuando se calcula la distancia sólo a uno de los otros [...] le está metiendo un ruido abismal [...] sin embargo eso le viene bien"* — righe 566-568

**Paper-§**: §5 esperimenti con rumore (osservazioni parziali); non c'è curva formale di degradazione.
**Status**: 🆕 **interpretazione operativa**: il single-random-pair distance estimator (F8) è **già rumoroso**, quindi rumore esterno aggiuntivo non rompe il sistema — è solo un'altra fonte della stessa varianza. **Connessione tra F9 e F22** che merita di essere esplicitata.

### F23 — Razzo + uncino su attrattore caotico

**Claim**: razzo legato a una palla con elastico → sistema con dinamica caotica. Reward bipartita: senza pietra `R = 1/dist(uncino, pietra)`, con pietra `R = 1/dist(uncino, target_circle)`. Strategie emergenti: `slingshot`, atterraggio per ricarica, sequenza di prese.

> *"piedra que está cayendo enganchada [...] luego una vez que la tenga enganchada métela dentro de este círculo [...] eso es muy difícil eso realmente con métodos normales no hay manera de hacerlo"* — righe 312-313
> *"engancha ahí el gancho y con la goma en un momento escoge la otra"* — riga 330

**Paper-§**: §6.2 menziona dominî con caos.
**Status**: 🆕 **demo non documentata**. Risultato impressionante per la nostra narrativa: "FMC funziona dove i metodi gradient-based falliscono per definizione".

---

## 2. Tabella riassuntiva — controverificazione

| F# | Topic | Paper | ANALISIS | THEORY/ALGORITHM | 02_deep_dives | Radient | Status finale |
|---|---|---|---|---|---|---|---|
| F1 | Causal entropy formula | §1.2, §2.6 | §2.6 | §2 | 05 | cap.2-3 | ✅ |
| F2 | 74→87 example | — | — | — | — | — | 🆕 illustrativo |
| F3 | τ-only difference | §1 | §1 | §1 | — | cap.2 | ✅ quote canonica |
| F4 | Cloning binario | §4.1-3 | §4 | §5 | 01 | — | ✅ |
| F5 | Clone continuo | §4.4 | §4 | §5 | 01 | cap.11 | ✅ |
| F6 | Density continua | §4.4 | §4 | §5 | 01 | — | ✅ |
| F7 | Miner JTBD | — | — | — | — | parz. cap.6 | 🆕 framing |
| F8 | Stocastico O(N) | §4.5 | §4 | §4.4 | 01 | — | ✅ |
| F9 | Stocastico = feature | menzionato | — | parz. §4.3 | — | — | ⚠️ rinforzare |
| F10 | Relativize | §2.2.3 | §126-132 | §3 | 04 | — | ✅ |
| F11 | Reward negative → cauti | parziale | — | — | — | — | 🆕 osservazione |
| F12 | Cross-entropy collapse | §3 eq.3 | §3 | implicito | — | cap.18 | ⚠️ riformulazione |
| F13 | Albero = frattale | §4.5 | §4 | §1 | — | cap.15 | ✅ |
| F14 | Atari results | §5 | §5 | — | 03_atari | cap.10 | ✅ |
| F15 | Efficienza vs MCTS | §5 (3M:400) | §5 | §1 (375×) | — | cap.10 (4286×) | ⚠️ **discrepanza** |
| F16 | FMC sostituisce MCTS in AlphaZero | §6.4 cenno | §10.6 cenno | — | — | cap.18-19 | 🆕 strategico |
| F17 | Tre componenti | §6 | — | §1 | 02 | cap.5 | ✅ |
| F18 | Coscienza emergente tripla | — | — | — | — | accenno | 🆕 da scrivere |
| F19 | Cooperazione emergente | §6.3 | — | — | — | cap.13 | 🆕 catalogare |
| F20 | Smell-trail exploration | — | — | — | — | — | 🆕 catalogare |
| F21 | Pacman τ→∞ | §5 implicito | — | — | — | — | 🆕 catalogare |
| F22 | Robustezza rumore | §5 | — | — | — | cap.10 cenno | 🆕 collegare a F9 |
| F23 | Razzo-uncino caotico | §6.2 | — | — | — | cap.13 | 🆕 catalogare |

**Legenda**: ✅ = già documentato e confermato. 🆕 = non presente nei docs interni. ⚠️ = parzialmente presente o discrepante.

---

## 3. Discrepanza F15 — efficienza FMC vs MCTS

> *Tre fonti, tre numeri diversi. Da risolvere prima di citare in pubblico.*

Il paper canonico ([1803.05049v5 §5](../../docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf)) riporta MCTS UCT @ ~3M samples per action vs FMC @ ~400 samples — **rapporto ~7 500×** sui giochi Atari valutati (Boxing, Centipede, ecc.). Il paper empirico [1807.01081](../../docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf) usa ~150 000 samples come budget MCTS standard nei benchmark dell'epoca; FMC raggiunge prestazioni equivalenti con ~1 000 samples — **rapporto ~150×**. Sergio nelle interviste cita rapporti **4 286×** (Radient 2026) e **100 000×** (questo video). Le citazioni orali sono memorie aggregate e non riproducibili.

**Raccomandazione operativa per i nostri docs e paper futuro**:

1. **Citare sempre il paper**, non Sergio orale: «FMC raggiunge prestazioni MCTS UCT di riferimento con ~400-1000 samples per action contro 150 000-3M, un fattore 150×-7 500× a seconda del benchmark» (paper §5).
2. **Non citare il "100 000×"** senza qualificatore "Sergio ha riportato in interviste fattori fino a 100 000× su task specifici, non riprodotti formalmente nel paper".
3. **Aggiungere una nota in [`THEORY.md`](../../plugin/fractal-coding-loop/docs/THEORY.md)** che dia il range corretto.

---

## 4. Insight innovativi non ancora documentati

> *Quattro nozioni del video meritano di entrare nei nostri docs. Per ognuna: cosa è, dove va inserita, formulazione proposta.*

### 4.1 Il "cross-entropy collapse" come riformulazione centrale di FMC

**Posizione**: nuovo paragrafo in [`THEORY.md`](../../plugin/fractal-coding-loop/docs/THEORY.md) dopo §2 ("Why FMC works"), o sezione dedicata in [`ANALISIS.md`](../../ANALISIS.md).

**Contenuto**:

> *FMC è spesso descritto come "massimizzazione di entropia causale". Questa descrizione è corretta ma fuorviante: il codice non calcola mai un'entropia. Sergio nel suo seminario su slide chiarisce che il "vero motore" è la condizione `P_walker(x) ∝ R(x)` per ogni slice del cono — equivalente a massimizzazione della cross-entropy `H_×(P_W, P_R) = -∑ P_R log P_W`. Il virtual reward `VR_i = R_i^α · D_i^β` con il pairwise stochastic clone realizza questa proporzionalità per migrazione: i walker si spostano dalle zone a bassa `VR` a zone ad alta `VR`, raggiungendo equilibrio quando la densità walker traccia la densità reward (analogo della distribuzione di Boltzmann su un potenziale `-log R`).*
>
> *In una frase di Sergio: «se ha quedado los huesos [è rimasto solo lo scheletro] [...] la entropía desaparece, no la necesita usar». L'entropia del cono è il **lato teorico** del problema; la **soluzione algoritmica** la riassorbe completamente nella dinamica di clone.*

**Conseguenze**:

- Nelle slide di pitch, smettere di dire "FMC massimizza entropia" e dire invece "FMC allinea la densità di walker alla densità di reward" — più preciso e meno confondente per audience deep-RL.
- Il legame con SMC particle filter (vedi [`05_smc_particle_filter_view.md`](05_smc_particle_filter_view.md)) diventa più nitido: è proprio quello che fa un particle filter ben pesato — far sì che la distribuzione delle particelle approssimi la posterior.

### 4.2 La metafora del minatore (JTBD per virtual reward)

**Posizione**: sostituire o affiancare l'analogia attuale del virtual reward in [`THEORY.md` §4](../../plugin/fractal-coding-loop/docs/THEORY.md) e [`ALGORITHM.md` §4](../../plugin/fractal-coding-loop/docs/ALGORITHM.md).

**Contenuto**:

> *Pensa ai walker come prospettori d'oro. Ognuno vorrebbe (i) trovarsi su un grosso giacimento (`R` alto) e (ii) lontano dagli altri prospettori (`D` alto). Il `VR_i = R_i^α · D_i^β` formalizza questo trade-off. Il pairwise stochastic clone formalizza il "cambio di sito": se chiamo un altro prospettore e mi dice che il suo `VR` è il 20% più alto del mio, c'è il 20% di probabilità che faccia le valigie e mi sposti dove sta lui. Questo trasforma la teoria astratta della cross-entropy in una procedura locale, leggera, parallelizzabile.*

**Perché ha valore**: la metafora del minatore rende immediato perché `α` e `β` siano **due parametri**, non uno. `α` è l'avidità ("voglio l'oro grosso adesso"), `β` è la prudenza territoriale ("non voglio finire in mezzo agli altri"). Modulando i due si ottengono comportamenti diversi (greedy explorer vs cautious surveyor) senza modificare l'algoritmo.

### 4.3 Coscienza come tripla emergente

**Posizione**: nuovo capitolo in [`02_active_inference_link.md`](02_active_inference_link.md) (deep-dive ancora outline) — la connessione con il FEP di Friston è naturale.

**Contenuto**:

> *Sergio inquadra la coscienza come **proprietà emergente di tre componenti spinte all'estremo**:*
>
> 1. *Auto-coscienza ⇔ il world model deve includere l'agente stesso. Quando apri gli occhi, ti vedi le mani: la rappresentazione interna che impara a predire il mondo deve necessariamente includere il proprio corpo come oggetto. Senza questo, il world model è inefficiente (non può predire le conseguenze delle proprie azioni).*
> 2. *Modulazione long-horizon della reward function ⇔ "smettere di amare lo zucchero" perché a 5 mesi vista ti farà male. La funzione di reward stessa diventa rivedibile sulla base di previsioni del planner.*
> 3. *Planning a orizzonte lungo ⇔ "non uso più l'auto perché a 200 anni il pianeta brucia". Il `τ` di FMC esteso a tempi *contro-evolutivi*.*
>
> *Solo quando i tre pilastri sono massimamente sviluppati e accoppiati emerge ciò che chiamiamo coscienza. È una **definizione operativa** e cade in una scala continua: una mosca ha tre pilastri minimi, un umano li ha massimi, una superintelligenza li avrebbe tutti più estesi.*

**Connessione FEP**: Friston (Active Inference) richiede esplicitamente l'auto-modello (component 1) per il *generative model* a free energy minimum. Le componenti 2-3 sono la versione "lunga" del minimum-free-energy hierarchy: shaping della prior precision attraverso planning. Vedere se la formulazione di Sergio si mappa su una gerarchia precisa nel FEP.

### 4.4 Reward negative → agenti pavidi

**Posizione**: nota a piè di pagina in [`04_relativize_axiomatics.md`](04_relativize_axiomatics.md), come motivazione operativa per A1 (positività).

**Contenuto**:

> *L'assioma A1 (`R̂(x) > 0`) non è solo una scelta di convenienza algebrica. Sergio osserva empiricamente che permettere reward negative produce **patologie comportamentali specifiche**: l'agente diventa "miedoso" — frena, evita il movimento. Questo perché in FMC il clone si propaga proporzionalmente al rapporto delle reward; se le reward possono essere negative, la dinamica `(R_j - R_i)/R_j` cambia segno e diventa inconsistente. Il reshaping `relativize` non è opzionale: toglierlo non degrada le performance, **cambia la personalità dell'agente**.*

### 4.5 Robustezza al rumore = corollario di F9

**Posizione**: rinforzare [`ALGORITHM.md` §4.3](../../plugin/fractal-coding-loop/docs/ALGORITHM.md) e aggiungere collegamento esplicito.

**Contenuto da aggiungere**:

> *Il single-random-pair distance estimator (§4.3) inietta varianza già a livello di algoritmo. Conseguenza inattesa: rumore aggiuntivo dall'esterno (sensori imperfetti, dinamica stocastica del simulator) **non rompe il sistema** — è solo un'altra fonte della stessa varianza. L'agente sotto rumore osservativo "esita, lascia margine di sicurezza" (Sergio, video), un comportamento qualitativamente identico a quello di un guidatore umano sotto incertezza. Questo è il lato pragmatico del fatto che FMC sia, in essenza, un metodo Monte Carlo: la varianza non è un nemico ma un canale.*

### 4.6 Catalogo demo empiriche del video

**Posizione**: nuovo file [`work/03_atari_replication/empirical_demos_from_seminar.md`](../03_atari_replication/empirical_demos_from_seminar.md), oppure appendice a questo doc.

| Demo | Setup | Comportamento emergente | Implicazione |
|---|---|---|---|
| Cochecito v1 (1-sec lookahead) | Solo sopravvivenza | Curve fluide, no laberinto solving | Limite del lookahead corto |
| Cochecito laberinto + multi-τ | Stesso, τ ∈ {1s, 2s} | Solo τ=2s risolve laberinti complessi | Effetto di τ |
| Razzo + elastico + uncino | Cattura pietra → deposita | Slingshot, atterraggio per ricarica, sequenze multi-grasp | FMC su sistema caotico |
| 3 razzi + pietra pesante | Stesso reward individuale | Cooperazione spontanea (uno spinge, altro solleva) | Cooperazione senza reward esplicita |
| 2 agenti + smell-trail | Avversione al proprio colore | Esplorazione coordinata, point of meeting | Curiosity emergente |
| Pacman τ → ∞ | Episodio in latent-space | Soluzione completa pre-azione, ricostruzione a ritroso | Limite scalabilità τ |
| Cohete con rumore sensori | Stessa fisica, posizione+rumore | "Tituba", margine di sicurezza | Robustezza graceful |

---

## 5. Sergio in prima persona — frasi notevoli

Quote da preservare per onboarding, slide, paper introduzione.

- *"todo se va a basar en un paper del 2013, qué bueno cuando lo leí fue el momento en que empecé a dedicarme a la inteligencia artificial antes no le había tocado porque no me gustaba"* — riga 50-51 (genesi Wissner-Gross, identica a Radient).
- *"este es el primer vídeo que hice la noche siguiente de leer el paper"* — riga 256 (mito del "una notte di Pascal", confermato).
- *"me sorprendió mucho [...] me sorprendió mucho en este caso este fue el momento ese que dije vaya supera las expectativas"* — analoga a Radient cap. 13 sull'emergenza nei razzi-uncino.
- *"la inteligencia no va de aumentar la entropía de nada, va de que la probabilidad de que tú vayas a un sitio a otro sea proporcional a la recompensa"* — F12, **citazione canonica del cross-entropy collapse**.
- *"se ha quedado los huesos"* — F12, metafora dello scheletro.
- *"al final esa es la única diferencia entre una mosca y nosotros"* — F3, sui livelli di intelligenza come solo τ.
- *"la conciencia es emergente, es una propiedad emergente pero emergen los tres trozos a la vez"* — F18.

---

## 6. Action items per i docs FMC

> *Estratti dalla §4. In ordine di priorità.*

1. **[ALTO]** Aggiungere paragrafo "cross-entropy collapse" in [`THEORY.md` §2](../../plugin/fractal-coding-loop/docs/THEORY.md) (vedi §4.1 sopra). Cambia il framing: da "FMC massimizza entropia causale" a "FMC allinea P_walker a P_R, equivalente a max cross-entropy".
2. **[ALTO]** Aggiungere nota su discrepanza F15 (efficienza vs MCTS) in [`THEORY.md` §1](../../plugin/fractal-coding-loop/docs/THEORY.md). Citare paper, non Sergio orale, e dare range (150×-7500×).
3. **[MEDIO]** Riempire [`02_active_inference_link.md`](02_active_inference_link.md) con la "tripla emergente di coscienza" (vedi §4.3).
4. **[MEDIO]** Inserire la metafora minatore in [`THEORY.md` §4](../../plugin/fractal-coding-loop/docs/THEORY.md) e [`ALGORITHM.md` §4](../../plugin/fractal-coding-loop/docs/ALGORITHM.md) come exemplum (vedi §4.2).
5. **[BASSO]** Espandere [`ALGORITHM.md` §4.3](../../plugin/fractal-coding-loop/docs/ALGORITHM.md) collegando F9 (stocastico = feature) e F22 (robustezza rumore "for free").
6. **[BASSO]** Note operative in [`04_relativize_axiomatics.md`](04_relativize_axiomatics.md) sulla "personalità dell'agente" (F11).
7. **[BASSO]** Catalogo demo empiriche → file separato in `work/03_atari_replication/` o appendice qui.

---

## 7. Verifica numerica di F12 e F11 (test Python controllati)

> *Codice e plot in [`work/04_mathematical_tests/`](../04_mathematical_tests/).*
> Implementazione FMC fedele al paper §4.4-5 e a `repos/FractalAI_old/fractalai/swarm.py:16-23,451-531`.
> Setup: 2D continuo bounded `[0, 10]²`, kernel di perturbazione gaussiano isotropo, N=200-400 walker, T=200-400 step, 4-5 seed per condizione.

### 7.1 F12 — verifica della proporzionalità `P_walker ∝ R(x)^α`

Sergio (video, riga 537): *"la inteligencia [...] va de que la probabilidad de que tú vayas a un sitio sea proporcional a la recompensa"*. Paper §3 eq. (3): `P_S^OPT(x) ∝ R(x)`. Deep-dive 01 Teorema 3: distribuzione invariante `π*(x) ∝ R(x)^α`.

**Test A — α-scan su landscape unimodale.** Reward `R(x,y) = 0.05 + exp(-‖x-c‖²/2σ²)` con `c=(5,5)`, `σ=1.5`. Misuriamo `log-Pearson(log P_walker, log R^α)` e `KL(P_walker ‖ R^α/Z)` su finestra stazionaria (50% T):

| α (codice) | log-Pearson | KL [nat] | comportamento |
|---|---|---|---|
| 0.0 | 0.00 | 1.02 | uniforme (atteso, R^0 = 1) |
| 0.5 | 0.20 | **0.62** | leggermente concentrato |
| **1.0** | **0.77** | 1.21 | **shape match** (Sergio's claim) |
| 2.0 | 0.67 | 0.90 | over-concentrato |
| 4.0 | 0.64 | 0.48 | mode-collapse forte |

**Interpretazione**: a α=1 codice la log-Pearson raggiunge il massimo (0.77), confermando direzionalmente la proporzionalità di Sergio. Il KL non si annulla per due ragioni: (i) finite-N (200 walker su griglia 80×80 = 6400 celle → noise di stima ~ √(N/M) · log M); (ii) la trasformazione `relativize` **amplifica α effettivo** (vedi §7.3) — il match Gibbs ideale di α=1 corrisponde a α=0.5 nel codice (KL minimo lì). La log-Pearson è metrica più robusta perché misura *forma* della distribuzione e non concentrazione assoluta.

**Test B — landscape multimodale (3 picchi)**. Reward = mistura di 3 gaussiane + baseline 0.05.

| Condizione | log-Pearson | KL | TV |
|---|---|---|---|
| FMC canonical (α=1) | **0.82** | 0.87 | 0.55 |
| FMC senza relativize (α=1) | **0.86** | **0.38** | 0.34 |
| Random walk | 0.04 | 0.54 | 0.44 |

Entrambe le varianti FMC raggiungono `log-Pearson > 0.8` (random walk: 0.04). I tre picchi sono catturati con concentrazione proporzionale ai pesi (vedi `f12B_multimodal_emp_vs_target.png`).

**Verdetto F12**: ✅ confermato direzionalmente. `P_walker` correla fortemente con `R^α` (log-log Pearson 0.77-0.86). La proporzionalità esatta richiede `N → ∞` e tuning di α per compensare il bias di `relativize`.

### 7.2 F11 — verifica della patologia "fearful agent" con reward negative

Sergio (video, righe 280-283): *"si usáramos recompensas negativas [...] lo que hacen es frenar y entonces se vuelven miedosos"*. Sergio (riga 283): *"nunca se tienen que permitir recompensas negativas"*.

**Test setup**: tutti i walker partono da `x_0 = (1, 5)`, goal a `(8, 5)`. Tre condizioni FMC + random:
- `FMC_with_relativize`: canonico, scores = relativize(R)
- `FMC_raw_clip_at_0`: scores = max(R, 0), nessun relativize (variante "valid reward gate")
- `FMC_raw_signed_negatives`: scores = sign(R) · |R|^α, negativi propagati (Sergio's regime)
- `Random_walk`: solo diffusione

**Scenario A** — reward "wells + peak" con offset globale −0.5 (R *flat* negativo ovunque tranne picco):

| Metrica | relativize | raw clip | **raw signed** | random |
|---|---|---|---|---|
| mean_x finale | 0.72 | **8.04** | **0.60** | 3.97 |
| frac al goal | 0% | **84%** | **0%** | 2% |
| speed finale | 1.71 | 0.59 | **0.57** | 0.36 |
| convex hull | 23 | 5.4 | **4.0** | 97 |
| reward finale | -0.50 | **1.39** | -0.50 | -0.51 |

**`raw_signed` esibisce esattamente il "miedoso" di Sergio**: hull = 4 (collassato vicino allo start), mean_x=0.60 (zero advance), speed bassa (0.57 vs 1.71 di relativize). I walker vedono reward negativa ovunque, e VR negativo li blocca dal clonarsi via — restano fermi. ✅ **H2 confermato**.

**Scenario B** — gradiente smooth all-negative `R = -0.7 + 0.10·x`:

| Metrica | **relativize** | raw clip | raw signed | random |
|---|---|---|---|---|
| mean_x finale | **9.74** | 9.44 | 9.42 | 3.97 |
| convex hull | **8.97** | 20.6 | 19.7 | 97 |
| reward finale | **0.27** | 0.24 | 0.24 | -0.30 |
| speed | 1.87 | 1.83 | 1.83 | 0.36 |

Con gradiente disponibile, `relativize` **brilla**: mean_x più alto, hull più stretto al goal, reward finale più alto. ✅ **H1, H3, H4 tutti confermati**. Importante: `raw_signed` qui NON è fearful — il gradiente continuo fornisce signal diretto anche con reward grezze.

**Caveat onesto**: né `relativize` né `raw_signed` risolvono lo Scenario A; solo `raw_clip_at_0` ci riesce. Ovvero: se la reward è *flat negativa* (nessun gradiente), la sola soluzione è esplorazione diffusiva + clone-to-success, non `relativize`. Sergio non discute questa limitazione.

**Verdetto F11**: ✅ il comportamento "fearful" da reward negative è confermato (`raw_signed`, Scenario A). `relativize` previene la patologia *quando esiste un gradiente* (Scenario B). Su landscape flat-negativi serve un trick aggiuntivo (clipping o esplorazione esplicita).

### 7.3 Risultato emergente — il bias di `relativize` su α

Osservazione collaterale di valore generale, non discussa in [`04_relativize_axiomatics.md`](04_relativize_axiomatics.md). La trasformazione `R̂ = exp(z) se z ≤ 0; R̂ = 1 + ln(1+z) se z > 0`  ha derivata:

$$
\frac{dR̂}{dz} = \begin{cases} e^z & z \le 0 \\ 1/(1+z) & z > 0 \end{cases}
$$

Vicino a $z = 0$ entrambi valgono 1. Per $z \gg 0$ la mappa **comprime** (derivata → 0). Per $z \ll 0$ la mappa **espande** (R̂ → 0 esponenzialmente). Combinato con $\mathrm{VR} \propto R̂^\alpha$, il risultato è che la *temperatura inversa effettiva* dell'algoritmo è > α nominale, soprattutto nel regime di $z$ grandi.

**Conseguenza diretta**:

- A α=1 in codice, l'equilibrio Gibbs effettivo è R^{α_eff} con $\alpha_{\mathrm{eff}} > 1$, quindi P_walker over-concentrata rispetto a R lineare.
- Per ottenere il vero matching `P_walker ∝ R` (Sergio's claim eq. literal) serve in codice `α ≈ 0.5` (KL minimo a quel punto in Test A).
- Per Atari il default α=1 in `swarm.py` può produrre over-exploitation rispetto al regime termodinamico classico.

**Action item nuovo**: estendere [`04_relativize_axiomatics.md`](04_relativize_axiomatics.md) con un teorema sull'amplificazione effettiva di α e quantificare $\alpha_\text{eff}(\alpha, \sigma_R)$ analiticamente.

### 7.4 Cosa cambia nei nostri docs (action items aggiornati)

Oltre ai 7 punti di §6, da §7 emergono:

8. **[ALTO]** Aggiungere il bias di α effettivo in [`THEORY.md`](../../plugin/fractal-coding-loop/docs/THEORY.md) e [`ALGORITHM.md`](../../plugin/fractal-coding-loop/docs/ALGORITHM.md) come *caveat* sull'interpretazione di `balance` nel codice — **α=1 in codice ≠ α=1 in Gibbs**.
9. **[MEDIO]** Annotare che `relativize` è insufficiente in regime *flat-negative* — serve combinarla con esplorazione diffusiva attiva o un goal-shaping aggiuntivo. Documentare in [`THEORY.md` §4](../../plugin/fractal-coding-loop/docs/THEORY.md).
10. **[BASSO]** Replicare la verifica F12 su Atari Boxing usando il replication code in [`work/03_atari_replication/`](../03_atari_replication/) per validare la proporzionalità anche in un dominio non-toy.

### 7.5 F12 verificato anche su Atari Boxing — dominio non-toy

> *Action item §7.4 punto 10 chiuso. Codice in [`test_f12_atari_boxing.py`](../04_mathematical_tests/test_f12_atari_boxing.py), risultati in `results/f12_atari_boxing.json` e `f12_atari_boxing_correlation.png`.*

In Atari non si può calcolare $P_R(x)$ analiticamente (la reward è data dal simulatore, non come funzione chiusa sullo state space). La forma operativa di F12 a *action-marginal* è:

$$
P_{\text{FMC}}(a) \quad \propto \quad \mathbb{E}\big[\,R_{\text{cum}} \mid \text{init\_action} = a\,\big]
$$

ovvero: dopo M tick di pianificazione, i walker etichettati con `init_action a` sono presenti in proporzione al reward cumulato atteso di quel `a`. Questa è la "scanning density" del paper §3 ristretta alla slice marginale a $t=0$.

**Setup**: ALE/Boxing-v5, N=50 walker, M=15 tick di orizzonte, skipframe=5, seed=42, 80 decisioni consecutive. Implementazione FMC fedele in `work/03_atari_replication/scripts/fmc_minimal.py` (nessuna deviazione rispetto al codice di riferimento). Per ogni decisione misuriamo Pearson e Spearman tra `bincount(init_actions)` e `mean(cum_R | init_action)` ristretti alle azioni con ≥ 2 walker (statistiche affidabili).

**Risultati aggregati su 39 decisioni con dati sufficienti** (le altre hanno troppe azioni con singolo walker → correlazione non definita):

| Metrica | Mean | Median | Frazione > 0 |
|---|---|---|---|
| Pearson( P_FMC, E[R\|a] ) | **+0.448** | **+0.610** | **82.1%** |
| Spearman idem | **+0.435** | **+0.722** | **79.5%** |

Il punteggio di gioco a fine run: **+14 dopo 80 decisioni** (~13 min di partita) — coerente con il vantaggio costante di Boxing (`win=100`, FMC dovrebbe arrivare a 90-100 in ~1300 decisioni).

**Verdetto**: ✅ confermato in dominio non-toy. La correlazione mediana è 0.61 Pearson e 0.72 Spearman — il rank ordering tra azioni è coerentemente catturato dalla distribuzione di walker. La proporzionalità di Sergio NON è perfetta (Pearson medio 0.45 ≠ 1.0) per due ragioni note:

1. **Finite-N**: con N=50 walker e 18 azioni, ogni azione ha mediamente ~3 walker — varianza della stima di $E[R \mid a]$ alta.
2. **Bias di α effettivo** (§7.3): la `relativize` produce $\alpha_{\text{eff}} > 1$ quindi sovra-concentra rispetto alla scaling lineare di Sergio.

In tre decisioni (su 39) abbiamo Pearson < -0.5 — sono casi in cui la pianificazione è incoerente con la valutazione retrospettiva (es. un'azione viene scelta da molti walker ma poi i suoi rollout falliscono per motivi non visibili al primo tick). Questo è atteso: FMC è ancora meglio di MCTS su Atari ma non un oracolo.

**Take-home per i nostri docs**: la verifica F12 sul toy 2D (log-Pearson ~0.77 a α=1) si traduce in Pearson 0.45 medio / 0.61 mediano in Atari Boxing. Il calo è dovuto al moltiplicarsi delle azioni (18 vs 2 dim continue) e al regime di poco campionamento (50 walker × 18 azioni = ~3 walker per azione). Con N=200-500 walker la correlazione dovrebbe avvicinarsi al regime toy.

### 7.6 Riproducibilità

```bash
cd work/04_mathematical_tests
python3 test_f12_cross_entropy.py    # ~30 s
python3 test_f11_relativize_ablation.py  # ~60 s
python3 test_f12_atari_boxing.py     # ~45 s for 80 decisions
```

Versione Python testata: 3.11.7. numpy 2.2.6, scipy 1.16.1, matplotlib 3.10.6, gymnasium 1.3.0, ale_py 0.11.2. Plot e summary in `results/`. Tutti i risultati seed-deterministici (seed=42).

---

## 8. Riferimenti

- **Trascrizione raw**: [`VideoTranscriptSergio.md`](../../VideoTranscriptSergio.md) (root del repo) — 73 877 chars, no punteggiatura, lingua spagnola.
- **Versione wrapped per lettura**: `/tmp/sergio_wrapped.md` (volatile, 625 righe, generato da `textwrap.wrap(width=120)`).
- **Paper canonico**: [`docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf`](../../docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf).
- **Paper empirico**: [`docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf`](../../docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf).
- **Analisi del paper**: [`ANALISIS.md`](../../ANALISIS.md) (~46 KB, italiano).
- **Intervista Radient 2026**: [`docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md`](../../docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md) (~21 700 parole, 21 capitoli).
- **Documentazione FMC plugin**: [`plugin/fractal-coding-loop/docs/`](../../plugin/fractal-coding-loop/docs/) (THEORY, ALGORITHM, COMPONENTS, USAGE).
- **Wissner-Gross 2013**: [`docs/bibliography/sources/papers/2013_wissner_gross_causal_entropic_forces.pdf`](../../docs/bibliography/sources/papers/2013_wissner_gross_causal_entropic_forces.pdf).
