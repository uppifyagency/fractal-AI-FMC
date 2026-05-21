# H-B1a — risultato: l'esponente di Lyapunov dello swarm (2026-05-21)

Test di esistenza della Congettura B riformulata ([deep dive 09](../02_deep_dives/09_chaos_order_frontier_formalization.md) §4.2):
**l'esponente di Lyapunov dello swarm $\lambda_1$ attraversa lo zero al variare di
$\alpha$?** Harness: [`lambda1_harness.py`](lambda1_harness.py). Dati grezzi:
[`results/hb1a_lambda1.json`](results/hb1a_lambda1.json).

> **Esito in una riga.** H-B1a è **INCONCLUSIVO** — non perché $\lambda_1$ non
> attraversi lo zero, ma perché **$\lambda_1$ non è scale-free**: cambia *segno*
> al variare della scala di perturbazione $\delta_0$. Il caveat che dd09 §3.1
> aveva previsto è confermato empiricamente. Non una falsificazione della
> Congettura B — una falsificazione dello *stimatore* $\Psi_1$ ingenuo.

---

## 1. Metodo

Twin-trajectory / Benettin (dd09 §6). Due swarm A (riferimento) e B (perturbato),
evoluti sotto la **stessa** alea realizzata (due RNG con seed identico → lockstep:
il pattern di consumo RNG di FMC non dipende dal virtual reward, quindi i tiri
restano sincronizzati). B parte come A con ogni walker perturbato; la separazione
totale in spazio-stati è riscalata a $\delta_0$. A ogni tick: una iterazione FMC
su ciascuno swarm, misura della separazione $\|W^A-W^B\|$ in spazio-`observe`
(wrap-safe), log della crescita, poi rinormalizzazione di B a $\delta_0$ (Benettin).
$\lambda_1$ = media della log-crescita per tick.

Kernel `fmc-core` **invariato**: l'harness replica il tick pubblico di `plan()` e
chiama `fmc.core.{virtual_reward, clone_step}` immutati. Due task con landscape di
reward diversi: `navigation2d` (distanza in spazio posizione) e `pendulum`
(energia angolare). $\beta=1$, $N=64$, $M=20$, 30 seed.

---

## 2. Risultati

### 2.1 Sweep di $\alpha$ ($\delta_0 = 10^{-3}$ fisso)

| $\alpha$ | $\lambda_1$ navigation2d | $\lambda_1$ pendulum |
|---:|---:|---:|
| 0.00 | $+0.044 \pm 0.020$ | $+0.026 \pm 0.014$ |
| 0.05 | $+0.007 \pm 0.011$ | $+0.029 \pm 0.014$ |
| 0.10 | $+0.020 \pm 0.017$ | $+0.007 \pm 0.003$ |
| 0.20 | $+0.023 \pm 0.014$ | $+0.032 \pm 0.014$ |
| 0.35 | $+0.002 \pm 0.011$ | $+0.006 \pm 0.010$ |
| 0.50 | $+0.020 \pm 0.015$ | $+0.012 \pm 0.012$ |
| 0.75 | $+0.019 \pm 0.015$ | $-0.004 \pm 0.004$ |
| 1.00 | $+0.003 \pm 0.012$ | $+0.014 \pm 0.013$ |

$\lambda_1 \approx 0$ entro il rumore **ovunque**, su entrambi i task. Nessun
attraversamento dello zero *sign-resolved* (l'unico valore negativo — pendulum
$\alpha=0.75$ — è a $\sim 1$ SEM da zero). Nessun trend monotono risolto.

### 2.2 Dipendenza dalla scala $\delta_0$ — il dato decisivo

| (task, $\alpha$) | $\delta_0{=}10^{-2}$ | $\delta_0{=}10^{-3}$ | $\delta_0{=}10^{-4}$ |
|---|---:|---:|---:|
| navigation2d, $\alpha{=}0.0$ | $+0.092 \pm 0.021$ | $+0.044 \pm 0.020$ | $\mathbf{-0.006 \pm 0.003}$ |
| navigation2d, $\alpha{=}0.1$ | $+0.082 \pm 0.023$ | $+0.020 \pm 0.017$ | $\mathbf{-0.008 \pm 0.002}$ |
| navigation2d, $\alpha{=}0.5$ | $+0.070 \pm 0.021$ | $+0.020 \pm 0.015$ | $\mathbf{-0.007 \pm 0.004}$ |
| pendulum, $\alpha{=}0.1$ | $+0.132 \pm 0.028$ | $+0.007 \pm 0.003$ | $+0.007 \pm 0.003$ |

Su `navigation2d`, a **tutti e tre** gli $\alpha$ testati, $\lambda_1$ **cambia
segno** al rimpicciolire di $\delta_0$: da chiaramente positivo ($\delta_0=10^{-2}$)
a chiaramente negativo e sign-resolved ($\delta_0=10^{-4}$, $\sim$ 3 SEM sotto zero).
Su `pendulum` non c'è cambio di segno nell'intervallo testato, ma $\lambda_1$ varia
comunque di $\sim$18× con $\delta_0$ ($+0.132 \to +0.007$). **$\lambda_1$ non è
scale-free.**

---

## 3. Interpretazione

Il meccanismo è chiaro e segue dalla struttura di FMC. L'operatore di cloning
(Def. 4) è **discontinuo** — un clone scatta o no, è una funzione a gradino del
virtual reward. Conseguenza per la separazione delle traiettorie gemelle:

- **$\delta_0$ piccolo** ($10^{-4}$): la perturbazione è troppo piccola per far
  *flippare* le decisioni di cloning — i due swarm prendono le stesse decisioni
  di clone, restano "dentro la stessa cella" della mappa a tratti, e l'unica
  dinamica è lo step del simulatore + il cloning identico, che **contrae**:
  $\lambda_1 < 0$ (lato ordinato).
- **$\delta_0$ grande** ($10^{-2}$): la perturbazione fa attraversare i confini
  delle celle di decisione → le decisioni di clone divergono → separazione che
  cresce: $\lambda_1 > 0$ (lato caotico).

La mappa dello swarm FMC è dunque **a tratti**, e il suo "esponente di Lyapunov"
è genuinamente scala-dipendente: ordinato alle scale infinitesime, caotico alle
scale finite. Non esiste un $\lambda_1$ scale-free, e quindi non esiste la linea
$\lambda_1 = 0$ su cui dd09 §4.3 aveva ancorato la frontiera.

Questo è **esattamente** ciò che dd09 §3.1 aveva segnalato come rischio:
> *"Il cloning è un operatore discontinuo… l'esponente di Lyapunov classico
> rischia di dare $\lambda_1=0$ in modo banale… è possibile che $\lambda_1$
> dipenda da $\delta$. Quella $\delta$-dipendenza va caratterizzata, non assunta
> via."*

L'harness la caratterizza: non solo $\lambda_1$ dipende da $\delta$ — **cambia
segno**.

---

## 4. Verdetto e implicazione per la Congettura B

**H-B1a: INCONCLUSIVO.** Lo stimatore twin-trajectory ingenuo non risolve un
$\Psi_1$ scale-free per la mappa-swarm di FMC. Lo sweep di $\alpha$ a $\delta_0$
fisso dà $\lambda_1 \approx 0$ entro il rumore ovunque — un risultato che, da
solo, sarebbe "non attraversa lo zero", ma che il δ-check rivela essere un
artefatto della scala scelta.

**Implicazione per dd09 / Congettura B.** dd09 §3 aveva gradato le tre candidate
$\Psi$: $\Psi_3$ ($b_{\text{eff}}$) **falsificata**; $\Psi_2$ (volume del cono)
**assorbita** in $\Psi_1$ via Pesin; $\Psi_1$ (Lyapunov) promossa a **candidata
principale** — "l'unica con valore critico universale". H-B1a mostra che $\Psi_1$,
misurata in modo ingenuo, **non è scale-free**. Stato aggiornato: **tutte e tre le
candidate $\Psi$ sono ora compromesse.** L'ipotesi nulla **H-B4** ("non esiste una
statistica di frontiera task-indipendente") **guadagna terreno**.

Onestà sui limiti: questo **non** falsifica la sostanza della Congettura B. Ha
testato $\Psi_1$ con *un* stimatore (twin-trajectory a $\delta$ finito). Un
trattamento *scale-resolved* — $\lambda_1(\delta_0)$ come curva, non come scalare —
o uno stimatore diverso potrebbero ancora salvare un $\Psi_1$ ben posto. Quello
che è stabilito: la frontiera $\lambda_1=0$ **come scalare** non esiste per FMC.

> **Nota speculativa** (non testata). Un sistema senza una scala caratteristica —
> $\lambda_1$ che dipende dalla scala invece di convergere a un valore — è esso
> stesso una firma classica di *criticità* (scale-invariance, self-organized
> criticality, Bak 1987). È *possibile* che la δ-dipendenza di FMC non sia un
> difetto di misura ma il modo in cui la frontiera caos/ordine si manifesta in un
> sistema a cloning discreto. Pura speculazione: andrebbe testata misurando se
> $\lambda_1(\delta_0)$ ha forma a legge di potenza. Fuori dallo scope di H-B1a.

---

## 5. Prossimi passi

1. **Trattamento scale-resolved** — misurare $\lambda_1(\delta_0)$ come curva su
   un range ampio di $\delta_0$; cercare (a) un eventuale plateau a $\delta_0\to 0$
   come vero $\Psi_1$, oppure (b) una forma a legge di potenza (la nota speculativa
   del §4).
2. **H-B1b / H-B1c sono bloccate** finché $\Psi_1$ non è ben posta: non si può
   testare "il throughput picca a $\lambda_1\approx 0$" se $\lambda_1$ non è
   definito. dd09 §4.2 va aggiornato.
3. In alternativa, una **$\Psi$ diversa** — non basata sulla sensibilità a
   perturbazione puntuale ma su una misura d'ensemble (entropia di stato dello
   swarm) che non soffra della discontinuità del cloning.

---

## 6. Riproducibilità

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" work/13_chaos_order/lambda1_harness.py     # ~14 s (CPU)
```

Kernel `fmc-core` invariato — l'harness replica solo il tick pubblico di `plan()`.

---

*Fine HB1A_RESULT.md. H-B1a inconclusivo: $\lambda_1$ non è scale-free (cambia
segno con $\delta_0$ su navigation2d, ∀α testato), confermando dd09 §3.1. Tutte e
tre le candidate $\Psi$ di dd09 ora compromesse; l'ipotesi nulla H-B4 guadagna
terreno. Prossimo passo: $\lambda_1(\delta_0)$ scale-resolved.*
