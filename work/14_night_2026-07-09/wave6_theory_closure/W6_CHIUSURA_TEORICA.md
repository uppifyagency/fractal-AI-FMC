# W6 — Chiusura dei due buchi teorici aperti di MATH_CANON v0.8.0

> **Data**: 2026-07-10 (continuazione della sessione night_2026-07-09). **Ruolo**: research associate + falsificatore.
> **Oggetto**: chiudere in **forma chiusa** i due open item del [`HANDOFF §6-teoria`](../HANDOFF.md):
> **(G1)** la correzione +13% del coefficiente di diffusione di Teorema 2′.5 (co-ancestry del resampling pairwise), che W3B aveva **misurato ma non derivato**;
> **(G2)** il ponte $\alpha_{\rm eff}\leftrightarrow s_{\rm eff}$ in un'unica derivazione, che risolve la tensione §7.3 del paper teorico.
> **Script** (numeri reali, seed 20260709, numpy 2.2.6 / scipy 1.16.1):
> [`w6a_coancestry_Ne.py`](w6a_coancestry_Ne.py) (G1), [`w6b_alpha_s_bridge.py`](w6b_alpha_s_bridge.py) (G2).
> **Marcatura**: `[DIM]` = forma chiusa dimostrata; `[DIM-LO]` = derivazione in forma chiusa al leading order in $1/N$ (chiusura di campo-medio del coalescente, resto $O(1/N^2)$), verificata numericamente; `[NUM]` = solo Monte Carlo.

---

## 0. Verdetto in una riga

- **G1 CHIUSA `[DIM-LO]`.** Il ~+13% è la **probabilità di co-ancestry pairwise** per tick. Enumerando le vie di coalescenza si ottiene la forma chiusa $\lambda N = 2\phi_0 + \langle a_{\rm in}^2\rangle - 2\langle a_{\rm in}a_{\rm out}\rangle$, **derivata (non fittata)**, che riproduce il valore misurato a **<0.1%** ($\sigma_v{=}0.5$: **0.6755** vs 0.6759). Il coefficiente di diffusione passa da `[NUM]` a `[DIM-LO]`. *(Numero convergente $+12.8\%$; il $+13.1\%$ del primo run era un artefatto di quadratura Gauss-Hermite deg=80, corretto in review — §5.)*
- **G2 CHIUSA `[DIM]` per l'identità, `[DIM-NUM]` per l'unificazione.** $\alpha_{\rm eff}$ e $s_{\rm eff}$ **non** sono due temperature rivali: sono la **stessa** selezione linearizzata in due sistemi di coordinate, composti dalla regola della catena — $\boxed{s_{\rm eff}=2\Phi'(0)\,\alpha_{\rm eff}\,\Delta R}$ — con costante di composizione $2\Phi'(0)$ in forma chiusa. Verificato end-to-end (err → 0.00% nel limite debole). **L'unificazione** (che $\sigma_v$ sia *determinato* da `relativize`, non un parametro libero) è **testata e confermata** da una simulazione accoppiata (§2.5, T3 <0.9% con $\tau$ vincolato dalla popolazione). **Bonus**: $\Phi(m)$ stessa ha forma chiusa in CDF normali.

---

## 1. G1 — La correzione di co-ancestry in forma chiusa `[DIM-LO]`

### 1.1 Il kernel esatto (neutro)

Da [`w3b_mutation_diffusion.one_tick`](../wave3_validation/w3b_mutation_diffusion.py) (verificato riga per riga): update **sincrono**, ogni walker $i$ estrae $g_i\sim\mathcal N(0,\sigma_v^2)$, $\mathrm{VR}_i=e^{g_i}$; sceglie un partner $j(i)$ **uniforme** tra gli $N-1$ diversi da sé (`offset ∈ {1..N-1}`); adotta il tipo del partner con probabilità $a_{\rm FMC}(\mathrm{VR}_{j}/\mathrm{VR}_i)=\operatorname{clip}(e^{g_j-g_i}-1,0,1)$. Tutti aggiornano in parallelo sui tipi **pre-update**.

### 1.2 Decadimento dell'eterozigosità = coalescenza pairwise

Il tasso di decadimento $\lambda$ di $H=\mathbb E[2x(1-x)]$ è, per una popolazione neutra ben mescolata, la **probabilità che due walker distinti condividano il genitore un tick indietro** ($\lambda = p_{\rm coal} = 1/N_e$; convenzione dello script: $\lambda N \leftrightarrow 2\phi_0$ leading-order). Guardando indietro, il genitore di $i$ è
$$\text{parent}(i)=\begin{cases} i & \text{se $i$ non ha clonato}\quad(\text{prob }1-a_{\rm out}(g_i))\\ j(i) & \text{se $i$ ha clonato}\quad(\text{prob }a_{\rm out}(g_i)).\end{cases}$$

**Definizioni chiave** (le due "viste" della clip, $t,g\sim\mathcal N(0,\sigma_v^2)$):
$$a_{\rm in}(t)=\mathbb E_g[\operatorname{clip}(e^{t-g}-1,0,1)]\ \ (\text{prob. che un walker qualsiasi cloni \emph{su} un bersaglio di log-fitness }t),$$
$$a_{\rm out}(t)=\mathbb E_g[\operatorname{clip}(e^{g-t}-1,0,1)]=a_{\rm in}(-t),\qquad \phi_0=\mathbb E_t[a_{\rm in}(t)]=\mathbb E_t[a_{\rm out}(t)].$$

### 1.3 Enumerazione completa delle vie di coalescenza (leading order $1/N$)

Due offspring distinti $i_1\ne i_2$ coalescono sse condividono il genitore $k$. Tre vie (esaustive; il caso "entrambi genitori-di-sé" richiede $i_1=i_2$, escluso):

| via | condizione | probabilità (leading $1/N$) |
|---|---|---|
| **(D)** entrambi clonano sullo **stesso** $k$ | $j(i_1)=j(i_2)=k$, entrambi accettano, $k\notin\{i_1,i_2\}$ | $\tfrac{N-2}{(N-1)^2}\,\mathbb E[a_{\rm in}^2]\approx\tfrac1N\mathbb E[a_{\rm in}^2]$ |
| **(B)** $i_1$ è genitore-di-sé, $i_2$ ci clona sopra | $j(i_2)=i_1$, $\mathrm{acc}_{i_2}{=}1$, $\mathrm{acc}_{i_1}{=}0$ | $\tfrac1{N-1}\mathbb E_t[a_{\rm in}(t)(1-a_{\rm out}(t))]$ |
| **(C)** simmetrico di (B) | $j(i_1)=i_2$, $\mathrm{acc}_{i_1}{=}1$, $\mathrm{acc}_{i_2}{=}0$ | $=$ (B) |

Le correlazioni in (B)/(C) passano per la fitness **condivisa** $g_{i_1}$ (di $i_2$ su $i_1$ e del mancato-clone di $i_1$); in (D) per la fitness condivisa del bersaglio $g_k$. Le eventuali collisioni di indice ($j(i_1)=i_2$ dentro (B), ecc.) sono $O(1/N)$ su termini già $O(1/N)$ → $O(1/N^2)$, trascurabili. Sommando e usando $\mathbb E[a_{\rm in}]=\phi_0$:
$$\boxed{\;\lambda N \;=\; \underbrace{2\phi_0}_{\text{baseline indip.}}\;+\;\underbrace{\big(\langle a_{\rm in}^2\rangle-2\langle a_{\rm in}\,a_{\rm out}\rangle\big)}_{\text{correzione di co-ancestry}}\;,\qquad N_e=\frac{N}{\lambda N}.\;}$$

Il baseline $2\phi_0$ è la stima "flip indipendenti" di W3B; il termine tra parentesi è la correzione **finora misurata al $+13\%$**, ora in forma chiusa (tre momenti della clip). Fattore di inflazione $\kappa=\lambda N/(2\phi_0)=1+\tfrac{\langle a_{\rm in}^2\rangle-2\langle a_{\rm in}a_{\rm out}\rangle}{2\phi_0}$.

### 1.4 Verifica: forma chiusa (a_in analitica + `scipy.quad` adattiva) vs kernel esatto

$a_{\rm in}(t)=\mathbb E_{u\sim\mathcal N(t,\sigma_v^2)}[\operatorname{clip}(e^u{-}1,0,1)]$ ha la **stessa forma chiusa** di $\Phi$ (§2.2) ma con varianza $\sigma_v^2$; le tre attese $\phi_0,\langle a_{\rm in}^2\rangle,\langle a_{\rm in}a_{\rm out}\rangle$ sono integrali 1-D calcolati con Gauss-Kronrod adattivo (nessun artefatto di spigolo).

| $\sigma_v$ | $2\phi_0$ (baseline) | correzione | $\kappa$ | $\lambda N$ **chiuso** | $\lambda N$ **misurato** (media $N{=}100{-}800$) | scarto |
|---|---|---|---|---|---|---|
| 0.25 | 0.34348 | $+11.3\%$ | 1.113 | 0.38240 | — | — |
| **0.5** | 0.59866 | $\mathbf{+12.8\%}$ | 1.128 | **0.67549** | **0.6759** | **+0.1%** |
| 1.0 | 0.78669 | $+8.8\%$ | 1.088 | 0.85623 | 0.8696 | +1.6% |

A $\sigma_v{=}0.5$ la forma chiusa combacia con il misurato a **+0.1%** (per-$N$: +1.4%, +0.2%, +1.0%, +0.5% su $N=100,200,400,800$). **Il ~+13% era esattamente la correzione di co-ancestry.** Il segno è ovvio a posteriori: i bersagli ad alta fitness ($a_{\rm in}$ grande) attraggono più cloni contemporaneamente ⇒ più coalescenze ⇒ $N_e$ **minore** ⇒ drift più rapido. Cross-check indipendente in review (parent-map counting + MC non-quadratura): correzione $+12.8\%$, confermata (§5).

### 1.5 Stato di prova onesto

**Ora `[DIM-LO]`** (era `[NUM]`): il coefficiente è **derivato in forma chiusa** al leading order in $1/N$, non più solo misurato. Resta **non** `[DIM]` pieno per un solo motivo standard e dichiarato:
1. **Chiusura di campo-medio** "genitori distinti ⇒ coppia casuale che contribuisce $H_t$": la review avversariale ha **dimostrato** che nel caso neutro la ricorsione $\mathbb E[H_{t+1}]=(1-p_{\rm coal})\mathbb E[H_t]$ è **esatta** (tipo ⊥ fitness ⊥ parent-map per scambiabilità), quindi la chiusura *non* è un'approssimazione al leading order; il resto $O(1/N^2)$ è solo la differenza fra $p_{\rm coal}$ finite-$N$ e la sua forma leading (verificata: gap $\sim1/N$, +1.84%→+0.25% per $N=50\to400$).
2. **Limite di diffusione funzionale** (martingale problem / Lindeberg con la clip a spigoli) non dimostrato — comune a tutta la letteratura Wright-Fisher, non specifico di qui. È l'unico buco rimasto verso `[DIM]` pieno.

Ciò che il task chiedeva — la correzione $\sim13\%$ in forma chiusa — **è chiuso**.

---

## 2. G2 — Il ponte $\alpha_{\rm eff}\leftrightarrow s_{\rm eff}$ `[DIM]`

### 2.1 La tensione da risolvere

Due quantità "effettive", entrambe corrette, sembravano rivali:
- **Teorema 4**: $\alpha_{\rm eff}=C\,\alpha/\sigma_R$ — temperatura inversa nello spazio $\log\mathrm{VR}$, per unità di reward.
- **Teorema 2′.5**: $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$ — drift di frequenza per tick, con $\Phi(m)=\mathbb E_{u\sim\mathcal N(m,2\sigma_v^2)}[\operatorname{clip}(e^u-1,0,1)]$.

**Risoluzione**: non sono due temperature; sono la **stessa** selezione linearizzata in **due sistemi di coordinate**, composti dalla regola della catena.

### 2.2 Bonus: $\Phi(m)$ e $\Phi'(0)$ in forma chiusa `[DIM]`

Spezzando per regioni della clip ($u\le0$: 0; $0<u<\ln2$: $e^u-1$; $u\ge\ln2$: 1) e completando il quadrato ($\int_0^{\ln2}e^u\mathcal N(u;m,\tau^2)du=e^{m+\tau^2/2}[F(\ln2;m{+}\tau^2)-F(0;m{+}\tau^2)]$, $\tau^2=2\sigma_v^2$, $F=$ CDF di $\mathcal N(\cdot,\tau^2)$):
$$\boxed{\;\Phi(m)=e^{m+\tau^2/2}\big[F(\ln2;m{+}\tau^2)-F(0;m{+}\tau^2)\big]\;+\;1-2F(\ln2;m)+F(0;m).\;}$$
Quindi $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$ è **forma chiusa piena** (non più integrale MC di W3B). Derivata della clip (via Stein, $\mathbb E[u\,h(u)]=\tau^2\mathbb E[h'(u)]$, con $c'(u)=e^u\mathbf 1_{0<u<\ln2}$):
$$\boxed{\;\Phi'(0)=\int_0^{\ln 2} e^u\,\mathcal N(u;0,\tau^2)\,du = e^{\tau^2/2}\big[F(\ln2;\tau^2)-F(0;\tau^2)\big].\;}$$
**Interpretazione**: solo la **banda di transizione** $0<u<\ln2$ (accettazione strettamente in $(0,1)$) trasmette selezione marginale — sotto 0 rigettato, sopra $\ln2$ saturo a 1. Verifica: chiuso vs finite-diff **0.00%**, vs Stein-MC **0.01–0.05%**.

### 2.3 I due link e la composizione `[DIM]`

**LINK A — `relativize` converte un gap di reward in un gap di $\log\mathrm{VR}$.** Con $\log\mathrm{VR}=\alpha\log\widehat R(z)$, $z=(R-\mu)/\sigma_R$. I due tipi sono uno **shift di $\pm\Delta R/2$ dell'intera distribuzione** di reward, quindi ogni walker contribuisce la sua pendenza locale $g(z_i)$ e la costante corretta è la **Jacobiana mediata sulla popolazione** $C=\mathbb E[g(z)]$, **non** $g(\bar z)$:
$$\delta=\mathbb E_A[\log\mathrm{VR}]-\mathbb E_B[\log\mathrm{VR}]\approx \alpha\,\mathbb E[g(z)]\,\frac{\Delta R}{\sigma_R}=\underbrace{C\,\frac{\alpha}{\sigma_R}}_{=\ \alpha_{\rm eff}\ (\text{Thm 4})}\,\Delta R=\alpha_{\rm eff}\,\Delta R.$$
> ⚠️ **Correzione in review (§5, Difetto 1):** una versione precedente scriveva $\delta\approx\alpha\,g(\bar z)\,\Delta R/\sigma_R$ con $g(0){=}1$, che darebbe erroneamente $\alpha_{\rm eff}=\alpha/\sigma_R$ (sovrastima $+39\%$). La costante giusta è $C=\mathbb E[g(z)]=0.7223$ (identità di Stein, come in Thm 4). Il codice usava già la pendenza OLS corretta; era l'unico passaggio *scritto* sbagliato.

Verifica (popolazione relativize, $N=4\times10^6$): $\delta_{\rm mis}/\Delta R\to\alpha_{\rm eff}=0.72204$ ($=C\alpha/\sigma_R$) con err **→0.01%** per $\Delta R\to0$.

**LINK B — la clip converte un gap di $\log\mathrm{VR}$ in drift di frequenza** (selezione debole, sviluppo dispari):
$$s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)=2\Phi'(0)\,\delta+O(\delta^3).$$
Verifica ($\sigma_v{=}0.5$, $2\Phi'(0)=0.94474$): $s_{\rm eff}/\delta$ **→ 0.00%** dal valore chiuso per $\delta\to0$ (0.89%→0.22%→0.06%→0.01%→0.00% su $\delta=0.2\!\downarrow\!0.01$).

**COMPOSIZIONE** (regola della catena):
$$\boxed{\;s_{\rm eff}\;=\;\underbrace{2\Phi'(0)}_{\text{trasmissione della clip}}\;\cdot\;\underbrace{\alpha_{\rm eff}}_{\text{temperatura inversa}}\;\cdot\;\Delta R\;+\;O(\Delta R^3).\;}$$
Verifica end-to-end: $s_{\rm eff}^{\rm(clip\ vero)}$ vs $2\Phi'(0)\alpha_{\rm eff}\Delta R$, err **→ 0.00%** ($\Delta R=0.1\to0.02$: 0.22%→0.05%→0.00%).

### 2.4 Cosa risolve

La §7.3 del paper teorico segnalava "$\alpha_{\rm eff}$ e $s_{\rm eff}$ sono due misure separate, entrambe corrette". Ora sono **una**: $\alpha_{\rm eff}$ è la sensibilità della selezione in **coordinate reward→log-VR**; $s_{\rm eff}$ la sensibilità in **coordinate log-VR→frequenza**; la costante di raccordo è $2\Phi'(0)$, la trasmissione marginale della clip. Il legame fra le due dispersioni: $\sigma_v$ (rumore per-tick di W3B) = spread **entro-tipo** di $\log\mathrm{VR}$, indotto da `relativize`.

### 2.5 Identificazione di $\sigma_v$ verificata su simulazione accoppiata `[DIM-NUM]`

> Chiude il **Difetto 2** della review (§5): "$\sigma_v$ è *determinato* da `relativize` o è un parametro libero?". Nei test §2.3 $\sigma_v$ era un input; qui **non lo è**. Script: [`w6c_coupled_identification.py`](w6c_coupled_identification.py).

Modello accoppiato, nessun input libero oltre al modello di reward: ogni tick (i) ridisegno $R_i\sim\mathcal N(m_{\rm tipo},\sigma_{\rm within}^2)$, $m_A{=}\mu{+}\Delta R/2$, $m_B{=}\mu{-}\Delta R/2$; (ii) applico `relativize` sulla popolazione **pooled** → $\log\mathrm{VR}$; (iii) eseguo il **vero** clone pairwise. Misuro il drift realizzato del tipo A a $x{=}0.5$: $s_{\rm eff}^{\rm mis}=4\,\mathbb E[\Delta x]$. Il $\sigma_v$ **non è imposto**: è *letto* dalla popolazione come spread entro-tipo di $\log\mathrm{VR}$ ($s_A,s_B$). Per una coppia mista A-B la clip media una **differenza** di log-VR $\sim\mathcal N(\delta,\,s_A^2{+}s_B^2)$, quindi l'identificazione onesta è
$$\boxed{\;\tau^2=s_A^2+s_B^2\quad(\text{= }2\sigma_v^2\text{ solo se }s_A=s_B).\;}$$

| $\Delta R$ | $s_A$ | $s_B$ | $s_A{=}s_B$? | $\tau$ (vincolato) | $s_{\rm eff}^{\rm mis}$ | $\Phi(\delta;\tau)-\Phi(-\delta;\tau)$ | **T3 err** | bridge $2\Phi'(0)\alpha_{\rm eff}\Delta R$ | T4 err |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.669 | 0.789 | NO | 1.035 | 0.24563 | 0.24528 | **0.1%** | 0.24763 | 0.8% |
| 0.15 | 0.707 | 0.781 | NO | 1.053 | 0.14913 | 0.14877 | **0.2%** | 0.14921 | 0.1% |
| 0.10 | 0.724 | 0.773 | NO | 1.059 | 0.09961 | 0.09947 | **0.1%** | 0.09958 | 0.0% |
| 0.05 | 0.739 | 0.764 | sì | 1.063 | 0.04957 | 0.04993 | **0.7%** | 0.04984 | 0.5% |
| 0.025 | 0.746 | 0.759 | sì | 1.064 | 0.02543 | 0.02521 | **0.9%** | 0.02493 | 2.0% |

**Esito (T3):** il drift misurato dalle **dinamiche di clone accoppiate** combacia con $\Phi(\delta;\tau)-\Phi(-\delta;\tau)$ a $\tau$ **vincolato dalla popolazione** entro **0.1–0.9%** su tutto il range. Quindi $\sigma_v$ **è determinato da `relativize`**, non un parametro libero: data la popolazione, lo spread entro-tipo di log-VR fissa completamente il drift della clip. L'unificazione regge. **Sfumatura onesta** (T1): $s_A\ne s_B$ lontano dal punto neutro (0.669 vs 0.789 a $\Delta R{=}0.25$), convergenti solo per $\Delta R\to0$ — $\sigma_v$ *non* è un unico scalare, ma la forma $\tau^2=s_A^2+s_B^2$ è esatta e T3 combacia anche dove $s_A\ne s_B$.

---

## 3. Enunciati aggiornati (forma pubblicabile)

> **Teorema 2′.5 (coefficiente di diffusione, chiuso) `[DIM-LO]`.** Per il kernel FMC neutro sincrono pairwise, il tasso di decadimento dell'eterozigosità per tick è, al leading order in $1/N$,
> $$\lambda=\frac1{N}\Big(2\phi_0+\langle a_{\rm in}^2\rangle-2\langle a_{\rm in}a_{\rm out}\rangle\Big),\qquad N_e=\frac{N}{\lambda N},$$
> con $a_{\rm in}(t)=\mathbb E_g[\operatorname{clip}(e^{t-g}-1,0,1)]$, $a_{\rm out}(t)=a_{\rm in}(-t)$, $\phi_0=\langle a_{\rm in}\rangle$, $g,t\sim\mathcal N(0,\sigma_v^2)$. La parentesi è la correzione di co-ancestry del resampling pairwise ($+13.1\%$ a $\sigma_v{=}0.5$), verificata a $+0.1\%$ contro il kernel esatto. Sostituisce l'$N_e$ leading-order misurato di W3B.

> **Teorema 4′ (ponte $\alpha_{\rm eff}\!\to\! s_{\rm eff}$) `[DIM]`.** Nel limite di selezione debole, drift di selezione e temperatura inversa effettiva sono legati da $s_{\rm eff}=2\Phi'(0)\,\alpha_{\rm eff}\,\Delta R+O(\Delta R^3)$, con $\Phi'(0)=e^{\tau^2/2}[F(\ln2;\tau^2)-F(0;\tau^2)]$ ($\tau^2=2\sigma_v^2$) trasmissione marginale della clip. Inoltre $\Phi(m)$ è in forma chiusa (CDF normali), quindi $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$ non richiede quadratura.

---

## 4. Cosa resta aperto (ridotto)

- **[G1 residuo]** $[DIM]$ pieno del coefficiente di diffusione = provare il **limite di diffusione funzionale** (martingale problem / Lindeberg con la clip a spigoli). La chiusura di campo-medio è ora dimostrata esatta al leading order (§1.5). Ridotto da "correzione ignota" a "tightness standard WF".
- **Estensione a $K$ tipi → Ewens (1972)** — invariata (l'$\mathcal S$ reale inietta configurazioni; qui 2 tipi).
- **Legame quantitativo con il compounding di exp17 (Cong. D)** — invariato; il ponte G2 dà ora la meccanica ($\alpha_{\rm eff}\propto1/\sigma_R$ → shaping tiered), ma la super-additività chain-tier resta da formalizzare.

---

## 5. Log della review avversariale (falsificatore Opus, refutation-oriented)

Un subagent con mandato di **refutare** ha riletto le derivazioni, ri-derivato l'enumerazione coalescente da zero e scritto due verificatori indipendenti ([`w6_adversarial_coalescence.py`](w6_adversarial_coalescence.py), [`w6_adversarial_bridge.py`](w6_adversarial_bridge.py), che **non** riusano quadratura/forme chiuse degli autori).

**G1 → CONFERMATO.** Enumerazione completa (nessun 4° cammino; le uguaglianze possibili del parent-map sono $i_1{=}i_2$ escluso, $i_1{=}j(i_2)$, $j(i_1){=}i_2$, $j(i_1){=}j(i_2)$). Prefattori verificati con parent-map counting (gap $\sim1/N$). Direzione di $a_{\rm out}$ corretta. Ricorsione $\mathbb E[H_{t+1}]{=}(1{-}p_{\rm coal})\mathbb E[H_t]$ **dimostrata esatta** nel neutro. MC non-quadratura: correzione $+12.8\%$ (positiva, non artefatto). **Nit sanato**: il $+13.1\%$ del primo run veniva da Gauss-Hermite deg=80; ricalcolato robusto → $+12.8\%$ ($\lambda N=0.6755$). Tag `[DIM-LO]` giudicato **onesto**.

**G2 → CONFERMATO-CON-CAVEAT, entrambi i caveat ora chiusi.** $\Phi(m)$ e $\Phi'(0)$ esatti a $\sim10^{-16}$ (brute-force `scipy.quad` con region-split) su un range di $m$ e $\sigma_v$; nessun termine delta mancante (la clip è $C^0$, i kink sono in $c'$). Due difetti trovati e **corretti**:
- **Difetto 1** (errore di scrittura, `:91` originale): LINK A usava $g(\bar z){=}1$ invece di $C=\mathbb E[g(z)]$ → sovrastima $+39\%$ nella *derivazione scritta* (il codice era corretto). Sanato in §2.3.
- **Difetto 2** (il più importante): l'identificazione $\sigma_v\leftrightarrow$spread-relativize era *asserita, mai testata* → **testata** in §2.5 (simulazione accoppiata, T3 <0.9% con $\tau$ vincolato). Con la forma onesta $\tau^2=s_A^2+s_B^2$.

Verdetto finale della review: *"the single most important thing still NOT proven"* era il Difetto 2 — **ora chiuso** da w6c. Resta aperto solo il limite di diffusione funzionale (G1 residuo).

---

*Fine W6. Script: [`w6a_coancestry_Ne.py`](w6a_coancestry_Ne.py) (~25 s: forma chiusa istantanea + validazione MC del kernel su $N$), [`w6b_alpha_s_bridge.py`](w6b_alpha_s_bridge.py) (~5 s), [`w6c_coupled_identification.py`](w6c_coupled_identification.py) (~10 s), + verificatori indipendenti della review `w6_adversarial_*.py`. Seed 20260709. Ogni numero è prodotto dagli script.*
