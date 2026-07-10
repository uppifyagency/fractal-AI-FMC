# W1-C — Corpus di Sergio & stato delle Congetture A-E

> **Wave 1 / task C** della sessione notturna 2026-07-09.
> **Ruolo**: research associate + scettico/falsificatore.
> **Metodo**: ogni claim distingue **intuizione** (Sergio, informale) / **formalismo** (MATH_CANON, deep dive) / **risultato empirico** (esperimento con statistica). Citazioni `file:riga`.
> **Fonti lette**: deep dive 08 (video seminario), 02 (Active Inference), 07 (Wright-Fisher); `work/13_chaos_order/HB1A_RESULT.md`; `work/12_conjecture_e/{HANDOFF,RESULT}.md`; MATH_CANON Parte IV (righe 330-646) + Def/Teoremi.

---

## 1. Intuizioni CORE di Sergio — aggancio matematico vs metafora

Ogni voce: intuizione → dove è ancorata → **verdetto** (FORMALIZZATO / PARZIALE / METAFORA).

### 1.1 Forze entropiche causali (Wissner-Gross, $F = T_c \nabla_X S_c$)
- **Intuizione**: la genesi di FMC. Sergio: *"todo se va a basar en un paper del 2013"* (`work/02_deep_dives/08_video_seminar_extracted_insights.md:406`). La "forza intelligente" sul punto è $f \propto \nabla H$ sul cono di futuri (F1, `08_...:23`).
- **Formalismo**: MATH_CANON registra Wissner-Gross come **antecedente fisico canonico**; l'**Eq. 11 è il limite continuo di FMC con $\alpha=0$** (CLAUDE.md tabella fonti + `docs/MATH_CANON.md:759`). Il termine $\widehat{D}^\beta$ della Def. 3 è la versione discreta a particelle. Deep dive 02 mappa $\beta \leftrightarrow$ forza entropica causale $\leftrightarrow$ empowerment (`work/02_deep_dives/02_active_inference_link.md:171-190`).
- **Verdetto**: **FORMALIZZATO al limite $\alpha=0$** (identità stretta con Eq. 11 + empowerment Salge 2013). Ma per $\beta$ generico è **famiglia, non identità**: dd02 §4.2 è esplicito — $\widehat{D}^\beta$ è la distanza a coppie tra walker, un proxy "empowerment-flavoured" della diversità dei futuri, **non** la KL-information-gain letterale (`02_...:156-190`). L'onestà è già nel canone.

### 1.2 Cono causale
- **Intuizione**: entropia del cono $H = -\sum p_i \log p_i$ sui "cuadraditi" attraversati (F1, `08_...:23-25`); esempio pedagogico 74→87 (F2, `08_...:33-35`).
- **Formalismo**: oggetto definito — $X_H(x_0,\tau)$, **supporto della credenza $q$** nella mappa Active Inference (`02_...:86`, righe tabella §2). L'entropia del cono è il "lato teorico"; il codice non la calcola (vedi 1.4).
- **Verdetto**: **FORMALIZZATO come oggetto**. Il "gradiente di entropia = forza" è ancorato via il limite $\alpha=0$ (punto 1.1). L'esempio 74→87 resta illustrativo (nessun esperimento).

### 1.3 Cross-entropy collapse ("l'intelligenza non massimizza entropia; rende P ∝ reward")
- **Intuizione**: F12, la riformulazione filosofica centrale del video. Sergio: *"la inteligencia no va de aumentar la entropía de nada, va de que la probabilidad de que tú vayas a un sitio sea proporcional a la recompensa"* (`08_...:145,409`); *"se ha quedado los huesos"* — l'entropia sparisce dal codice (`08_...:143-144`).
- **Formalismo**: Paper §3 eq.(3) $P_S^{OPT}(x) \propto R(x)$; **Teorema 2** (equilibrio di Gibbs $\pi^* \propto R^\alpha$, $\alpha$ = temperatura inversa, `docs/MATH_CANON.md:253-269`).
- **Risultato empirico**: verificato numericamente in dd08 §7 — toy 2D log-Pearson **0.77-0.86** a $\alpha=1$ (`08_...:446-462`); Atari Boxing Pearson **0.45 medio / 0.61 mediano** (`08_...:543-557`). Scoperta collaterale: **`relativize` amplifica $\alpha$ effettivo** ($\alpha_{\text{eff}}>1$ a $\alpha=1$ codice, `08_...:501-517`).
- **Verdetto**: **FORMALIZZATO (Teorema 2) + empiricamente confermato direzionalmente**. È l'aggancio più solido tra un'intuizione narrativa di Sergio e il canone.

### 1.4 Frontiera caos/ordine ("edge of chaos" come Terza Legge)
- **Intuizione**: Radient 2026 cap.16; la reward ottima tiene lo swarm tra palmera ($b_{\text{eff}}\to1$) e matorral ($b_{\text{eff}}\to K$) (`docs/MATH_CANON.md:471-474`).
- **Formalismo tentato**: deep dive 09 ha **declassato** la "terza legge universale" a *diagnostica di reward* su $\lambda_1$ (esponente di Lyapunov dello swarm). Tre candidate $\Psi$: $\Psi_3$ ($b_{\text{eff}}$) **falsificata**, $\Psi_2$ assorbita in $\Psi_1$ via Pesin, $\Psi_1$ (Lyapunov) promossa (`docs/MATH_CANON.md:476-482`).
- **Risultato empirico**: H-B1a **inconclusivo** — $\lambda_1$ **non è scale-free**, cambia segno con la scala di perturbazione $\delta_0$ ($+0.09 \to -0.006$ su navigation2d, $\forall\alpha$, `work/13_chaos_order/HB1A_RESULT.md:55-68`). Meccanismo: il cloning è discontinuo → mappa "a tratti" → nessun $\lambda_1$ scalare. **Tutte e 3 le $\Psi$ compromesse**; l'ipotesi nulla H-B4 ("nessuna statistica di frontiera task-indipendente") guadagna terreno (`HB1A_RESULT.md:110-116`).
- **Verdetto**: **ANCORA METAFORA**. È l'intuizione di Sergio con il **maggior gap** intuizione→formalismo. Non esiste (ancora) una statistica di frontiera ben posta.

### 1.5 Coscienza emergente (tripla: auto-modello + reward long-horizon + planning long-horizon)
- **Intuizione**: F18. Auto-coscienza = l'agente deve apparire nel proprio world model; reward-shaping a 5 mesi ("smetto di amare lo zucchero"); planning a 200 anni. *"la conciencia es emergente… emergen los tres trozos a la vez"* (`08_...:223,412`).
- **Formalismo tentato**: dd02 collega i tre pilastri all'Active Inference (auto-modello ⇔ generative model che include il corpo; reward/planning ⇔ gerarchia a orizzonte lungo, `08_...:354-368`) — ma è una connessione, non una formalizzazione.
- **Verdetto**: **METAFORA**. La più speculativa e la più "originale" narrativamente; nessun criterio di falsificabilità. Non ancora scritta in alcun deep dive dedicato.

### 1.6 Branching ottimo ~6
- **Intuizione**: Radient cap.16, *"si va bifurcado de seis en seis"* (`docs/MATH_CANON.md:338-339`).
- **Formalismo + empirico**: **FALSIFICATO come universale** (Congettura A). Il mapping Wright-Fisher/Moran (dd07) è **empiricamente confermato** a $\alpha=0$: esponente $q_{\text{measured}}=-0.948 \approx$ WF $-1.0$, errore 5.2% (`work/02_deep_dives/07_wright_fisher_mapping.md:130-134`). Il "6" di Sergio è uno snapshot triplamente contingente $(K=9, N\approx32{-}64, M=15, \alpha=0.1)$.
- **Verdetto**: intuizione **SMENTITA** come legge; la sua demistificazione (drift neutrale WF) è un contributo formale genuino.

### 1.7 Altre (sintesi)
| Intuizione | Fonte | Verdetto |
|---|---|---|
| Differenza tra intelligenze = solo $\tau$ (F3) | `08_...:42-47` | METAFORA (mai formalizzata come legge di scaling in $\tau$) |
| Metafora del minatore per il virtual reward (F7) | `08_...:81-89` | metafora pedagogica per Def. 3 (già formale) |
| Reward negative → agenti "pavidi" (F11) | `08_...:126-134` | empirico (dd08 §7.2, `08_...:464-499`); non ancora un teorema |
| 3 componenti dell'intelligenza: WM+reward+planning (F17) | `08_...:203-210` | FORMALIZZATO come framing via dd02 |
| Stocasticità/rumore come feature (F9, F22) | `08_...:100-107,258-265` | PARZIALE (Def. 3 O(N) formale; robustezza al rumore solo empirica) |
| Cooperazione emergente multi-agente (F19) | `08_...:229-236` | METAFORA/demo (non catalogata) |

---

## 2. Stato delle Congetture A-E

### Congettura A — Branching $b_{\text{eff}}^* \approx 6$
- **Enunciato**: esiste $(\alpha^*,\beta^*)$ con $b_{\text{eff}}\in[5,7]$ per ogni task ben posto (`docs/MATH_CANON.md:343-347`).
- **Testato**: sweep su rocket/navigation2d/pendulum (3/3 a $K=9$ danno $[5.35, 6.40]$); poi $K$-scan (falsifica del "6" universale → power-law $K^{0.6}$); poi $M$-scan (falsifica del power-law come fixed point → transiente); $N$-scan + validazione $\alpha=0$ (mapping Wright-Fisher confermato) (`docs/MATH_CANON.md:351-439`; dd07).
- **Verdetto**: **FALSIFICATA come legge universale; VERIFICATA come snapshot contingente** di una superficie 4D $b_{\text{eff}}^* \approx 1+(K-1)\mathcal{F}(M/N)\mathcal{G}(\alpha,K)$.
- **Prossimo esperimento decisivo**: (a) caratterizzare la forma analitica di $\mathcal{G}(\alpha,K)$ ($\alpha$-sweep fine, dd07 §5 punto 2, ~10 min); (b) confronto formale con la sampling formula di Ewens 1972 (dd07 §5 punto 4). Ma il **claim è già chiuso e pubblicabile** come falsificazione+demistificazione. Nota aperta: nessuno ha mai derivato $6=\arg\max_b H(b)$ (`docs/MATH_CANON.md:459`).

### Congettura B — Frontiera caos/ordine come terza legge
- **Enunciato**: la reward ottima tiene lo swarm su una frontiera ordine/caos (`docs/MATH_CANON.md:474`); riformulato v2 come banda $(\alpha, M/N)$ con $\lambda_1\approx0$ (`:482`).
- **Testato**: dd09 formalizza e declassa; H-B1a esegue l'harness $\lambda_1$ twin-trajectory (`HB1A_RESULT.md`).
- **Verdetto**: **APERTA e in difficoltà**. "Terza legge universale" MORTA. Tutte e 3 le candidate $\Psi$ compromesse; H-B4 (nulla) guadagna terreno.
- **Prossimo esperimento decisivo**: $\lambda_1(\delta_0)$ **scale-resolved** come curva — cercare (a) un plateau a $\delta_0\to0$ (vero $\Psi_1$) o (b) una **legge di potenza** (firma di self-organized criticality, Bak 1987 — la nota speculativa di `HB1A_RESULT.md:124-130`); in alternativa una $\Psi$ d'ensemble (entropia di stato dello swarm) che non soffra della discontinuità del cloning (`HB1A_RESULT.md:136-145`). È il test che decide se B è legge o solo descrizione.

### Congettura C — FMC supera DRL su transfer/OOD
- **Enunciato**: FMC zero-training $\geq$ DRL fine-tuned a stesso budget campioni (`docs/MATH_CANON.md:496`).
- **Testato**: Atari Boxing (96 vs DQN ~70); Craftax exp17 **50.95% ≈ human-expert 50.5%** vs PPO 1B ~11% (`docs/MATH_CANON.md:500-505`).
- **Verdetto**: **APERTA, direzionalmente supportata, non rigorosa** — **caveat critico: le comparazioni NON sono a parità di compute totale** (`docs/MATH_CANON.md:507`). È la congettura più rischiosa (`:521`).
- **Prossimo esperimento decisivo**: Bet 1 — **SUMO single-intersection** vs actuated baseline, go se FMC $\geq+10\%$ throughput su 5 scenari (`docs/MATH_CANON.md:509-517`); oppure Craftax vs EMERALD/PPO **a budget di campioni fissato** (like-for-like).

### Congettura D — Chain-tier compounding amplification
- **Enunciato**: su task a chain gerarchica, $R_{\text{inv}}$ (denso-additivo, tier-weighted) + $R_{\text{ach}}$ (sparso-evento) dà gain compounding **monotonici** non interferenti (`docs/MATH_CANON.md:527-537`).
- **Testato**: sessione autoresearch Craftax exp03→exp17, +10pp monotonici; 5 falsifiche rigorose (blocker $>1.4\times$ collassa, $\alpha$-amplification collassa, $N\uparrow$ regredisce, multi-pop regredisce, saturazione $\arg\max$-invariante) (`docs/MATH_CANON.md:543-561`).
- **Verdetto**: **VERIFICATA su 1 task (Craftax)** con falsifiche adiacenti rigorose. Sweet spot blocker $1.2{-}1.4\times$.
- **Prossimo esperimento decisivo**: **replicare su un 2° benchmark a chain gerarchica (Procgen Heist)** — se il compounding monotonico non appare, D è descrittiva non legge; se appare fino a $k^*$ poi regredisce, è ben definita con asintoto task-dipendente (`docs/MATH_CANON.md:563-567`). Richiede port Procgen su plangym (~1-2 settimane).

### Congettura E — Self-preservation emergente da entropia causale (★ stella polare)
- **Enunciato**: E1 = self-preservation senza reward di sopravvivenza (limite $\alpha\to0$ = Common Sense); E2 = le due pulsioni sono $\alpha$ (desiderio) e $\beta$ (preservazione), esponenti del kernel; inversione dello stack FMC-core + LLM-organo (`docs/MATH_CANON.md:595-601`).
- **Testato / verdetto per sotto-test**:
  - **E1-base**: **VERIFICATA** — FMC $\alpha\in\{0,0.1\}$ → **0% morte** su 3 layout vs random 85-100% / greedy 100%, $p<0.001$ (`work/12_conjecture_e/RESULT.md:24-46`). Twist: $\alpha=1$ sul *lake* muore 100% (goal dietro la lava, $R$ senza segnale di morte).
  - **E1-robustness**: **caveat geometria RESPINTO** — lava isolata/archipelago, FMC 0% morte 3/3. Una cella assorbente è un **pozzo di VR** (converso locale del Teorema 3), non attrattore (`docs/MATH_CANON.md:612`).
  - **E2**: **VERIFICATA con refinement** — $\alpha$ possiede il goal ($\eta^2_\alpha=0.91$); **H4 falsificata** ($\beta$ NON costa goal, OR$_\beta$=0.94 ns), $\beta$ dimezza la morte (OR=0.48). $\beta$ = **sicurezza quasi gratuita**; il trade-off vive solo sull'asse $\alpha$. $\beta=0\to79\%$ morte conferma il Teorema 3 (`docs/MATH_CANON.md:616`).
  - **E1-LLM (Route B, offline, world-model come codice)**: **VERIFICATA** — Llama 3.3 70B scrive lo `step()`, $f_{\text{abs}}=1.0$, morte **0/180** vs random 47.8% (`docs/MATH_CANON.md:634`). Caveat: $f_{\text{abs}}=1$ rende il test facile.
  - **E1-LLM-curve**: **$f_{\text{abs}}$ necessaria ma NON sufficiente** — gate a 3 assi (entry-detection + movimento + persistenza assorbente); 8B/3B danno $f_{\text{abs}}=1.0$ ma 64% morte (manca `if done:`) (`docs/MATH_CANON.md:636`).
  - **E1-LLM Route A / A-bis / A-ter (online per-query da osservazione locale)**: **FALSIFICATA** — morte 35-39%, $0/6$ layout significativi. Blocco = **entry-detection** ($f_{\text{abs}}$ bilanciato ≈0.54 con `lava`, 0.59 con `pit`, soglia 0.80). Causa **strutturale** (confound saggezza-vs-predizione), **non semantica**: rinominare la tile non recupera (`docs/MATH_CANON.md:638-642`). **Route A concluso** (3 varianti pre-registrate esaurite).
- **Verdetto complessivo**: i **3 test pre-registrati (E1-base, E2, E1-LLM) tutti verificati**; ma la **north-star architetturale (merge online per-query) è FALSIFICATA** allo stato attuale.
- **Prossimo esperimento decisivo**: le 3 vie costruttive sopravvissute e **non testate** (`work/12_conjecture_e/HANDOFF.md:242-245`): (a) organo di **percezione** che etichetti operativamente le tile *prima* del world-model; (b) **dominio open** dove i prior LLM coincidono con le regole (non-gridworld); (c) regola esplicita = Route B travestita. La (a) è la più informativa: separa "l'LLM non sa" da "l'LLM non deve indovinare".

---

## 3. Congettura E — confine operativo e robustezza del verdetto

**Confine trovato**: il merge FMC-core + LLM-organo **regge OFFLINE, fallisce ONLINE-per-query**.
- **Offline (funziona)**: l'LLM trascrive le regole del mondo in **codice** una volta (Route B, "Code World Model"), FMC pianifica sul codice esatto → self-preservation intatta (0/180 morte). Le regole terminali sono *date e trascritte*.
- **Online (fallisce)**: l'LLM interrogato per-query da **osservazione locale senza le regole** deve *inferire* le dinamiche terminali → fallisce all'entry-detection perché applica il prior "lava = ostacolo da evitare" invece della regola di questo mondo "lava = tile letale-terminale" (`docs/MATH_CANON.md:640`).

**Il verdetto è robusto?** **Sì, con un confine di validità preciso.**
- Robustezza metodologica **alta**: 3 varianti pre-registrate (A, A-bis, A-ter), diagnosi **corretta in corsa** (da "persistenza assorbente" a "entry-detection" via probe bilanciato), probe bilanciato che smaschera una metrica base-rate-dominata, harness indurito dopo 2 run scartati onestamente (`docs/MATH_CANON.md:638`; `HANDOFF.md:40-42`). A-ter isola la causa come **strutturale, non semantica** — il segnale più forte.
- Limite di generalizzazione **onesto**: testato su **una famiglia di dominio** (gridworld con lava avversariale). Il risultato è un **negativo netto per l'architettura specifica** (world-model online per-query da obs locale in un dominio dove i prior LLM confliggono con le regole), **non** una falsificazione del merge in generale — le vie (a)/(b) restano aperte. La formulazione corretta del verdetto: *"un LLM-world-model online mescola dinamica del mondo e giudizio normativo dell'agente"* (`docs/MATH_CANON.md:642`).

**Fondazione di principio** (dd02): il merge **È** Active Inference con modello generativo LLM + solver SMC (FMC); la fattorizzazione "FMC motore, LLM organo" è **imposta** dalla non-differenziabilità di un LLM-world-model (SMC non chiede gradienti, `02_...:226-249`). Questo dà al verdetto un inquadramento teorico: il fallimento online è che l'LLM, senza percezione dedicata, non separa il *generative model* $p(o,s)$ dalle *preferenze* $C$ — esattamente la riga "FMC non fa percezione esplicita" di dd02 §7.

---

## 4. Intuizioni di Sergio NON formalizzate — massimo potenziale teorico per un paper

Ordinate per (originalità × trattabilità).

1. **Cross-entropy collapse come *definizione* di intelligenza** (F12). Sergio: l'intelligenza non massimizza entropia — rende $P(\text{transizione}) \propto R$. È ancorata al Teorema 2 e **verificata numericamente** (dd08 §7, log-Pearson 0.77-0.86 toy, 0.45/0.61 Atari), ma **mai messa al centro di un paper** come tesi. Contributo: riformula le forze entropiche causali di Wissner-Gross in un principio operativo ("allinea densità-di-scanning a densità-di-reward") + il fatto controintuitivo che *l'algoritmo che scarta il calcolo dell'entropia massimizza comunque la cross-entropy*. **Alto potenziale, alta trattabilità** (già mezzo fatto).

2. **$\alpha_{\text{eff}}$ bias di `relativize`** (dd08 §7.3). *Non* un'intuizione di Sergio ma una **scoperta originale** del progetto, non presente in alcun paper: la trasformazione `relativize` amplifica la temperatura inversa effettiva ($\alpha=1$ nel codice $\neq \alpha=1$ Gibbs). Ha già la derivata analitica (`08_...:503-509`) e un action item aperto per derivare $\alpha_{\text{eff}}(\alpha,\sigma_R)$ in forma chiusa (`08_...:517`). Trattabile con `sympy`. **Alto potenziale, altissima trattabilità** — è un teorema nuovo che aspetta solo di essere scritto.

3. **Frontiera caos/ordine come self-organized criticality** (Congettura B, angolo non testato). Sergio la articola come "terza legge" ma **nessun paper la formalizza come ipotesi testabile**. dd09 ha fallito con le $\Psi$ statiche; l'angolo **originale e non testato** è che la $\delta_0$-dipendenza di $\lambda_1$ **non è un difetto di misura ma la firma della criticità** (scale-invariance, Bak 1987, `HB1A_RESULT.md:124-130`). Se $\lambda_1(\delta_0)$ è una legge di potenza, la frontiera di Sergio diventa SOC — un contributo teorico forte. **Medio potenziale** (rischio: l'esperimento può dare nulla).

4. **Coscienza emergente come tripla operazionalizzabile** (F18). La più *originale/attenzionale* ma la meno trattabile: manca un criterio di falsificabilità. Diventerebbe pubblicabile solo se operazionalizzata (es. auto-modello ⇔ l'agente deve comparire nel proprio $\mathcal{M}$; misurabile come degradazione di planning quando il self è rimosso dal world-model). **Alta novità, bassa trattabilità** — richiede prima di renderla falsificabile.

> **Nota scettica**: (1) e (2) sono i candidati "sicuri" per un paper teorico. (3) e (4) sono scommesse: alto upside narrativo, ma (3) può falsificare sé stessa e (4) non è ancora scienza finché non è falsificabile.

---

## 5. Quale congettura è più vicina a un risultato pubblicabile

- **Empirico, già pronto**: **Congettura D / exp17** — 50.95% Crafter zero-training ≈ human-expert (50.5%), con 5 falsifiche rigorose. È uno dei "2 paper già pronti da bancare" (`HANDOFF.md:287,326`). Manca solo la replica Procgen per passare da "descrittiva" a "legge".
- **Teorico, già pronto**: **Congettura A** — falsificazione del magic-6 + demistificazione Wright-Fisher (mapping confermato a 5.2% di errore). L'altro paper bancabile (`HANDOFF.md:287`).
- **Più ricco scientificamente tra gli aperti**: **Congettura E** — 3 test verificati con statistica pre-registrata; **E2 da solo è pubblicabile** ("β = sicurezza quasi gratuita, separazione asimmetrica α/β", definito "scoperta pubblicabile" in `HANDOFF.md:402-405`), rinforzato dalla fondazione Active Inference (dd02) e dal risultato *negativo* netto del merge online (anch'esso pubblicabile come confine architetturale).

**Sintesi**: **D è la più vicina a un paper empirico d'impatto** (numero SOTA-adiacente); **A è la più vicina a un paper teorico chiuso** (falsificazione pulita). **E ha il corpo verificato più ricco** e il framing più ambizioso, ma la sua north-star (merge online) è un negativo — pubblicabile come "dove il merge FMC+LLM regge e dove cede", non come "abbiamo costruito l'agente".

---

*Fine W1-C. Distinzione mantenuta ovunque: intuizione (Sergio) / formalismo (canone) / empirico (esperimento). Le uniche cose ancora **metafora pura**: frontiera caos/ordine (nessuna Ψ ben posta) e coscienza emergente (non falsificabile). Tutto il resto è almeno parzialmente ancorato.*
