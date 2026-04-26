# Fractal Coding Loop: visione

> *"Il Ralph loop è una linea retta nel tempo. Il Fractal Coding Loop è uno sciame di linee che si confrontano. Ogni iterazione è un cono causale di possibilità, non una freccia."*

Documento visionario, scritto **insieme** all'utente. Immagina concretamente come tradurre il principio FMC + Fractal Memory in un'infrastruttura di self-programming basata su Claude Code (file, sub-agent, hooks, MCP).

---

## 0. Il problema con il Ralph loop

Il Ralph loop, popolarizzato da Geoffrey Huntley (2024), è la prima approssimazione di un agente di coding autonomo:

```
while true; do
  cat PROMPT.md | claude --dangerously-skip-permissions
done
```

Funziona perché Claude è bravo, ma ha **un vizio strutturale**: ogni iterazione è una **freccia singola nel tempo**. Niente confronto tra alternative, niente evaluation di "stiamo andando nella direzione giusta?", niente meccanismo di rollback intelligente.

**Carvanicoli**? Sì. Funziona? Anche. Bello? No.

Visualmente, il Ralph loop:

```
     RALPH LOOP — single trajectory
     ──────────────────────────────────────────────────────►  time

     ●───────●───────●───────●───────●───────●───────●     ...
     │       │       │       │       │       │       │
   prompt  code    prompt  code    prompt  code    prompt

     Quality = funzione del prompt + variance random
     Introspection = nessuna
     Rollback = manuale (git reset)
     Cost = O(N) iterazioni * cost(claude_call)
```

---

## 1. Il salto: Fractal Coding Loop

L'idea del Fractal Coding Loop (FCL) è applicare FMC alla decisione "qual è la prossima azione di coding?".

Anziché una freccia, un **cono**:

```
     FRACTAL CODING LOOP — swarm of trajectories per decision
     ─────────────────────────────────────────────────────►  time

     t=0                                                   t=N
       state x₀
         │
         │     ┌─── DECISION POINT ───┐
         │     │                       │
         │     │  spawn N walkers      │
         │     │  (sub-agents)         │
         │     │                       │
         │     │   ●₁  ●₂  ●₃  ...  ●ₙ │
         │     │  /│\ /│\ /│\      /│\ │
         │     │  ●●● ●●● ●●●      ●●● │  M-tick fractal cone
         │     │ /│\\///\\\///\\\\  /│\ │  of context
         │     │●●●● ●●● ●●●● ●●●●  ●●●│
         │     │ ↓    ↓   ↓    ↓     ↓ │
         │     │ evaluate leaves       │
         │     │ R · Dist → VR         │
         │     │ cloning step          │
         │     │                       │
         │     │ winner: walker_i      │
         │     │ initial_action: A     │
         │     └───────┬───────────────┘
         │             │
         │             ↓
         │        execute A in real codebase
         │
         ↓
       state x₁ (= x₀ + A applied)
         │
       (repeat: new fractal cone from x₁)
         ↓
        ...

     Quality = funzione di N walkers × M depth × reward signal
     Introspection = built-in (walker convergence = confidence)
     Rollback = automatic (loser walkers contain "what NOT to do" signal)
     Cost = O(N · M · cost) per decision; trade-off explicit
```

**Il punto-chiave**: ogni decisione vera nel codebase è preceduta da uno **sciame esplorativo nel "futuro virtuale"**, dove i walker sono sub-agent Claude che lavorano su branch git temporanei.

---

## 2. Il Fractal Cone of Context

Il concetto-chiave (introdotto dall'utente) è il **fractal cone of context**.

Definizione: il context window di ogni walker non è una copia identica del contesto del main agent. È un **fork condizionato** alla decisione iniziale candidate.

```
FRACTAL CONE OF CONTEXT
─────────────────────────

main agent context window:
┌────────────────────────────────────────────────┐
│ ⟨codebase, goal, history, recent_edits, todo⟩ │
└────────┬───────────────────────────────────────┘
         │
         │ FORK into N variants
         │
    ┌────┴────┬────────┬────────┬─────────┬───────┐
    ↓         ↓        ↓        ↓         ↓       ↓
  ┌────┐   ┌────┐   ┌────┐   ┌────┐    ┌────┐  ┌────┐
  │ x₀ │   │ x₀ │   │ x₀ │   │ x₀ │    │ x₀ │  │ x₀ │
  │ +  │   │ +  │   │ +  │   │ +  │    │ +  │  │ +  │
  │ A₁ │   │ A₂ │   │ A₃ │   │ A₄ │    │ A₅ │  │ Aₙ │
  └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘    └─┬──┘  └─┬──┘
    │        │        │        │         │       │
   walker₁ walker₂ walker₃ walker₄    walker₅  walkerₙ

   Each walker explores a CONDITIONAL FUTURE:
   "what would happen if I started by doing A_i?"

   Recursively, each walker can spawn its own M sub-walkers:
   ┌────┐
   │ x₀ │
   │ +  │
   │ A₂ │
   └─┬──┘
     │
   ┌──┴──┬──────┬──────┐
   ↓     ↓      ↓      ↓
  walker₂.₁ ... walker₂.ₘ        ← M-deep fractal recursion

   This is the "fractal" part: the cone is self-similar at every depth.
   Each sub-walker has the same structural mechanism.
```

**Implicazione computazionale**: per default, $N = 5, M = 3$ → $5^3 = 125$ context-windows attivi simultaneamente. Costo significativo. Ma il principio FMC è che **non tutti i walker sopravvivono** — il cloning step elimina rapidamente quelli che vanno male, ricongelano la popolazione su quelli buoni.

---

## 3. Self-programming: la chiusura dell'anello

Il vero salto dal Ralph loop al Fractal Coding Loop è la **chiusura dell'anello su sé stesso**: il sistema non ottimizza solo "il codice del progetto", ma anche **il proprio reward function**, **la propria distance metric**, e **la propria scelta di sub-agent**.

```
SELF-PROGRAMMING FRACTAL LOOP — la struttura completa
══════════════════════════════════════════════════════

     ┌──────────────────────────────────────────────┐
     │                                              │
     │   META-LEVEL (Book #2 §3.4: Outer Loop)      │
     │   ───────────────────────────────────────    │
     │   Reward function:                           │
     │     R(state) = combo of:                     │
     │       • tests pass?                          │
     │       • lints clean?                         │
     │       • diff size sensata?                   │
     │       • user goal closer?                    │
     │       • user feedback signal?                │
     │       • coherence with project conventions?  │
     │                                              │
     │   Distance metric:                           │
     │     d(state₁, state₂) =                      │
     │       AST diff + file diff + semantic emb    │
     │                                              │
     │   Architecture:                              │
     │     N walkers, M depth, N_iter cycles        │
     │     these are PARAMETERS that can also be    │
     │     optimized via Badger structure!          │
     │                                              │
     └──────────────────┬───────────────────────────┘
                        │
                        │  governs
                        ↓
     ┌──────────────────────────────────────────────┐
     │                                              │
     │   COMPUTE-LEVEL (Book #2 §3.4.1: FMC Loop)   │
     │   ───────────────────────────────────────    │
     │                                              │
     │   for each decision:                         │
     │       state x₀ = current_codebase            │
     │       walkers = spawn_sub_agents(N, x₀)      │
     │       for tick in range(M):                  │
     │           each walker → tool_use + edit      │
     │           snapshot state to git stash        │
     │           reward = R(walker.state)           │
     │           distance = d(walker, walker_j)     │
     │           VR = relativize(R) ⋅ relativize(D) │
     │           cloning step                       │
     │       chosen = argmax(initial_action_count)  │
     │       apply chosen to real codebase          │
     │                                              │
     └──────────────────┬───────────────────────────┘
                        │
                        │  feeds back
                        ↓
     ┌──────────────────────────────────────────────┐
     │                                              │
     │   FEEDBACK LEVEL (= Reward learning)         │
     │   ───────────────────────────────────────    │
     │   user says: "no, that wasn't right"         │
     │     → R is updated (this state has -loss)    │
     │   user says: "yes, perfect"                  │
     │     → R is updated (this state has +loss)    │
     │   tests fail unexpectedly                    │
     │     → R is updated (path to here is bad)     │
     │                                              │
     │   The R itself becomes a learned artifact    │
     │   stored in .fractal_memory/ — git-tracked.  │
     │                                              │
     └──────────────────────────────────────────────┘

     This is the "Self-Programming" closure: the system not only
     decides actions, it improves its own decision-making by
     watching what works in this specific user/project.
```

---

## 4. Implementazione concreta su Claude Code

Mappiamo i concetti astratti sui meccanismi reali di Claude Code:

### 4.1 Walker = Sub-Agent

Ogni walker è un **sub-agent** lanciato via il tool `Agent`:

```
main agent: Claude
  │
  ├─ Agent(subagent_type=general-purpose, prompt="walker_1: try approach A")
  ├─ Agent(subagent_type=general-purpose, prompt="walker_2: try approach B")
  ├─ Agent(subagent_type=general-purpose, prompt="walker_3: try approach C")
  └─ ... (parallel via single message with multiple Agent calls)
```

Ogni sub-agent lavora in un **worktree git separato** (parametro `isolation: "worktree"`) per evitare conflitti.

### 4.2 State snapshot = git stash/branch

Lo "stato del walker" al tempo $t$ = snapshot del worktree:

```
walker_1.state(t=0) → git branch fractal_walker_1_t0
walker_1.state(t=1) → git branch fractal_walker_1_t1
walker_1.state(t=2) → git branch fractal_walker_1_t2
...

Cloning walker_1 → walker_2 means:
  git checkout walker_2_branch → checkout walker_1_state branch
  (effectively: walker_2 inherits walker_1's progress)
```

### 4.3 Reward = composite signal

Il reward $R$ è calcolato in modo composito:

```
def R(walker_state) -> float:
    score = 1.0

    # Hard constraints (multiplicative — zero if violated)
    score *= 0.0 if not tests_pass(walker_state) else 1.0
    score *= 0.0 if has_syntax_errors(walker_state) else 1.0
    score *= 0.0 if has_obvious_security_issues(walker_state) else 1.0

    # Soft constraints (additive bonus)
    score += 0.3 * lint_score(walker_state)
    score += 0.2 * type_check_score(walker_state)
    score += 0.5 * goal_alignment_score(walker_state)  # ← LLM judge
    score += 0.2 * (1 - diff_size(walker_state) / max_diff)

    # User-signal components (when available)
    if user_feedback_exists(walker_state):
        score += 1.0 * user_signal(walker_state)

    return score
```

### 4.4 Distance = embedding + structural

```
def d(state_1, state_2) -> float:
    # Structural: AST diff
    d_ast = ast_distance(state_1.changed_files, state_2.changed_files)

    # Semantic: embedding del diff
    d_sem = embedding_distance(
        embed(diff(x_0, state_1)),
        embed(diff(x_0, state_2))
    )

    # File-level: simple file overlap
    d_files = jaccard(state_1.changed_files, state_2.changed_files)

    return 0.5*d_ast + 0.3*d_sem + 0.2*(1 - d_files)
```

### 4.5 Hooks per orchestrazione

Tre hook fondamentali in `.claude/settings.json`:

```
UserPromptSubmit:
  - inietta memoria fractal pertinente al prompt
  - decide se attivare fractal mode (se prompt complesso) o ralph mode

Stop:
  - salva la traiettoria completa nella .fractal_memory/
  - aggiorna i reward dei state visitati
  - esegue cloning step sul memory bank

PreToolUse(Edit/Write):
  - se in fractal mode, intercetta e propaga ai sub-agent
  - serializza lo stato pre-edit per rollback
```

### 4.6 MCP server per coordinazione

Un MCP server custom (`fractal-coordinator`) espone tool come:

```
spawn_walker(initial_action) → walker_id
get_walker_state(walker_id) → state_snapshot
calculate_virtual_reward(walker_id, partner_id) → VR
clone_walker(walker_id, target_id) → success
sync_walker_to_main(walker_id) → applies winning trajectory

# memoria
save_memory(state, reward, label) → memory_id
sample_memory(query) → relevant memories with walker counts
update_memory_walkers() → cloning step on memory bank
```

---

## 5. La struttura del filesystem

L'idea di un "Fractal Memory" memorizzato sul filesystem:

```
project/
├── .fractal/
│   ├── memory/                       ← Fractal Memory (Slide doc)
│   │   ├── 0001_python_decorators.md   {visits: 12, reward: 0.85}
│   │   ├── 0002_fastapi_pattern.md     {visits: 3,  reward: 0.42}
│   │   ├── 0003_test_strategy.md       {visits: 7,  reward: 0.91}
│   │   └── INDEX.json                ← walkers map: memory_id → walker_count
│   │
│   ├── reward/                       ← Reward learning artifact
│   │   ├── R_function.py             ← evolves over time
│   │   ├── feedback_log.jsonl        ← user signals
│   │   └── lineage.md                ← changelog of R changes
│   │
│   ├── walkers/                      ← Active walker state (FCL session)
│   │   ├── current_session_id.txt
│   │   ├── walker_1.json             ← {git_branch, init_action, reward, state}
│   │   ├── walker_2.json
│   │   └── ...
│   │
│   ├── lineage/                      ← Fractal cones graveyard
│   │   ├── 2026-04-26_18:50_session/
│   │   │   ├── decision_001/
│   │   │   │   ├── walkers.json
│   │   │   │   ├── chosen.json
│   │   │   │   └── losers/           ← walkers that died (= valuable signal)
│   │   │   └── decision_002/
│   │   └── ...
│   │
│   └── config.yaml                   ← N walkers, M depth, balance, etc.
│
├── .claude/
│   ├── settings.json                 ← hooks for fractal orchestration
│   └── agents/                       ← custom sub-agent definitions
│       ├── walker_explorer.md
│       ├── reward_judge.md
│       └── distance_evaluator.md
│
└── CLAUDE.md                         ← project guidelines (already exists)
```

---

## 6. Modalità operative

Il sistema può operare in tre modalità, switchabili via flag o euristiche:

```
══════════════════════════════════════════════════════════════════
RALPH MODE (default — basso costo)
═══════════════════════════════════
  Decision: 1 sub-agent, M=1 (single trajectory)
  Cost: 1× claude call per decision
  Use case: quick fixes, well-defined tasks

══════════════════════════════════════════════════════════════════
FRACTAL LITE (medium — bilanciato)
═══════════════════════════════════
  Decision: N=3 sub-agents, M=2 (mini swarm)
  Cost: ~6× claude calls per decision
  Use case: refactoring, ambiguous fixes

══════════════════════════════════════════════════════════════════
FRACTAL FULL (high cost — high quality)
═══════════════════════════════════════
  Decision: N=5 sub-agents, M=3-5 (full swarm)
  Cost: ~15-25× claude calls per decision
  Use case: architectural decisions, hard refactors,
            new feature design, debug complex bugs

══════════════════════════════════════════════════════════════════
```

L'utente decide via:
- Slash command (`/fractal`, `/fractal-lite`, `/ralph`)
- Auto-heuristica basata sulla complessità del prompt
- Modalità progressiva: parte ralph, escala a fractal se la prima soluzione fallisce

---

## 7. La promessa: "stiamo andando nella direzione giusta?"

Il punto-chiave dell'utente:

> *"Voglio usare questo progetto per creare un'infrastruttura FMC e fractal per il coding, che meglio riesca a comprendere se la scelta che sta prendendo come next iteration è nella giusta direzione o meno."*

**Come il Fractal Coding Loop risponde a questa domanda**:

Dopo $M$ tick di fractal cone, hai tre indicatori espliciti:

1. **Walker convergence ratio**: percentuale di walker che hanno scelto l'azione vincente come $\text{initial\_action}$.
   - Convergenza alta (>70%) → la decisione è chiara, alta confidenza
   - Convergenza bassa (<30%) → c'è ambiguità, la decisione è rischiosa

2. **Reward variance**: deviazione standard del reward tra walker survivor.
   - Bassa varianza → tutti i path buoni convergono allo stesso reward → robusto
   - Alta varianza → il reward dipende da dettagli sottili → fragile

3. **Distance spread**: quanto sono diverse le posizioni finali dei walker.
   - Spread basso → l'azione porta a uno stato definito → committed
   - Spread alto → l'azione apre molte possibilità → tieni ottoni aperti (Common Sense)

```
            ┌────────────────────────────────────────┐
            │   DECISION CONFIDENCE INDICATOR        │
            │                                        │
            │   walker_convergence:  ████████░░ 80%  │
            │   reward_variance:     ██░░░░░░░░ 18%  │
            │   distance_spread:     █████░░░░░ 50%  │
            │                                        │
            │   ✓ HIGH CONFIDENCE                    │
            │   Action: refactor auth middleware     │
            │                                        │
            │   Alternative considered (#2):         │
            │     "add tests first" — 12% support    │
            │   Alternative considered (#3):         │
            │     "split into 2 PRs"   — 5% support  │
            └────────────────────────────────────────┘
```

Questo è qualitativamente diverso dal Ralph loop, dove **non hai nessuna idea di quanto sei sicuro**.

---

## 8. Visualizzazione interattiva (immaginata)

L'utente potrebbe vedere live in un dashboard:

```
╔══════════════════════════════════════════════════════════════════════╗
║  FRACTAL CODING LOOP — live decision visualization                   ║
║──────────────────────────────────────────────────────────────────────║
║                                                                      ║
║  Goal: "Add auth middleware with JWT"                                ║
║                                                                      ║
║  Decision tree (cone of contexts):                                   ║
║                                                                      ║
║                          ●  state_0 (codebase)                       ║
║                         /│\                                          ║
║                        / │ \                                         ║
║                       /  │  \                                        ║
║              walker_1●   │   ●walker_3                               ║
║              "create  ●     /│\ "split into                          ║
║              middleware/│\ ● ● ●  middleware                         ║
║              first"   ● ● ●         + tests"                         ║
║                       │ │ │                                          ║
║                      [✓][✗][✓]    leaf evaluation                    ║
║                       │   ↓   │                                      ║
║                       │ killed │     walker_2 cloned to walker_1     ║
║                       │   │    │                                     ║
║                       reward profile:                                ║
║                       walker_1: 0.85  ← winner                       ║
║                       walker_2: 0.41                                 ║
║                       walker_3: 0.78                                 ║
║                                                                      ║
║  Convergence: 67% (walker_1 + walker_3 ≈ same approach)              ║
║  Variance:    18% reward std                                         ║
║                                                                      ║
║  → Executing winning action: "create middleware first"               ║
║  → Memory updated: 1 new entry                                       ║
║  → Reward function: +0.02 weight on "test_first" detection           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

Tecnicamente realizzabile via TUI (`rich`/`textual` Python) o panel browser-based.

---

## 9. Roadmap di implementazione

In ordine di rischio crescente:

### Phase 0 — Manual fractal (settimane 1-2)
- Implementare comandi slash (`/fractal-decide`, `/fractal-evaluate`)
- L'utente trigger esplicitamente la modalità fractal
- Solo N=3, M=1 (ultra-light)
- **Output**: feeling se l'idea funziona

### Phase 1 — Auto-orchestration (mesi 1-2)
- Hook `UserPromptSubmit` con auto-detect di "complex prompt"
- Sub-agent worktree-isolated per ciascun walker
- Reward composta hardcoded (tests + lints + diff size)
- Distance metric semplice (file-level Jaccard)
- **Output**: paragone empirico Ralph vs Fractal Lite su task reali

### Phase 2 — Fractal Memory (mesi 2-3)
- `.fractal/memory/` come Fractal Memory dataset
- Hook `Stop` che aggiorna walker counts, fa cloning step
- Sample memory all'init di ogni nuova conversazione
- **Output**: catastrophic-forgetting test (insegnamento di una convention, poi switch task, poi ritorno)

### Phase 3 — Self-improving reward (mesi 3-4)
- Reward function come artifact aggiornabile
- User feedback signal hook
- Lineage tracking (changelog del reward)
- **Output**: A/B test fractal-no-feedback vs fractal-with-feedback

### Phase 4 — Multi-NN Fractal Memory (mesi 4-6)
- 3-5 sub-agent specializzati come "esperti" (debug, refactor, test, doc, review)
- Ognuno con il proprio walker count
- Routing automatico via FMC
- **Output**: la Mixture-of-Experts non-supervisionata del Slide doc

### Phase 5 — Vision dashboard (mesi 6+)
- TUI o panel-based dashboard
- Visualizzazione fractal cone in real-time
- **Output**: la versione "Iron Man / Tony Stark" del workflow

---

## 10. Critiche oneste a questa visione

Prima che l'entusiasmo prenda piede:

1. **Costo token**: N=5, M=3 = 15× il costo di Ralph. Per ogni decisione. Bisogna quantificare se il payoff vale.

2. **Complessità infrastrutturale**: hooks, MCP, sub-agent, worktree git, file FM — tutto deve cooperare. Una catena lunga è una catena fragile.

3. **Reward signal noise**: Claude come judge ha varianza alta. Il reward su "goal alignment" è inerentemente rumoroso.

4. **Valutazione assente**: Ralph è valutabile (codice gira / non gira). Fractal richiede metriche più sottili. Senza queste, non sai se sta migliorando.

5. **Premature abstraction**: il Book #2 stesso è draft V0.2. Stiamo costruendo su fondazioni teoriche non completamente validate.

6. **User cognitive load**: 3 modalità (ralph / lite / full) + dashboard + feedback signal = nuove abitudini da imparare.

---

## 11. Il vero test

> *"Il sistema sa se sta andando nella direzione giusta?"*

Test concreto: prendere un bug reale, già risolto da un umano (con git commit history). Lanciare Fractal Coding Loop su quel bug. Verificare:

a) La direzione vincente del walker swarm coincide con la direzione del fix umano?

b) La walker convergence ratio è alta quando il fix è ovvio, bassa quando il fix è ambiguo?

c) Il reward signal cresce monotonicamente lungo la traiettoria di soluzione, scende per traiettorie sbagliate?

Se a, b, c sono "sì", il sistema ha **introspection genuina**. Se no, è marketing.

---

## 12. Pensieri finali (un po' visionari)

Questa visione è ambiziosa. Forse troppo. Ma c'è un motivo per provarla:

> *Il Ralph loop dimostra che la pura iterazione su Claude funziona meglio di quanto chiunque si aspettasse. Ma ha un soffitto: non capisce quando sta sbagliando.*
>
> *Il Fractal Coding Loop introduce **introspection strutturale** via il principio Hernández-Cerezo: l'intelligenza è equilibrio tra esplorazione e sfruttamento, e l'introspection emerge dal **pluralismo** di walker che si confrontano.*
>
> *Se funziona, abbiamo qualcosa di nuovo nel coding agent design. Se non funziona, abbiamo capito empiricamente perché.*

In entrambi i casi, vale la pena provarci.

---

*Documento visionario. Scritto in dialogo con l'utente. Non è una specifica tecnica chiusa — è il punto di partenza per una specifica tecnica chiusa.*

*Per riferimenti accademici vedi [`docs/bibliography/CORPUS.md`](../bibliography/CORPUS.md). Per la teoria vedi [`work/02_deep_dives/06_book2_badger_fractal_memory.md`](../../work/02_deep_dives/06_book2_badger_fractal_memory.md).*

---

# Aggiornamento (2026-04-26) — dopo l'archiviazione bibliografica

> *Tre lezioni dalla bibliografia 2014-2021 cambiano il design del Fractal Coding Loop. Documento la differenza per chi tornerà a costruire.*

## V1. Reward composita: moltiplicativa, non additiva

### Cosa avevo proposto

Nel design originale (§4.3 sopra) avevo scritto:

```python
def R(walker_state) -> float:
    score = 1.0
    score *= 0.0 if not tests_pass(walker_state) else 1.0
    score *= 0.0 if has_syntax_errors(walker_state) else 1.0
    # Soft constraints (additive bonus)
    score += 0.3 * lint_score(walker_state)
    score += 0.2 * type_check_score(walker_state)
    score += 0.5 * goal_alignment_score(walker_state)
    ...
```

**Errore**: mescola hard constraints moltiplicativi (`× 0`) con soft constraints additivi (`+= w_i * x_i`). Questo è il pattern Pareto-style che Sergio ha **esplicitamente rigettato** nel post 2016 [`pareto_frontiers.md`](../bibliography/sources/blog_posts/2016-04_pareto_frontiers.md):

> *"Real-world problems typically have single underlying objectives. We only have one goal in life — maximizing long-term well-being."*

E che il paper 2018 §2.2.2 formalizza così:

$$
R(s) = R_0(s) \times R_1(s) \times \ldots \times R_n(s)
$$

### Cosa proponiamo ora (V1)

Reward composita **completamente moltiplicativa**, con `relativize` per gestire scale arbitrarie:

```python
def R(walker_state) -> float:
    # Componenti (ognuna in [0, max] invece di [0,1] o ℝ)
    R_alive       = 1.0 if not has_syntax_errors(walker_state) else 0.0
    R_tests       = test_pass_ratio(walker_state)        # ∈ [0, 1]
    R_lint        = lint_score_normalized(walker_state)  # ∈ [0, 1]
    R_types       = type_check_score(walker_state)       # ∈ [0, 1]
    R_diff        = max(0, 1 - diff_size / max_acceptable)
    R_goal        = llm_judge_alignment(walker_state)    # ∈ [0, 1]
    R_user        = user_signal(walker_state)            # ∈ [0, 1] if available, else 1.0

    # Composizione moltiplicativa: morto-in-uno → morto-globale
    return R_alive * R_tests * (1 + R_lint) * (1 + R_types) * \
           (1 + R_diff) * (1 + R_goal) * (1 + R_user)
```

**Pattern**: hard constraint = moltiplicazione diretta (può andare a 0). Soft contribution = `(1 + x)` per evitare zero-collapse e mantenere la composizione sempre attiva.

Questo è coerente sia col paper 2018 che col post 2016. È più elegante e più robusto.

## V2. La gerarchia "fingers → hand → octopus" è il riferimento concettuale

### Cosa avevo nascosto

Il post [2015-12 *Fractal AI Collaboration*](../bibliography/sources/blog_posts/2015-12_fractal_ai_collaboration.md) descrive testualmente:

> *"a fractal tree of intelligence layers where fingers coordinate as a hand, hands function as an octopus, and multiple octopuses respond to collective instructions like retrieving objects."*

**Questa metafora è il prototipo del Badger Structure formale del 2020.**

### Mappatura precisa per il Fractal Coding Loop

| Livello biologico | Livello tecnico Claude Code | Esempio concreto |
|---|---|---|
| **Finger** (walker) | Sub-agent singolo | Un sub-agent che propone "rinomina la funzione X" |
| **Hand** (expert level) | Cluster di sub-agent dedicato a un task atomico | 5 sub-agent che esplorano "come refactor auth middleware" |
| **Octopus** (loop level) | Sessione completa con multiple hand | Una sessione di sviluppo che coordina refactor + tests + docs |
| **Multiple octopuses** | Collaborazione tra sessioni Claude (cross-instance) | Team di sessioni che lavorano su feature diverse coordinate |

Implicazione: il design **non è 1 livello ma 3-4** anche per il caso single-machine.

### ASCII: la gerarchia rivelata

```
                              MULTIPLE OCTOPUSES
                              (cross-instance pool)
                                     │
                       ┌─────────────┼─────────────┐
                       ↓             ↓             ↓
                    OCTOPUS-1     OCTOPUS-2     OCTOPUS-3
                    (sessione)    (sessione)    (sessione)
                       │             │             │
                  ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
                  ↓         ↓   ↓         ↓   ↓         ↓
                HAND      HAND  HAND   HAND  HAND   HAND
                (refactor) (test) (docs) (debug) (review) (deploy)
                  │         │     │      │      │       │
              ┌───┼───┐   ┌─┴─┐ ┌─┴─┐  ┌─┴─┐  ┌─┴─┐   ┌─┴─┐
              ↓   ↓   ↓   ↓   ↓ ↓   ↓  ↓   ↓  ↓   ↓   ↓   ↓
            FINGER FINGER ... walkers (sub-agent N=5 per HAND)
            (sub-agent)
```

**Phase 0** (immediato): solo 1 OCTOPUS con 1 HAND con N=3-5 FINGER.
**Phase 4-5** (futuro): full hierarchy.

## V3. Wigner reward per la Fractal Memory di codice

### Cosa avevo dato per scontato

Il design proponeva una memory bank in `.fractal/memory/` con reward generico. Non avevo specificato **come pesare** gli esempi di codice durante il batching/sampling.

### Cosa proponiamo ora (V3)

Il [Slide doc 2020](../bibliography/sources/books/2020_fractal_memory_slides.md) propone esplicitamente:

$$
R'(x) = \frac{\pi}{2} x \exp\left(-\frac{\pi}{4} x^2\right) \quad \text{con } x = \frac{\text{loss}_i}{\text{avg loss}}
$$

Per la Fractal Coding Memory questo si traduce così:

```python
def memory_weight(memory_entry, fractal_memory_bank):
    # "Loss" surrogato = quanto male il sistema ha fatto su questo example
    loss = memory_entry.failure_rate  # ∈ [0, 1]: fraction of times this was wrong
    avg_loss = fractal_memory_bank.average_failure_rate
    if avg_loss == 0: return 1.0  # niente fallimenti, niente pesi
    x = loss / avg_loss
    # Wigner reward: peso massimo per memorie a difficoltà media
    R = (Math.PI/2) * x * Math.exp(-Math.PI/4 * x*x)
    # Penalize over-visited (already learned)
    return R / (1 + Math.log(1 + memory_entry.visit_count))
```

**Conseguenza**: quando Claude attinge dalla memory bank per generare un nuovo prompt, le memorie vengono campionate **non uniformemente**:

- Memorie a difficoltà bassa (già imparate) → poco peso
- Memorie a difficoltà media → peso massimo (ottimal curriculum)
- Memorie a difficoltà alta (troppo difficili) → poco peso (skip until ready)

Risultato dichiarato: **curriculum learning automatico** + **niente catastrophic forgetting**.

### Diagramma ASCII della distribuzione di memoria

```
peso ↑
     |
     |       Wigner curve
     |        ╱╲
   1.0       ╱  ╲          ← peak around x=1 (loss = avg)
     |     ╱      ╲
     |   ╱          ╲___
   0.5 ╱                ‾─
     |╱                   ‾─___
     └────┬────┬────┬────┬────→ loss / avg_loss
          0    1    2    3    4
               ↑
        zona ideale per imparare
```

Questo è il pattern di selezione che dovrebbe regolare l'attenzione del sistema sulla memory.

## V4. Il design risultante (versione 2 del Fractal Coding Loop)

Mettendo insieme V1+V2+V3, il design aggiornato del FCL diventa:

```
┌─ INPUT ──────────────────────────────────────────────────────┐
│ user_prompt + project_state                                   │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
┌─ MEMORY RECALL (Wigner-weighted Fractal Memory) ─────────────┐
│ sample top-K memories from .fractal/memory/                   │
│ with weight ∝ Wigner(loss/avg_loss) / (1+log(1+visits))       │
│ inject into context                                           │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
┌─ HAND DECISION LEVEL (FMC swarm) ─────────────────────────────┐
│ spawn N=5 sub-agent (FINGER) in worktree-isolated git branches│
│ each FINGER: 1 candidate initial action (refactor X / test Y) │
│ evolve M=3 ticks (each = bash/edit/grep tool use)             │
│ score each FINGER end-state with multiplicative R:            │
│   R = R_alive · R_tests · (1+R_lint) · (1+R_goal) · ...       │
│ distance: AST-diff between FINGER states                      │
│ virtual_reward: relativize(R)^α · relativize(D)^β             │
│ cloning: bad FINGER inherit good FINGER's git branch          │
│ winner: argmax bincount(initial_action) among survivors       │
└──────────────────────────┬───────────────────────────────────┘
                           ↓
┌─ EXECUTE & FEEDBACK ─────────────────────────────────────────┐
│ apply winning action to real codebase                        │
│ measure outcome (tests pass? user happy?)                    │
│ append (state, action, outcome) to .fractal/memory/          │
│ update memory loss values                                    │
└──────────────────────────────────────────────────────────────┘
                           ↓
                 OPTIONAL: trigger re-sample
                 of memory bank with cloning step
                 (Fractal Memory dataset evolution)
```

**Differenze chiave da V1**:
1. Reward function è **moltiplicativa pura** (V1 era ibrida)
2. Architettura è **gerarchica** finger→hand (V1 era flat)
3. Memory sampling è **Wigner-weighted** (V1 non specificato)

## V5. Implicazione operativa: Phase 0 rimane valido

Buona notizia: **Phase 0 (slash command `/fractal-decide`) non cambia**. Implementa la HAND singola con N=3-5 FINGER, reward moltiplicativa, no memory bank ancora. È esattamente il punto di partenza.

Le V1+V2+V3 si integrano **incrementalmente**:
- Phase 1: aggiungi memory bank persistente (V3)
- Phase 2: aggiungi gerarchia OCTOPUS (V2)
- Phase 3: tuning della reward moltiplicativa (V1 raffinata)

## V6. Le simulazioni HTML come validation step

Le 4 simulazioni in [`simulations/`](../../simulations/) sono il **proof-of-concept visivo** del Fractal Coding Loop:

- 🚀 **Rocket** = analogo della HAND singola che pianifica (single agent)
- 🏎 **Kart** = analogo del task discreto con checkpoint reward (test pass milestones)
- 🕹 **Pong** = analogo del reactive control (real-time decisions in coding)
- 🐙 **Octopus** = analogo della gerarchia hand→octopus (multi-sub-agent)

L'utente che apre [`simulations/index.html`](../../simulations/index.html) e vede lo sciame muoversi sta vedendo **lo stesso meccanismo computazionale** che useremmo nel Fractal Coding Loop, solo applicato a giochi 2D invece che a codice.

**Validazione completata**: tutti e 4 i demo passano test end-to-end (vedi [`simulations/_test_envs.js`](../../simulations/_test_envs.js)).

## V7. Pronti per Phase 0?

Dopo questo aggiornamento, il design è completo. Le decisioni rimanenti sono engineering:

1. ✅ Reward function strutturata (V1)
2. ✅ Architettura gerarchica chiara (V2)
3. ✅ Memory weighting specificato (V3)
4. ✅ Implementazione di riferimento esistente (le 4 simulazioni)
5. ⏳ Phase 0 da implementare: slash command `/fractal-decide`

Manca solo decidere **quando partire**.

---

*Aggiornamento 2026-04-26 — post-bibliografia. La visione iniziale resta valida; i dettagli concreti (reward, gerarchia, weighting) sono ora ancorati alla letteratura primaria.*
