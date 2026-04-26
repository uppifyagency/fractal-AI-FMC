---
description: Octopus loop — repeated FMC decisions toward a goal G until completion or budget exhausted. Each iteration is one /fractal-decide; the winning init_commit is cherry-picked to main; a judge checks goal completion.
argument-hint: "[goal description]"
allowed-tools:
  - Task
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - TodoWrite
---

# /octopus — outer loop goal-directed

Stai eseguendo un **OCTOPUS LOOP** — la struttura gerarchica di Hernández-Cerezo:

> *fingers (walker) → hand (FMC decision) → octopus (questa sessione, K iterazioni) → multiple octopuses (team)*

L'algoritmo è strutturalmente identico al main loop di Atari: ad ogni "frame", FMC sceglie UNA azione (qui: UN commit), il mondo evolve (qui: il main branch avanza), si ricontrolla la condizione di terminazione (qui: goal raggiunto?).

```
GOAL G = $ARGUMENTS
K_MAX = 10  (max iterazioni)
THRESHOLD_GOAL = 0.95  (judge score per dichiarare goal raggiunto)

iteration = 0
goal_score = 0.0

while iteration < K_MAX and goal_score < THRESHOLD_GOAL:
    decision = run_fractal_decide(goal=G)
    cherry_pick(decision.winner_init_commit_sha to main)
    goal_score = run_goal_judge(main_HEAD, G)
    iteration += 1

if goal_score >= THRESHOLD_GOAL:
    print "GOAL REACHED in {iteration} iterations"
else:
    print "BUDGET EXHAUSTED, partial progress: {goal_score}"
```

---

## Phase 0 — Setup

```bash
GOAL="$ARGUMENTS"
K_MAX="${OCTOPUS_K_MAX:-10}"
THRESHOLD="${OCTOPUS_THRESHOLD:-0.95}"
MAIN_BRANCH=$(git branch --show-current)
START_HEAD=$(git rev-parse HEAD)

echo "Octopus loop:"
echo "  goal: $GOAL"
echo "  main: $MAIN_BRANCH @ $START_HEAD"
echo "  K_max: $K_MAX, threshold: $THRESHOLD"
```

Verifica precondizioni:
1. `git status` clean
2. Toolchain test/lint identificabile
3. Goal G ben definito (se troppo vago, **chiedi prima** di lanciare il loop — ogni iterazione costa N×M sub-agent call)

Crea log file:
```bash
LOG=".fractal/octopus_$(date +%Y%m%d_%H%M%S).log"
mkdir -p .fractal
echo "octopus_start: $(date -Iseconds)" > $LOG
echo "goal: $GOAL" >> $LOG
echo "start_head: $START_HEAD" >> $LOG
```

Usa **TodoWrite** per tracciare le iterazioni.

---

## Phase 1 — Loop principale

Per `iteration` in `1..K_MAX`:

### 1a — Esegui UNA decisione FMC

Invoca `/fractal-decide $GOAL` come sub-procedura. Cattura il JSON di decisione **senza fare cherry-pick** (vedi `/fractal-decide` Phase 8 — quando chiamato da octopus, deve solo emettere la decisione).

In pratica: esegui le Phase 0-7 di `/fractal-decide` direttamente in questo loop. Salta Phase 8 (cherry-pick); fallo qui sotto.

```bash
# Cattura decision JSON
DECISION_JSON=$(... output di Phase 6 di fractal-decide ...)
WINNER_SHA=$(echo "$DECISION_JSON" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['winner_init_commit_sha'])")
WINNER_LABEL=$(echo "$DECISION_JSON" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['winner_label'])")
CONFIDENCE=$(echo "$DECISION_JSON" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['confidence'])")

echo "[iter $iteration] winner=$WINNER_LABEL confidence=$CONFIDENCE sha=$WINNER_SHA" | tee -a $LOG
```

### 1b — Confidence gate + tie gate

Se `confidence < 0.50` OR `is_tie == true`: **stop e chiedi all'utente**. Decisione troppo rumorosa o ambigua per applicare automaticamente.

> "Iteration $iteration: confidence=$CONFIDENCE, is_tie=$IS_TIE.
>  Tied labels: $TIED_LABELS.
>  Vuoi: continue (applica winner via tie-break) / stop / inspect (mostro i diff dei tied walker)?"

Se `is_tie == true`, mostra all'utente i diff di TUTTI i walker con label tied (non solo del winner) prima di chiedergli se procedere. Le strategie tied sono funzionalmente equivalenti per lo sciame; la scelta finale tra loro merita ispezione umana.

### 1c — Cherry-pick il winning init_commit nel main

```bash
git checkout $MAIN_BRANCH
git cherry-pick $WINNER_SHA
NEW_HEAD=$(git rev-parse HEAD)
echo "  applied: $NEW_HEAD" | tee -a $LOG
```

Se cherry-pick fallisce per conflitto:
1. Logga conflitto nel $LOG
2. **Stop il loop** e chiedi intervento utente — un conflitto in cherry-pick è segnale che main è divergente da quello che i walker stavano vedendo all'inizio del FMC decision

### 1d — Cleanup worktree non-winner

```bash
# Solo i non-winner — il winner worktree può essere mantenuto temporaneamente per ispezione
for path in <non-winner paths>; do
    git worktree remove "$path" --force || true
done
```

### 1e — Goal check (judge sub-agent)

Invoca un task `fractal-judge` con prompt:
```
You are evaluating GOAL COMPLETION (not goal alignment of a single walker).

Original goal: $GOAL
Current state of main branch ($MAIN_BRANCH @ $NEW_HEAD):
<git log --oneline da $START_HEAD>
<git diff $START_HEAD..$NEW_HEAD --stat>

Question: Is the goal NOW COMPLETE? Score 0..1 where:
  1.0 = fully complete, all aspects of the goal addressed
  0.95 = essentially complete, minor polish only would remain
  0.7 = substantial progress, but more work needed
  0.5 = partial progress, key aspects still missing
  0.0 = no meaningful progress

Return JSON:
{
  "goal_score": <float>,
  "rationale": "<2-3 sentences>",
  "remaining_aspects": ["..."],
  "ready_to_stop": <bool>
}
```

```bash
GOAL_SCORE=$(... extract from judge response ...)
echo "  goal_score: $GOAL_SCORE" | tee -a $LOG
```

### 1f — Termination check

```bash
if [ "$(echo "$GOAL_SCORE >= $THRESHOLD" | bc)" = "1" ]; then
    echo "GOAL REACHED at iteration $iteration"
    break
fi
```

Aggiorna TodoWrite.

---

## Phase 2 — Reporting finale

Dopo che il loop termina (per goal o per budget):

```bash
END_HEAD=$(git rev-parse HEAD)
git log --oneline $START_HEAD..$END_HEAD

echo "octopus_end: $(date -Iseconds)" >> $LOG
echo "iterations: $iteration" >> $LOG
echo "final_goal_score: $GOAL_SCORE" >> $LOG
echo "commits_applied: $(git rev-list --count $START_HEAD..$END_HEAD)" >> $LOG
```

Stampa all'utente:

```
╔══════════════════════════════════════════════════════════════╗
║  OCTOPUS LOOP — final report                                 ║
║──────────────────────────────────────────────────────────────║
║  Goal:           $GOAL                                        ║
║  Iterations:     $iteration / $K_MAX                          ║
║  Final score:    $GOAL_SCORE (threshold $THRESHOLD)            ║
║  Commits to main: <count> (from $START_HEAD..$END_HEAD)        ║
║  Status:         [REACHED ✓ / BUDGET ✗ / STOPPED USER]         ║
║──────────────────────────────────────────────────────────────║
║  Iteration log:                                               ║
║  iter 1: winner=<label> conf=<x> goal_score=<y>               ║
║  iter 2: ...                                                  ║
║  ...                                                          ║
║──────────────────────────────────────────────────────────────║
║  Full log: $LOG                                               ║
╚══════════════════════════════════════════════════════════════╝
```

Se `[BUDGET ✗]`: suggerisci all'utente di:
- Riavviare con K_MAX più alto, OPPURE
- Restringere il goal e rilanciare

Se `[STOPPED USER]`: il branch è in stato intermedio, l'utente decide se continuare o rollback con `git reset --hard $START_HEAD`.

---

## Phase 3 — Rollback option (se richiesto)

Se l'utente alla fine non è soddisfatto:
```bash
git reset --hard $START_HEAD
```

Tutte le decisioni dell'octopus sono in $LOG e nel `.fractal/sessions/` se vuole replicare manualmente alcune.

---

## Vincoli importanti

- **Goal G deve essere espresso come acceptance criterion**, non come task generico. Esempio:
  - ❌ "fai funzionare l'auth"
  - ✅ "endpoint POST /login accetta credenziali, ritorna JWT valido, test in tests/auth_test.py passa"
- **Costo per iterazione**: N×M sub-agent walker call + 1 judge call. Con N=3, M=3 → 10 chiamate per iter. Con K_MAX=10 → ~100 chiamate totali. Stima il costo prima di lanciare.
- **Conflitti di cherry-pick fermano il loop**: significa che la traiettoria walker è divergente da main. Investigare prima di forzare.
- **Confidence < 0.5 ferma il loop**: significa che lo sciame non ha consenso. Forzare significa propagare rumore nel main branch.
- **Logging completo è non-negoziabile**: ogni iterazione, decisione, judge_score nel $LOG. È il replay dell'octopus.

Inizia.
