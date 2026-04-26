---
description: ONE FMC decision — N walkers × M ticks of perturbation+cloning toward a goal, then cherry-pick the winning init_commit. Use standalone for a single planning step, or loop via /octopus for goal-directed sessions.
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

# Fractal Decide — UNA decisione FMC verso un goal

Stai eseguendo **una singola decisione FMC** verso il goal: **$ARGUMENTS**.

Algoritmo (Hernández-Cerezo & Duran-Ballester 2020, §4.3):
> N walker proiettano M tick nel futuro in worktree git isolati. Tra un tick e l'altro avviene il cloning probabilistico via virtual reward (con ESS check per saltare quando lo sciame è già diverso). La strategia iniziale modale tra i sopravvissuti vince. Il **primo commit** del walker vincente entra nel branch principale.

**Default**: N=3 walker, M=3 tick. Override con env `FMC_N` / `FMC_M` se necessario.

---

## Phase 0 — Precondizioni

Verifica:
1. `git status` clean (o solo file untracked irrilevanti). Se non lo è, **chiedi all'utente** prima di procedere.
2. La main branch è identificabile: `MAIN_BRANCH=$(git branch --show-current)`.
3. La toolchain di test/lint è rilevabile (`package.json`, `pyproject.toml`, ecc.). Se nessuna esiste, **avvisa**: i reward saranno parziali (solo R_alive + R_diff).

Salva il punto di partenza:
```bash
MAIN_HEAD=$(git rev-parse HEAD)
MAIN_BRANCH=$(git branch --show-current)
```

---

## Phase 1 — Genera N strategie iniziali distinte

Analizza il goal `$ARGUMENTS`. Proponi **N=3 strategie ortogonali** che differiscano per *struttura*, non solo implementazione. Esempi di assi:

- **Scope**: minimale / medio / completo
- **Test strategy**: test-first / impl-first / refactor-then-add
- **Architettura**: in-place / extract-module / rewrite
- **Dependencies**: zero-deps / library-driven / framework-driven

Output di questa fase: una lista numerata 1..N con per ognuno un **label conciso** (max 3 parole) e una **descrizione** (15-30 parole).

Esempio:
```
1. label: "in-place-minimal"
   desc: "Modifica diretta del file esistente, niente moduli nuovi, ~20 righe"
2. label: "extract-module"
   desc: "Estrai un nuovo modulo dedicato, refactoring più ampio, test separati"
3. label: "test-first-tdd"
   desc: "Scrivi prima i test, poi l'implementazione minima per farli passare"
```

---

## Phase 2 — Inizializza la sessione FMC

```bash
SESS=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py init \
    --task "$ARGUMENTS" \
    --goal "$ARGUMENTS" \
    --n 3 --m 3 \
    --alpha 1.0 --beta 1.0 \
    --ess-threshold 0.7 \
    | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['session_id'])")
echo "FMC session: $SESS"
```

---

## Phase 3 — Tick 0: spawn N walker in INIT mode (in PARALLELO)

In **un singolo messaggio**, invoca il Task tool N=3 volte:

- `subagent_type: "fractal-walker"`
- `description: "Walker N: <label>"` (max 5 parole)
- `isolation: "worktree"` ← **obbligatorio**, ogni walker in worktree git separato
- `prompt`:
  ```
  MODE: init

  approach_label: <label_i>
  approach_description: <desc_i>

  Goal: $ARGUMENTS
  Task: $ARGUMENTS

  Follow the fractal-walker INIT protocol. Make exactly ONE commit
  in your worktree at the end (do not split into multiple commits).
  Return the JSON described in the protocol.
  ```

⚠️ **Lancia tutti N Task in parallelo** (singolo messaggio con N tool_use blocks). Aspetta la fine di tutti.

---

## Phase 4 — Registra tick 0

Raccogli i JSON dei walker in un file:
```bash
cat > /tmp/walkers_t0.json <<EOF
[<json_walker_1>, <json_walker_2>, <json_walker_3>]
EOF

python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py record \
    --session $SESS --file /tmp/walkers_t0.json
```

---

## Phase 5 — M-1 cicli di (step → clone → continuation)

**Per ciascun tick t in 1..M-1** (cioè 2 cicli con M=3):

### Phase 5a — Compute step (VR + ESS + clone_plan)

```bash
STEP_RESULT=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py step \
    --session $SESS --seed $((42 + t)))
echo "$STEP_RESULT"
```

Estrai dai risultati:
- `cloning_skipped`: bool (se ESS > 0.7 * N)
- `clone_plan`: lista di clonazioni da eseguire
- `ess`, `ess_threshold_abs`: per logging
- `walkers`: lista di {idx, alive, R, VR, init_action_label}

### Phase 5b — Esegui git reset per ogni entry del clone_plan

Se `cloning_skipped: true`, salta a Phase 5c.

Altrimenti, per ogni entry in `clone_plan`:
```bash
# Reset il worktree del walker dst allo stato del walker src
cd <clone.dst_path>
git reset --hard <clone.src_head>
cd -
```

### Phase 5c — Apply clones nello stato

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py apply-clones --session $SESS
```

Lo state ora contiene init_action_label e init_commit_sha aggiornati per i walker clonati.

### Phase 5d — Spawn N walker in CONTINUATION mode (in PARALLELO)

Recupera lo stato corrente di ogni walker:
```bash
STATUS=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py status --session $SESS)
```

Per ogni walker idx in 0..N-1, invoca Task tool con:
- `subagent_type: "fractal-walker"`
- `description: "Walker $idx: cont t=$t"`
- **Non usare `isolation: worktree`** stavolta — riusa il worktree esistente del walker passandone il path nel prompt
- `prompt`:
  ```
  MODE: continuation

  strategy_label: <walker.init_action_label>
  strategy_description: <walker.init_action_desc>

  Goal: $ARGUMENTS

  Your worktree is at: <walker.current_path>
  cd to it before doing anything else.

  Recent history (last 5 commits):
  <output di `git -C <walker.current_path> log --oneline -5`>

  Follow the fractal-walker CONTINUATION protocol. Make ONE small
  step toward the goal, preserving the strategy. Max 30 lines diff,
  one commit. Return the JSON described in the protocol.
  ```

⚠️ Lancia tutti N in parallelo.

### Phase 5e — Registra tick t

```bash
cat > /tmp/walkers_t${t}.json <<EOF
[...]
EOF
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py record \
    --session $SESS --file /tmp/walkers_t${t}.json
```

---

## Phase 6 — Tick finale + decide

Al tick M-1 (l'ultimo), dopo aver registrato i walker, fai un **ultimo step** (per VR + bincount finale) e poi `decide`:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py step --session $SESS --seed 999
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py apply-clones --session $SESS

DECISION=$(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_loop.py decide --session $SESS)
echo "$DECISION"
```

Il JSON `DECISION` contiene:
- `winner_label`
- `winner_init_commit_sha` ← QUESTO è cosa cherry-picchiamo
- `confidence`
- `is_tie` (bool) — true se più label condividono il bincount massimo
- `tied_labels` (list[str]) — sempre presente (length 1 se no tie)
- `tie_break_method` (str|null) — `"highest_R_among_tied"` quando is_tie, altrimenti null
- `vote_distribution`
- `alive_count` / `total_walkers`
- `winner_path`, `winner_branch` (per debug)

---

## Phase 7 — Tabella di confronto all'utente

Stampa:

```
| #  | Strategy           | Final R | VR (last) | Alive | Votes |
|----|--------------------|---------|-----------|-------|-------|
| 1  | in-place-minimal   | 4.32    | 0.94      | ✓     | 2     |
| 2  | extract-module     | 2.81    | 0.41      | ✓     | 1     |
| 3  | test-first-tdd     | 0.00    | 0.00      | ✗     | 0     |

Winner: in-place-minimal (confidence 67%, ESS history: [2.1, 2.7, 2.9])
First commit: <sha>  "FMC walker [init]: in-place-minimal — ..."
```

Sotto la tabella: 2-3 righe di analisi (perché ha vinto, eventuali caveat su confidence bassa).

⚠️ **Avviso esplicito** se confidence < 0.50: "Decisione rischiosa, considera review manuale."

⚠️ **Se `is_tie: true`** (più label condividono il bincount massimo): mostra TUTTI i `tied_labels` e indica esplicitamente "**3-way tie**: il vincitore è stato scelto via highest-R tie-break (`tie_break_method`), ma le tre strategie sono funzionalmente equivalenti per lo sciame. Considera di ispezionare manualmente i tre worktree prima di applicare." Mostra il path di OGNI walker con label tied, non solo del winner.

---

## Phase 8 — Cherry-pick OR return

Se questo `/fractal-decide` è chiamato da `/octopus`, **non fare cherry-pick** — solo emettere il JSON di decisione su stdout (l'octopus loop lo cattura).

Se chiamato standalone, chiedi all'utente:
> "Apply winner_init_commit_sha to $MAIN_BRANCH? (yes / no / inspect)"

Se yes:
```bash
git checkout $MAIN_BRANCH
git cherry-pick <winner_init_commit_sha>
```

Se inspect: mostra il diff del commit, poi richiedi.

Se no: lascia tutto in stato corrente, l'utente potrà cherry-pickare manualmente.

---

## Phase 9 — Cleanup

Distruggi i worktree dei walker non vincenti:
```bash
for path in <non-winner walker paths>; do
    git worktree remove "$path" --force || true
done
for branch in <non-winner walker branches>; do
    git branch -D "$branch" || true
done
```

Conserva il worktree del winner finché l'utente non conferma il merge (per ispezione).

Salva la sessione in `.fractal/sessions/$SESS/` per audit/replay.

---

## Vincoli importanti

- **Reward composta moltiplicativa** (paper §2.2.2 + post Pareto 2016): hard constraint × hard × (1+soft) × (1+soft)
- **Worktree isolation OBBLIGATORIA al tick 0** (init mode). Riuso del worktree ai tick > 0.
- **N≥3 per non avere ratios degeneri nel relativize**. Default 3, alza a 5 per goal complessi.
- **M≥2 per avere senso del cloning intermedio**. M=1 sarebbe degenerare al singolo voto. Default 3.
- **Solo il PRIMO commit del walker winner** entra nel main. Il resto del rollout è scartato (è solo proiezione).

Procedi con disciplina. Se a qualsiasi phase serve un chiarimento, chiedi PRIMA di lanciare walker (ogni walker costa).
