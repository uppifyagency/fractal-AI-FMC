---
description: Spawn N walker sub-agents to explore distinct approaches in parallel; pick the winner via FMC virtual reward
argument-hint: "[task description]"
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

# Fractal Decide — pianificazione FMC su task di coding

Stai eseguendo un **Fractal Monte Carlo decision** per il task: **$ARGUMENTS**.

L'algoritmo è quello di Hernández-Cerezo & Duran-Ballester (2020): N walker proiettano il futuro in parallelo, si confrontano via virtual reward, e l'azione iniziale modale tra i sopravvissuti vince.

## Procedura (segui esattamente)

### Fase 1 — Generate 3 candidate approaches (initial actions)

Analizza brevemente il task `$ARGUMENTS` e proponi **3 approcci distinti** che differiscano per *strategia*, non solo per implementazione. Esempi di assi di differenziazione:

- **strategia di refactoring**: in-place vs estrazione modulo vs riscrittura completa
- **strategia di test**: test-first vs implementazione-first vs no-test
- **strategia di scope**: minimale vs medio vs completo

Output di questa fase: una lista numerata 1) 2) 3) con per ognuno un **label conciso** (es. "in-place minimal", "extract-module", "test-first") e una **frase di descrizione** (10-25 parole) che spiega l'approccio.

### Fase 2 — Spawn N=3 walkers via Task tool (in parallelo)

In **un singolo messaggio**, invoca il Task tool 3 volte (una per approccio) con questi parametri:

- `subagent_type: "fractal-walker"`
- `description: "Walker N: [label]"` (max 5 parole)
- `isolation: "worktree"` ← **fondamentale**, ogni walker lavora in un worktree git isolato
- `prompt: "[approach_description]\n\nTask: $ARGUMENTS\n\nFollow the fractal-walker protocol to implement this approach in your worktree."`

⚠️ Lancia tutti e 3 i Task in **parallelo** (singolo messaggio con 3 tool_use blocks). Aspettare la fine di tutti.

### Fase 3 — Collect walker outputs

Ogni walker ritorna un JSON strutturato come:
```json
{
  "approach_label": "...",
  "worktree_path": "...",
  "worktree_branch": "...",
  "files_changed": [...],
  "diff_summary": "...",
  "tests_run": true|false,
  "tests_passed": int,
  "tests_total": int,
  "lint_warnings": int,
  "compile_ok": true|false,
  "summary": "..."
}
```

Raccogli gli output dei 3 walker.

### Fase 4 — Compute multiplicative reward

Usa lo script bundled nel plugin:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_reward.py --walker-jsons '<json1>' '<json2>' '<json3>'
```

Lo script applica la formula moltiplicativa del paper §2.2.2:

```
R = R_alive · R_tests · (1 + R_lint) · (1 + R_diff) · (1 + R_goal)
```

con `relativize` post-process. Ritorna un JSON `{walkers: [...], winner_idx: int, confidence: float, breakdown: {...}}`.

### Fase 5 — Show comparison table to user

Stampa una tabella markdown con:

| # | Approach | Tests | Lint | Diff size | Reward | Confidence |
|---|---|---|---|---|---|---|

Sotto la tabella: **2-3 righe di analisi** (es. "L'approccio 2 vince per +X% di reward perché Y. L'approccio 3 ha tests rotti").

### Fase 6 — Ask user which to merge

Chiedi all'utente esplicitamente: *"Quale approccio vuoi mergeare? (1/2/3, oppure 'cancel' per scartare tutti)"*.

### Fase 7 — Apply winner

Se l'utente sceglie:
- **N (1/2/3)**: cherry-pick o merge dal worktree del walker N nella branch principale.
  ```bash
  cd <worktree_path_N> && git format-patch HEAD~..HEAD --stdout | (cd <main_repo> && git apply)
  ```
  Oppure (se preferito): `git merge --no-ff <branch_N>` se il workflow lo permette.

- **cancel**: non applicare nulla. Mostrare solo "Decisione scartata. Walker disponibili in: <list_branches>".

### Fase 8 — Persist to memory bank

Indipendentemente dalla scelta dell'utente, salva l'episodio nella Fractal Memory:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_memory.py append \
    --task "$ARGUMENTS" \
    --winner-idx <chosen_or_-1_if_cancelled> \
    --walkers-json '<full_walkers_array_json>'
```

Lo script aggiunge una entry a `.fractal/memory/<timestamp>_<task_slug>.md` con frontmatter strutturato. Verrà usata in futuri `/fractal-decide` per Wigner-weighted recall.

### Fase 9 — Cleanup

Distruggi i worktree dei walker non scelti:

```bash
git worktree remove <walker_path> --force  # for each non-winner
git branch -D <walker_branch>               # for each non-winner
```

## Vincoli importanti

- **Reward composta moltiplicativa**, non additiva (post Pareto 2016 + paper §2.2.2)
- **Sub-agent isolation: worktree** è obbligatoria — niente walker che fanno casino sulla branch attuale
- **Mostra la confidence anche bassa**: convergenza < 50% = "decisione rischiosa, considera review manuale"
- Il task `$ARGUMENTS` ha la priorità — se l'utente è ambiguo, **chiedi prima** di lanciare i walker (che costano)

## Stato iniziale: verifica precondizioni

Prima della Fase 1, verifica:

1. Il working directory è un repo git pulito (`git status` clean o solo file untracked irrilevanti). Se ha modifiche non-committate, **chiedi all'utente** se proseguire (potrebbero perdersi nei worktree branch).
2. La branch principale è identificabile (`git branch --show-current`).
3. Esistono comandi test/lint comuni (cerca `package.json`, `pyproject.toml`, `Cargo.toml`, ecc.). Se non c'è nessuna toolchain test/lint, **avvisa** l'utente che i reward saranno parziali (solo R_alive + R_diff).

Procedi.
