# Installation — Fractal Coding Loop plugin

## Requisiti

- **Claude Code** ≥ recent version (with plugin support)
- **Python 3.8+** (per gli script di reward + memory)
- **git** ≥ 2.5 (per `git worktree`)
- *Opzionale*: la toolchain di test/lint del tuo progetto (pytest, eslint, ecc.)

## Step 1 — Installa il plugin

### Opzione A: symlink (raccomandato per dev)

```bash
mkdir -p ~/.claude/plugins
ln -s "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/plugin/fractal-coding-loop" \
      ~/.claude/plugins/fractal-coding-loop
```

### Opzione B: copia (per uso stabile)

```bash
cp -r "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/plugin/fractal-coding-loop" \
      ~/.claude/plugins/
```

## Step 2 — Verifica installazione

Restart Claude Code, poi in una nuova sessione:

```
> /help
```

Dovresti vedere `/fractal-decide`, `/fractal-recall`, `/fractal-memory-show` nella lista.

Se non li vedi:
- Verifica che il symlink/copia esista: `ls ~/.claude/plugins/fractal-coding-loop/`
- Verifica `.claude-plugin/plugin.json`: deve essere parsabile JSON
- Restart Claude Code completamente

## Step 3 — Verifica gli script Python

```bash
python3 ~/.claude/plugins/fractal-coding-loop/scripts/fractal_reward.py --help
python3 ~/.claude/plugins/fractal-coding-loop/scripts/fractal_memory.py --help
```

Entrambi dovrebbero stampare l'help senza errori.

## Step 4 — Run smoke test

```bash
bash ~/.claude/plugins/fractal-coding-loop/tests/e2e_test.sh
```

Output atteso: `OK ✓ — all components functional`.

## Step 5 — Primo utilizzo

In un repo git pulito:

```
> /fractal-decide explain what this codebase does
```

(Task volutamente leggero per il primo test.)

Il plugin lancerà 3 sub-agent in parallelo, ognuno produrrà una spiegazione, e ti mostrerà la comparison.

## Troubleshooting

| Sintomo | Causa probabile | Soluzione |
|---|---|---|
| `/fractal-decide` non appare | plugin non riconosciuto | check `.claude-plugin/plugin.json` syntax |
| Walker fail with "not in worktree" | repo not git-init | `git init && git commit` first |
| Reward script `ModuleNotFoundError` | Python missing | install python3 ≥ 3.8 |
| Memory bank empty | first run | normal, popolato dopo prima `/fractal-decide` |
| Walkers all return same approach | sub-agent non distinguono | rivedi le approach descriptions in fase 1 |
| Confidence sempre < 30% | high reward variance | aumenta N o specifica approcci più diversi |

## Disinstallazione

```bash
rm ~/.claude/plugins/fractal-coding-loop  # if symlink
# OR
rm -rf ~/.claude/plugins/fractal-coding-loop  # if copied
```

I worktree e la memory bank di un progetto vivono in `<project>/.fractal/` e `<project>/<worktree-paths>/`. Per ripulire un progetto:

```bash
cd <your-project>
rm -rf .fractal/
git worktree list  # see active fractal worktrees
git worktree remove <path>  # for each fractal-walker-* path
git branch -D fractal-walker-*
```
