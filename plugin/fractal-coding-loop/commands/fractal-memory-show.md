---
description: Mostra tutte le memorie nella Fractal Memory bank con i loro stats
allowed-tools:
  - Bash
  - Read
---

# Fractal Memory Show

Stato attuale della memory bank:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_memory.py show
```

Mostra il JSON come tabella ordinata per timestamp DESC. Aggiungi:

- **Stats globali**: numero memorie totali, loss media, visits medi
- **Distribuzione**: quante memorie con loss < 0.2 ("imparate"), 0.2-0.5 ("zona apprendimento"), > 0.5 ("ancora difficili")
- **Suggerimento**: se ci sono memorie con `loss < 0.05 AND visits > 10`, suggerisci `/fractal-memory-prune` per pulire.
