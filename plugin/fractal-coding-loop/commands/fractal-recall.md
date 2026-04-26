---
description: Recall the most relevant past Fractal Coding decisions (Wigner-weighted memory bank)
argument-hint: "[query keyword]"
allowed-tools:
  - Bash
---

# Fractal Recall — memoria selettiva

Mostro le decisioni FMC passate più rilevanti al query: **$ARGUMENTS**.

L'algoritmo di sampling è **Wigner-weighted** (Slide doc 2020): privilegia memorie a difficoltà media (loss vicino alla media), debiased per visite.

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_memory.py recall \
    --query "$ARGUMENTS" \
    --top-k 5 \
    --mark-visited
```

Mostrami output JSON formattato come tabella markdown con colonne:
| weight | task | winner | confidence | loss | visits |

Sotto la tabella, **una riga di interpretazione**: "memorie più rilevanti = quelle con loss vicino a media (zona di apprendimento attiva). Memorie con loss troppo basso sono già padroneggiate; troppo alto sono ancora troppo difficili."
