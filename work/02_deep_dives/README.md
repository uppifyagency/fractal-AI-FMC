# 02 — Deep dives teoriche

**Goal verificabile**: ogni deep-dive sta tra 600 e 1200 righe, con riferimenti puntuali al codice (`file:linea`) e bibliografia citata. Ogni claim non banale ha una citazione.

## Indice

| File | Argomento | Stato | Pagine paper di riferimento |
|---|---|---|---|
| [`01_cloning_mathematics.md`](01_cloning_mathematics.md) | Matematica del cloning come Markov chain con equilibrio di Gibbs | ✅ scritto | Book #1 §4.2 (29-34) |
| [`02_active_inference_link.md`](02_active_inference_link.md) | Connessione con il Free Energy Principle di Friston | 📝 outline | Book #1 §6.4 (51) |
| [`03_standard_model_cognition.md`](03_standard_model_cognition.md) | Standard Model of Cognition (gauge theory) di Fragile Mechanics | 📝 outline | n/a (fragile-rl) |
| [`04_relativize_axiomatics.md`](04_relativize_axiomatics.md) | Caratterizzazione assiomatica della trasformazione `relativize` | 📝 outline | Book #1 §2.2.3 (12-13) |
| [`05_smc_particle_filter_view.md`](05_smc_particle_filter_view.md) | FMC come Sequential Monte Carlo con peso virtual reward | ✅ scritto | Book #1 §4.4 (36-39) |
| [`06_book2_badger_fractal_memory.md`](06_book2_badger_fractal_memory.md) | Book #2 (AGI Structure) + Fractal Memory: Badger di sciami nidificati, learning as collapse, FM su dataset/sinapsi/NN | ✅ scritto | Book #2 (V0.2), Hives, Fractal Memory Slide |
| [`07_wright_fisher_mapping.md`](07_wright_fisher_mapping.md) | FMC come processo neutro Wright-Fisher; falsifica empirica del "magic 6" branching di Sergio: $b_{\text{eff}}^* \approx 1.53\,K^{0.6}$ (dipende da K). Sweep numerico in [`work/07_sergio_branching_sweep/REPORT.md`](../07_sergio_branching_sweep/REPORT.md). | ✅ scritto | n/a (analisi empirica originale; correlazione vs Radient 2026 cap.16) |
| [`08_video_seminar_extracted_insights.md`](08_video_seminar_extracted_insights.md) | Estrazione e controverificazione delle formule e nozioni dal video-seminario su slide di Sergio (`VideoTranscriptSergio.md`); cross-entropy collapse, metafora minatore, coscienza tripla, discrepanza efficienza vs MCTS. **§7 contiene verifica numerica controllata di F12 (Gibbs equilibrium, log-Pearson 0.77 a α=1) e F11 (raw_signed produce "fearful agent" come predetto da Sergio).** Codice in [`work/04_mathematical_tests/`](../04_mathematical_tests/). | ✅ scritto + test | Confronto vs paper §2.6, §3, §4.4-5, §5; vs Radient 2026; vs THEORY/ALGORITHM |

## Convenzioni

- **Citazioni paper**: `(Hernández-Cerezo & Duran-Ballester, 2020, §X.Y, p. Z)`
- **Citazioni codice**: link markdown a `repos/<nome>/<file>:<linea>`
- **Notazione matematica**: LaTeX inline `$...$`, blocchi `$$...$$`
- **Lemmi e teoremi**: numerati `Lemma 1.1`, `Teorema 2.3`, ecc.
- **Dimostrazioni**: chiuse con `∎`

## Priorità di scrittura

In ordine di valore-per-tempo:

1. **01 — Cloning mathematics** (✅ scritto). È il cuore algoritmico: senza questo, il resto non si capisce.
2. **05 — SMC view** (✅ scritto). Mappa FMC sulla letteratura SMC consolidata; forte per pubblicazione.
3. **06 — Book #2 + Fractal Memory** (✅ scritto). Espande oltre il paper isolato; necessario per il quadro completo.
4. **02 — Active Inference**. Lega Fractal AI alla cornice di Friston, oggi dominante in neuroscienze computazionali.
5. **04 — Relativize axiomatics**. Risolve il "buco" assiomatico segnalato in §10.3 di [`ANALISIS.md`](../../ANALISIS.md).
6. **03 — Standard Model of Cognition**. Più speculativo; richiede maggior tempo di studio dei docs `fragile-rl`.
7. **07 — Wright-Fisher mapping** (✅ scritto). Falsifica empirica di un claim specifico di Sergio (Radient cap.16). Lavoro originale, non derivato.
8. **08 — Video seminario** (✅ scritto + test). Sintesi della seconda fonte orale di Sergio; incrocia paper, Radient e video; chiude alcune discrepanze, ne lascia aperte altre (vedi tabella in [`CLAUDE.md`](../../CLAUDE.md)).
