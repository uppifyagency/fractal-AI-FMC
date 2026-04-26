# Run log — smoke test of /fractal-decide

> *Eseguito 2026-04-26. Orchestrazione manuale (replicata dal main agent della sessione corrente, perché slash command non invocabile self-referentially). Pipeline FMC fedele.*

## Run metadata

- **Date**: 2026-04-26 23:13 CEST
- **Plugin version**: 0.1.0
- **Models used**: walker=sonnet (general-purpose sub-agent), no judge calls (R_goal default 0.7)
- **Plugin parameters**: N=3, M=3, ess_threshold=0.7, alpha=1.0, beta=1.0
- **Repo destination**: /tmp/fmc-smoke-repo
- **Initial scaffold commit**: 30123be431f7dd2a6c72407f65c897aa6ec0da94
- **FMC session id**: 20260426_231328_6749

## Phase-by-phase observations

### Pre-flight
✓ Plugin scripts presenti, math test 5/5 + e2e 17/17, pytest 9.0.2 disponibile sistema.

### Phase 1 — Strategy generation (eseguita dal main agent — io)
Strategie scelte (assegnate manualmente, replicando ciò che farebbe il `/fractal-decide` Phase 1):
1. **early-return-cascade** — `if/elif` chain, controllo most-specific (15) per primo
2. **boolean-string-build** — truthy-multiplication: `'Fizz' * (n%3==0) + 'Buzz' * (n%5==0)`
3. **match-statement** — `match/case` su tupla `(n%3==0, n%5==0)`

Genuinamente ortogonali: cascade imperativo vs concatenazione idiomatica vs pattern matching dichiarativo.

### Phase 3 — Walker spawn tick 0 (init mode, 3 in parallelo)
✓ 3 worktree creati: `/tmp/fmc-walker-0/1/2`, branched da `main@30123be`.
✓ Tutti e 3 walker hanno completato init in **~33-37s** ciascuno (parallelo).

| Walker | Strategy | Init commit SHA | LOC +/- | Tests |
|---|---|---|---|---|
| 0 | early-return-cascade | `0befdaf9...` | +11/-2 | **5/5** |
| 1 | boolean-string-build | `11a9a25b...` | +5/-2 | **5/5** |
| 2 | match-statement | `de7f16b2...` | +12/-2 | **5/5** |

### Phase 4 — Record tick 0
✓ State JSON inizializzato. Tutti e 3 walker registrati con `init_commit_sha` correttamente catturato.

### Phase 5, tick 1 — step + clone
- **ESS = 2.567**, threshold = 2.100 → **cloning_skipped = True**
- VR distribuzione: idx 0=0.288, idx 1=0.925, idx 2=0.794 (D term penalizza walker 0 per pairing partner ravvicinato)
- clone_plan: vuoto (saltato per ESS alto)
- Apply-clones: 0 applicate, tick avanzato a 1.

### Phase 5, tick 1 — continuation walker
- W0: +13 righe (docstring + comment ordering rationale), commit `99d9dc12...`
- W1: +1 riga (inline comment idiom), commit `f445aae0...`
- W2: +1 riga (inline comment tuple key), commit `32b1fa5f...`
- Tutti 5/5, py_compile ok.

### Phase 5, tick 2 — step + clone
- **ESS = 2.868** → **cloning_skipped = True** (ancora!)
- VR: 0.457 / 0.757 / 0.757 (W1 e W2 ora identici per file_overlap distance)
- clone_plan vuoto.

### Phase 5, tick 2 — continuation walker
- W0: +10/-3 (estratto `Final[int]` constants per 15/3/5), commit `db9d01f7...`
- W1: +7/-1 (docstring + `result: str` annotation), commit `7e5c46a1...`
- W2: +6/-1 (`DivKey = tuple[bool, bool]` alias), commit `f46f49a4...`
- Tutti 5/5.

### Phase 6 — Final step + decide
- **ESS = 1.735** < 2.100 → **cloning_skipped = False** (finalmente!)
- Tuttavia clone_plan vuoto (`applied: 0`): le pairing random non hanno prodotto VR_partner > VR_self con random draw fortunato per nessuno dei 3 walker. **Cloning mechanism abilitato ma 0 clone fatti per via della stocasticità del partner sampling**.
- VR finale: idx 0=0.380, idx 1=2.165, idx 2=0.398. Walker 1 ha la VR più alta ma NON vince per bincount.

**DECIDE output**:
```json
{
  "winner_label": "early-return-cascade",
  "winner_init_commit_sha": "0befdaf9c12e947e3f28c46a30c8a4d842730350",
  "vote_distribution": {"early-return-cascade": 1, "boolean-string-build": 1, "match-statement": 1},
  "confidence": 0.333,
  "alive_count": 3,
  "ticks_completed": 3
}
```

### Phase 8 — Cherry-pick to main
✓ `git cherry-pick 0befdaf9...` su main → nuovo HEAD `b372e553...`
✓ Cherry-pick clean, niente conflitti.

### Phase 9 — Cleanup
✓ Worktree walker-1 e walker-2 rimossi. Branch fmc-walker-1 e fmc-walker-2 cancellati.
✓ Worktree walker-0 (winner) preservato per ispezione utente — comportamento documentato.

## Final verification

```
$ pytest -v
tests/test_fizzbuzz.py::test_returns_string_for_normal_number PASSED
tests/test_fizzbuzz.py::test_multiple_of_three_returns_fizz PASSED
tests/test_fizzbuzz.py::test_multiple_of_five_returns_buzz PASSED
tests/test_fizzbuzz.py::test_multiple_of_fifteen_returns_fizzbuzz PASSED
tests/test_fizzbuzz.py::test_zero_or_negative_raises_value_error PASSED

============================== 5 passed in 0.01s ===============================
```

✅ **Tutti e 5 i test passano sul main**.

LOC della soluzione su main (esclusa docstring già presente):
```
def fizzbuzz(n: int) -> str:
    if n <= 0: raise ValueError(...)
    if n % 15 == 0: return "FizzBuzz"
    elif n % 3 == 0: return "Fizz"
    elif n % 5 == 0: return "Buzz"
    else: return str(n)
```
~11 righe di logica.

## Cost & time

- **Wall time totale (init → cherry-pick + verifica)**: ~10 minuti
- **Sub-agent calls**: 9 walker (3 init + 3 cont t1 + 3 cont t2). Zero judge call (R_goal default).
- **Token usati**: ~21k per walker × 9 = ~189k token totali walker side
- **Costo stimato**: ~$3-4 (Sonnet at quoted rates)

## Bugs / surprises observed

1. **Git identity non configurata nei worktree** — Severity 2.
   Tutti i walker hanno dovuto usare `git -c user.email=... -c user.name=...` inline per committare. Il `setup_smoke_repo.sh` configura solo l'identity per il commit iniziale di main, non per i worktree creati DOPO. Fix: aggiungere `git config user.email/user.name` globalmente nello script, oppure includerlo nel walker prompt come obbligatorio.

2. **Tiebreak indefinito quando bincount è uniforme** — Severity 2.
   Con 3 walker tutti vivi e nessun cloning, la distribution era {1,1,1}. `Counter.most_common(1)` ha ritornato il primo inserito (walker 0). Questa è una **scelta arbitraria** non comunicata all'utente: la confidence 33% è il segnale corretto, ma il "winner_label = early-return-cascade" può sembrare più certo di quanto sia. Fix proposto: quando confidence < THRESHOLD, mostrare TUTTE le opzioni equivalenti ("3-way tie, considera review manuale").

3. **Cloning mechanism non triggerato dalla logica della task** — Severity 1 metodologico.
   FizzBuzz è troppo semplice: tutti i walker raggiungono R≈5.0 (5/5 test, similar diff). VR differisce solo per il termine D (random partner). ESS resta alto → cloning skip. Il PIPELINE del cloning è stato testato (chiamate al codice corretto, branch logici percorsi) ma il **comportamento di selezione** (clone da walker povero a walker ricco) non è osservabile su una task così piatta. Per validare cloning dinamico serve task dove almeno un walker FALLISCE (R=0 da `compile_ok=False` o test rotti). Confermato il caveat della review pre-test.

4. **Continuation walker producono lavoro accessorio** — Severity 3 (osservazione).
   I tick 1 e 2 hanno aggiunto: docstring, type hints, type aliases, `Final` constants. Lavoro non strettamente necessario per soddisfare il goal. Su FizzBuzz è padding; su task più complesse questi step potrebbero essere informativi (es. estrazione di helper). Per smoke test va bene. Per benchmark serio va calibrato il prompt continuation.

5. **Phase 6 finale: cloning_skipped=False ma 0 clone applicate** — Severity 3.
   ESS al tick 2 = 1.735 < 2.10, quindi il sistema ha valutato il cloning. Ma per **fortuna stocastica** del partner pairing (seed=271), nessun walker ha pescato un partner con VR sufficientemente più alto da triggerare un clone. Questo è comportamento corretto del paper §4.4 — la stocasticità è la feature, non il bug. Lo segnalo perché è interessante: con altro seed i risultati potrebbero variare.

## Verdict

- ✅ Plugin gira end-to-end senza intervento **bloccante**
- ✅ 5/5 test acceptance passano su main dopo cherry-pick
- ✅ State machine funziona: 3 tick × {step, apply-clones} corretti
- ✅ ESS-adaptive cloning attivato correttamente (skip ai tick 0 e 1)
- ✅ init_commit_sha tracking attraverso i tick funziona
- ✅ Cherry-pick e cleanup worktree corretti
- ⚠️ Confidence 33% (corretto matematicamente, ma orchestrator dovrebbe avvertire utente)
- ⚠️ Git identity setup imperfetto nei worktree (workaround inline funziona)
- ⚠️ FizzBuzz troppo semplice per testare cloning dinamico

## Notes for next iteration

Bug da fixare prima del prossimo test:

1. `setup_smoke_repo.sh` deve configurare git user globalmente o nel repo path
2. Quando `confidence < ess_threshold/N` (= bincount uniforme), il decide output dovrebbe avere `is_tie: true` flag esplicito
3. Per il prossimo test usare task dove almeno una strategia genuinamente fallisce (così il cloning si triggera). Candidato: task con un edge case sottile che alcuni walker missano (es. handling Unicode in un parser).

## Conclusione onesta

Il plugin **funziona** end-to-end nel suo flusso happy-path. Il smoke test ha confermato:
- Setup → init → spawn parallelo → record → step → ESS → apply-clones → continuation → decide → cherry-pick è corretto
- I test acceptance passano

Il smoke test **non ha potuto** validare:
- Comportamento cloning dinamico (necessita task con varianza R reale)
- Walker dead handling (necessita task dove `compile_ok=False` può capitare)
- Goal-judge sub-agent call (saltato qui — `R_goal` default 0.7 usato)

Per dimostrare *valore* (non solo *funzionamento*) servirebbero le condizioni della Severità 1.4 dell'review precedente: task dove il single-shot baseline a volte fallisce e il plugin invece converge correttamente.

Per ora: il plugin è **certified runnable**. Pronto per il prossimo livello di test (task harder con baseline reale di confronto).
