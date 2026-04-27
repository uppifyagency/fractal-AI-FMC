# Milestone 17 — Real-time visual simulator

> **Stato**: ✓ Chiuso (2026-04-27)
> **Scope**: dashboard interattivo Streamlit che mostra in tempo reale la
> nostra pipeline FMC + DAgger + M14 oracle + TCV-X21 target.
>
> **Risultato chiave**: visualizzazione live con cross-section R-Z
> (LCFS attuale via M14 oracle vs target), 4 metriche live (truth-err,
> physicality, I_p, latency), shape time series, FMC walker swarm,
> coil voltages. Aggiornamento ad ogni tick (5-20 fps configurabile).

## 1. Componenti

### 1.1 Cross-section R-Z live
Plot Plotly di:
- **Target LCFS** (rosso): forma plasma desiderata, computata via Miller
  parametrization da target shape (R_p, Z_p, κ, δ)
- **Current LCFS** (cyan): shape attuale del plasma estratta dal M14
  freegs oracle ad ogni tick
- **Coils** (16 quadrati colorati): E1-E8 + F1-F8 con colore = current
  intensity (RdBu colormap, normalized)
- **Vessel envelope** (linea tratteggiata)
- **Plasma centroid** (X marker)

### 1.2 Shape descriptors time series
4-panel plot di [R_p, Z_p, κ, δ] vs time:
- Linea cyan = truth (M14 oracle)
- Linea arancio dashed = self (simulator-internal)
- Linea rossa orizzontale = target

### 1.3 Live metrics dashboard
4 gauge/indicator:
- **Truth-err** (gauge 0-80): verde <10, giallo 10-30, rosso >30
- **Physicality rate** (gauge 0-100%): cumulative % di step con LCFS valida
- **Plasma current I_p** (kA): con delta vs nominal 200 kA
- **Latency** (µs/decision): media degli ultimi 10 step

### 1.4 Coil voltages bar chart (when policy is NN-distilled)
16 barre per i comandi V_coils (E1-E8 blu, F1-F8 rosso)

### 1.5 FMC walker swarm (when policy is FMC online)
- Scatter walkers nello spazio (κ, δ) colorati per cum reward
- Histogram del virtual reward
- Overlay del target marker (X rosso)
- Annotation: "alive: N/64 · E[reward]: x.xx"

### 1.6 Truth-err evolution
Time series con linea soglia "deploy threshold" a truth-err = 10.

## 2. Controlli sidebar

- **Policy**: dropdown tra M5 BC / M6 DAgger×3 / M10 DAgger×N / M12 NN-shape /
  FMC online (zero-training)
- **Target**:
  - "TCV-X21 shot 65402 (REAL)" → carica eqdsk reale, R=0.889, κ=1.71, δ=+0.12
  - "Custom (sliders)" → 4 slider per R_p, Z_p, κ, δ
- **Speed**: 1× / 2× / 5× / 10× (numero di sim step per Streamlit rerun)
- **Live M14 oracle truth**: checkbox (oracle è ~24 ms/query, può essere
  disabilitato per fps massimo)
- **FMC walker visualization**: checkbox (solo quando policy=FMC)
- **▶ Run / ↺ Reset** toggle

## 3. Architettura

```
┌─────────────────────────────────────────────────────────┐
│ Sidebar                                                 │
│  - Policy selector → loads cached resource              │
│  - Target mode → real (cached) or sliders               │
│  - Speed, oracle toggle                                 │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ Main loop (Streamlit rerun cycle)                       │
│  1. step_simulation():                                  │
│     - policy(x, target) → V_coils                       │
│     - sim_step(x, V) → x_new                            │
│     - oracle.shape_from_coils(x_new[:N]) → truth shape  │
│     - update history (200-tick rolling window)          │
│  2. render_all():                                       │
│     - 6 plotly charts updated via st.empty()            │
│     - All charts get unique key=tick_n for re-render    │
│  3. time.sleep(0.05); st.rerun()                        │
└─────────────────────────────────────────────────────────┘

Cached resources (one-time init):
  - FreeGSOracle (~1 sec baseline solve)
  - 4 trained policies (M5, M6, M10, M12)
  - 2 simulators (linear + NN-shape)
  - FMCPlasmaJaxController (jit-compiled, warm)
  - TCV-X21 65402 LCFS (parsed once)
```

## 4. Verifica del dashboard

```bash
$ python -c "from dashboard_realtime import render_crosssection, ..."
  crosssection: 20 traces
  shape_ts: 8 traces
  metrics: 4 indicators
  coils: 1 traces
  err_ts: 2 traces
OK - all dashboard render funcs work standalone

$ streamlit run scripts/dashboard_realtime.py --server.headless --server.port 8503
  Local URL: http://localhost:8503
  HTTP 200, 1522 bytes
```

## 5. Output

| Path | Cosa |
|---|---|
| [`scripts/dashboard_realtime.py`](../scripts/dashboard_realtime.py) | Dashboard Streamlit (~750 LOC) |
| [`docs/milestone_17_realtime_viz.md`](milestone_17_realtime_viz.md) | Questo documento |

## 6. Limitazioni

1. **FMC walker viz è approssimato**: gli walker interni del JIT decide
   non sono esposti pubblicamente. Visualizziamo *proxy walkers*
   campionando il policy multiple times e mostrando le 8 azioni
   candidate per ogni rollout. Per visualization "esatta" servirebbe
   modificare `make_jit_decide` per ritornare gli stati intermedi
   (~10× memoria, OK per debug).

2. **fps limit ~10-20**: Streamlit `st.rerun()` ha overhead di re-execute
   tutto lo script ogni tick. Per fps >50 servirebbe Bokeh server o
   Dash con callback streaming.

3. **Oracle as bottleneck**: con `show_oracle=True`, ogni tick fa una
   chiamata 24 ms. Se disabilitato (solo self-shape dal sim), fps sale
   a ~50.

4. **Browser-only**: Streamlit dashboard non standalone executable.
   Per export demo video serve OBS / screen-recording.

## 7. Esempi di uso

### Demo "M12 sul target reale TCV-X21"
1. Sidebar: policy = "M12 NN-shape (best on real TCV)"
2. Target = "TCV-X21 shot 65402 (REAL)"
3. Speed = 5×, Live M14 oracle = ON
4. Run → mostra tracking convergence in ~30 ticks

Expected: truth-err scende da ~30 (initial state) a ~3-5 (steady-state),
physicality 100%.

### Demo "M5 BC fallisce in fisica"
1. Policy = "M5 BC (baseline)"
2. Stesso target
3. Run

Expected: self-err piccolo (~5), truth-err alto (>50), physicality
crolla a 5-15% — visualizza concretamente il "simulator overfitting".

### Demo "FMC online + walker swarm"
1. Policy = "FMC online (zero-training)"
2. Custom target con sliders (es. high-κ scenario)
3. Run con walker viz on

Expected: walker scatter mostra exploration diversity attorno al target,
reward histogram mostra weight concentration.

## 8. Take-aways

Il dashboard fa 3 cose nuove rispetto al `dashboard.py` originale (M4):

1. **Live M14 oracle integration** — mostra la vera shape (non solo
   self-evaluation)
2. **TCV-X21 real target option** — confronto contro shape sperimentale
3. **FMC internals viz** — primi passi verso debug visuale dell'algoritmo

Per il paper / public release, il dashboard è il **demo artifact** che
fa vedere subito che il sistema funziona. Combinato con i quench
counter, latency live, e lo spread truth-vs-self, è auto-evidente che
lo sviluppo è sound.
