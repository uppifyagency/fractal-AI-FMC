# Tier 1 FragileTech repos — teardown architetturale

> **Stato**: scritto 2026-04-28.
> **Scope**: 4 repo aggiunti a [`repos/`](../../repos/) come dipendenze hard di [`fragile`](../../repos/fragile/) e [`fragile-rl`](../../repos/fragile-rl/).
> **Obiettivo**: documentare cosa hanno implementato Sergio & co., come si compone lo stack, e perché va usato come **impalcatura** per i nostri simulatori al posto delle pagine HTML statiche in [`simulations/`](../../simulations/).

## 0. TL;DR

| Repo | Ruolo nel mondo FMC | LOC core | Indispensabile? |
|---|---|---|---|
| **plangym** | Estende Gymnasium con `get_state()` / `set_state()` atomico → permette rollout deterministici da stati arbitrari → **fa esistere FMC** | ~3 000 | ⭐⭐⭐ Sì |
| **shaolin** | Dashboard live (`holoviews + panel + bokeh`) per visualizzare swarm walker mentre gira | ~1 200 | ⭐⭐ Per demo |
| **hydraclick** | `Hydra config + Click CLI` — un solo entrypoint per tutti gli sweep di iperparametri | ~600 | ⭐ Quality-of-life |
| **flogging** | Logging strutturato JSON-line + human-readable colorato | ~150 | ⭐ Quality-of-life |

Smoking gun: [`fragile/core.py:716`](../../repos/fragile/src/fragile/core.py) chiama `self.env.step_batch(states=..., actions=..., dt=...)` e [`core.py:839`](../../repos/fragile/src/fragile/core.py) chiama `env.set_state(state)`. Questi sono i due call-site che fanno girare l'algoritmo FMC su qualunque ambiente.

---

## 1. plangym — il tier che importa davvero

### Cosa è (in una riga)

> *Plangym estende Gymnasium con la capacità di **`get_state()` / `set_state()`** atomico — abilita rollout deterministici da stati arbitrari. È il prerequisito che fa esistere FMC.*

Riferimento autore: [`repos/plangym/CLAUDE.md`](../../repos/plangym/CLAUDE.md) (file scritto da Sergio per Claude Code).

### Architettura (8 backend pronti)

```
PlanEnv (ABC, core.py:17)
└── PlangymEnv (gym wrapper, core.py:492)
    ├── ClassicControl   (CartPole, MountainCar, Acrobot, Pendulum)
    ├── Box2DEnv         (LunarLander, BipedalWalker — serializza b2World!)
    ├── DMControlEnv     (DeepMind Control Suite, MuJoCo)
    ├── MujocoEnv        (Ant, HalfCheetah, Humanoid — usa mjSTATE_INTEGRATION)
    ├── BalloonEnv       (Google Loon stratospheric balloon)
    ├── AtariEnv         (ALE — usa cloneSystemState)
    ├── RetroEnv         (Sega Genesis, NES via stable-retro)
    ├── MarioEnv         (Super Mario Bros via nes-py)
    └── MontezumaEnv     (con room tracking custom)

VectorizedEnv (vectorization/env.py)
├── ParallelEnv  (multiprocessing.Pipe — n_workers locali)
└── RayEnv       (Ray distributed — cluster scaling)
```

### L'API che fa funzionare FMC (4 metodi astratti)

`PlanEnv` è 854 righe ma il contratto reale è **4 metodi astratti** (vedi [`core.py:464-489`](../../repos/plangym/src/plangym/core.py)):

```python
def apply_action(self, action): ...   # avanza di 1 tick
def apply_reset(self): ...
def get_state(self) -> Any: ...        # snapshot completo serializzabile
def set_state(self, state) -> None:    # restore atomico
```

E un solo metodo pubblico load-bearing per FMC ([`core.py:205-242`](../../repos/plangym/src/plangym/core.py)):

```python
def step_batch(self, actions, states=None, dt=1, return_state=True):
    """Vettorizza N walker in parallelo: ogni walker ha (action, state, dt)."""
```

Questo è **il punto di contatto** con [`fragile/core.py:716`](../../repos/fragile/src/fragile/core.py) — l'intera dinamica di clone FMC si riduce a:

```python
# pseudo-code di un tick FMC
new_states, obs, rewards, ends, _, info = env.step_batch(
    states=walker_states,           # N stati clonati pre-clone
    actions=sample_actions(),       # perturbazioni stocastiche
    dt=frameskip * tau,
)
# poi: virtual reward → clone pairwise → ripeti
```

### Cosa rende non-banale `get_state` / `set_state` per ogni dominio

Ogni env risolve un problema di serializzazione diverso:

| Env | Strategia | LOC | Trick |
|---|---|---|---|
| **Atari** ([`videogames/atari.py`](../../repos/plangym/src/plangym/videogames/atari.py)) | `ale.cloneSystemState()` + flag `clone_seeds` | 368 | Determinismo opzionale (con/senza RNG state) |
| **MuJoCo** ([`control/mujoco.py`](../../repos/plangym/src/plangym/control/mujoco.py)) | `mjSTATE_INTEGRATION` + `mj_forward()` per ricomputare quantità derivate | 111 | **Cattura warmstart del solver** — senza questo i rollout divergono |
| **Box2D** ([`control/box_2d.py`](../../repos/plangym/src/plangym/control/box_2d.py)) | Walk del `b2World`: serializza ogni `body` (mass, transform, velocity, ...) in dict | 157 | Custom — Box2D non ha snapshot nativo |
| **DM Control** | Internal state vector di MuJoCo + override task | — | Stesso pattern di MujocoEnv |
| **Balloon** ([`control/balloon.py`](../../repos/plangym/src/plangym/control/balloon.py)) | `arena.set_simulator_state()` di Google BLE | 72 | State *non* è array → flag `STATE_IS_ARRAY=False` |

> **Implicazione per i nostri simulatori**: per ogni nuovo simulatore custom che vogliamo guidare con FMC, dobbiamo solo implementare `get_state` / `set_state` / `apply_action` / `apply_reset`. Sono ~80-150 LOC per env.

### Vectorization gratuita

`ParallelEnv` ([`vectorization/parallel.py`](../../repos/plangym/src/plangym/vectorization/parallel.py), 512 LOC) sharda N walker su processi separati via `multiprocessing.Pipe`. `RayEnv` lo stesso ma su cluster. Già pronto, non serve scriverlo: l'ABI di `step_batch` è preservata.

### Tier-down: cosa hanno implementato vs cosa abbiamo noi

| Funzionalità | plangym | nostro [`fmc-core/`](../../fmc-core/) | nostro [`work/03_atari_replication/`](../../work/03_atari_replication/) |
|---|---|---|---|
| State save/restore | ✅ 7 backend | ❌ N/A (toy) | ⚠️ ALE diretto, no abstraction |
| `step_batch` parallelo | ✅ Pipe + Ray | ❌ | ❌ seriale |
| Frameskip + dt compositing | ✅ nativo | ⚠️ semplificato | ✅ |
| Headless rendering | ✅ (`MUJOCO_GL=egl`) | ❌ | ❌ |
| Wrapper composition | ✅ gym.Wrapper | ❌ | ❌ |
| Delayed setup (per worker) | ✅ | ❌ | ❌ |

---

## 2. shaolin — viz dashboard streaming

### Cosa è

Stack `holoviews + panel + bokeh` per **dashboard interattive che si aggiornano via stream live** (a differenza di matplotlib statico). Pensato per fare debug visuale di run FMC mentre girano.

### Moduli ([`repos/shaolin/src/shaolin/`](../../repos/shaolin/src/shaolin/))

| Modulo | LOC | Cosa fa |
|---|---|---|
| `stream_plots.py` | ~180 | `StreamingPlot` base — wrap di un holoviews plot con `Pipe`/`Buffer` per push live ([`stream_plots.py:13`](../../repos/shaolin/src/shaolin/stream_plots.py#L13)) |
| `graph.py` | ~250 | Plot di grafi NetworkX con `graphviz_layout` — **è quello che serve per visualizzare l'albero walker (F13 fractal tree)** |
| `dataframe.py` | ~200 | Scatter/line plot da pandas con hover tooltip + selezione |
| `dimension_mapper.py` | ~180 | Mappa colonne di un DF a dimensioni visuali (`SizeDim`, `ColorDim`, `AlphaDim`, `LineWidthDim`) |
| `colormaps.py` | ~50 | Catalogo `matplotlib + colorcet` |

### Come `fragile` la usa

> ⚠️ **Punto critico**: [`repos/fragile/src/fragile/shaolin/`](../../repos/fragile/src/fragile/shaolin/) contiene un **fork inlined** di shaolin (10 moduli) — non è la dep esterna. È la versione patchata che gira nei demo. Vale la pena fare un diff prima di modificare.

In [`fragile/dataviz.py`](../../repos/fragile/src/fragile/dataviz.py) c'è `MontezumaDisplay` che riceve `fai` (FractalAI swarm state) ad ogni tick e aggiorna dashboard di:
- best RGB frame
- heatmap di visite per stanza
- mappa rooms visitate

Esattamente quello che serve per i nostri simulatori.

---

## 3. hydraclick — CLI config-driven

### Cosa è

Wrapper minimo (~600 LOC totali) che fa **Click decorators + Hydra config** lavorare insieme. Un comando `@hydraclick.run("config_name")` ti dà:

- Override CLI: `python run.py +foo=bar swarm.n_walkers=500`
- Multi-run: sweep automatico su grid di config
- Shell completion gratis
- Display config strutturato a schermo prima del run

### Perché serve

`fragile-rl` ha decine di iperparametri (`n_walkers`, `balance`, `frameskip`, `env`, `dt`, ...). Senza Hydra finisci a hardcodare YAML. Con `hydraclick` un singolo CLI gestisce tutti gli sweep di benchmark.

---

## 4. flogging — logging strutturato

Una sola feature concettuale: switch tra **human-readable colorato** (dev) e **JSON line-delimited** (prod, per log aggregation). + `set_context()` per metadata propagati. + `log_multipart()` per messaggi grossi spezzati.

~150 LOC utili. Niente di esotico ma evita di reinventare il pattern in ogni run.

---

## 5. Come usare lo stack come **impalcatura al posto dell'HTML**

> *Oggi [`simulations/`](../../simulations/) ha sims HTML/JS/WebGPU (rocket, kart, pong, octopus, highway, SUMO, game-of-life). Sono carine per il blog, ma:*
> - *non parlano a fragile-rl*,
> - *non hanno save/restore di stato (quindi FMC non può girarci sopra)*,
> - *gli walker sono mockup visuale, non rollout veri.*

### La sostituzione che ha senso

```
[OGGI]
simulations/rocket.html                  ← matlab-style, visualizzazione decorativa
fmc-core/ (NumPy + JS port)              ← toy 2D, ok per blog ma non scala

[PROPOSTO]
work/08_simulators/{env_name}/env.py        ← env plangym custom (impl get/set_state)
work/08_simulators/{env_name}/run_fmc.py    ← run fragile FMC + shaolin dashboard live
work/08_simulators/{env_name}/web_export/   ← export Bokeh standalone (statico)
                                              generato da shaolin → riusabile blog
```

### Ricetta concreta per un nuovo simulatore

1. **Implementa l'env** ereditando `PlanEnv` (~80-150 LOC):
   ```python
   class FooEnv(PlanEnv):
       def apply_action(self, action): ...   # un tick fisica
       def get_state(self):
           return numpy.concatenate([self.pos, self.vel, [self.angle], ...])
       def set_state(self, state):           # restore atomico
           self.pos, self.vel, ... = unpack(state)
       def apply_reset(self): ...
   ```
2. **Registra** in `plangym.make()` o usa direttamente la classe.
3. **Run FMC** con `fragile.Swarm(env=FooEnv(), n_walkers=200, balance=1.0)`.
4. **Dashboard live** via `shaolin.StreamingPlot` con grafo dell'albero walker (`shaolin/graph.py`) e RGB del best walker (`stream_plots.RGB`).
5. **Export web** con `holoviews.save(plot, "demo.html")` — produce HTML standalone con Bokeh embedded, **interattivo**, ~400 KB. Si carica nel blog senza server.
6. **CLI** con `@hydraclick.run` — un solo entrypoint per tutti gli sweep di balance/τ/seed.

### Vantaggi rispetto all'HTML attuale

| Aspetto | HTML/JS oggi | Plangym + fragile + shaolin |
|---|---|---|
| FMC reale | ❌ mockup/animazione | ✅ rollout veri con N walker |
| Riproducibilità seed-deterministica | ⚠️ JS RNG | ✅ ALE/MuJoCo/Box2D state-based |
| Vectorization | ❌ 1 walker | ✅ ParallelEnv / Ray |
| Riuso codice fra demo | ❌ ogni sim duplica | ✅ stesso swarm, env diverso |
| Export blog | ✅ nativo | ✅ holoviews → HTML standalone |
| Ablation studies | ❌ infattibile | ✅ hydraclick sweep |
| Costo per nuovo sim | 500-1000 LOC JS | **~100 LOC Python (i 4 metodi PlanEnv)** |
| Compatibile con paper benchmarks | ❌ | ✅ stesso stack di Procgen/Crafter |

### Roadmap consigliata

1. **Pilot**: porta il `rocket-uncino` (la demo F23 più scenografica del video di Sergio) come `RocketHookEnv` plangym + dashboard shaolin → **vedi [`work/08_simulators/rocket_hook/`](../../work/08_simulators/rocket_hook/)**.
2. **Estendi**: kart, pong, highway → tutti come env custom. Riusi 90% del codice di run.
3. **Sostituisci**: una volta che 3-4 demo girano, deprechi le HTML in `simulations/` mantenendo solo il loro export Bokeh.
4. **Bonus**: `fragile-rl` ha già RLib hooks → puoi confrontare FMC vs PPO sui *tuoi* simulatori, non solo Atari.

---

## 6. Riferimenti

- Paper FMC: [`docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf`](../bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf)
- Plangym docs: [`repos/plangym/CLAUDE.md`](../../repos/plangym/CLAUDE.md)
- Fragile core: [`repos/fragile/src/fragile/core.py`](../../repos/fragile/src/fragile/core.py)
- Sergio's seminar (video): [`VideoTranscriptSergio.md`](../../VideoTranscriptSergio.md), insight estratti in [`work/02_deep_dives/08_video_seminar_extracted_insights.md`](../../work/02_deep_dives/08_video_seminar_extracted_insights.md)
- Demo F23 (rocket-uncino) descritta in seminar: vedi `08_video_seminar_extracted_insights.md` §F23
