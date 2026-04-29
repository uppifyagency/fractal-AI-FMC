# work/08_simulators — FMC su plangym envs

> Sostituisce le sim HTML statiche di [`simulations/`](../../simulations/) con env Python che usano lo stack ufficiale FragileTech (`plangym + fragile`). Vedi [`docs/architecture/tier1_repos_teardown.md`](../../docs/architecture/tier1_repos_teardown.md) per il razionale.

## Quick start — far girare FMC su uno dei loro env (1 comando)

```bash
cd work/08_simulators
python3 run_plangym_fmc.py --env CartPole-v1 --gif cartpole.gif
```

Output: `cum_reward ≈ 298 / 500`, GIF di 299 frame con il pole tenuto in equilibrio.

Altri env supportati out-of-the-box (built-in di plangym, nessun setup):

| Env | Comando | Note |
|---|---|---|
| **CartPole-v1** | `python3 run_plangym_fmc.py --env CartPole-v1 --n_walkers 50 --time_horizon 15` | classico, 2.9s wall |
| **MountainCar-v0** | `python3 run_plangym_fmc.py --env MountainCar-v0 --n_walkers 50 --time_horizon 30` | reward sparso |
| **Acrobot-v1** | `python3 run_plangym_fmc.py --env Acrobot-v1 --n_walkers 80 --time_horizon 25` | 2-link pendulum |
| **ALE/Boxing-v5** | `python3 run_plangym_fmc.py --env ALE/Boxing-v5 --n_walkers 30 --time_horizon 10 --max_steps 200` | Sergio's signature demo |
| **ALE/Pong-v5** | `python3 run_plangym_fmc.py --env ALE/Pong-v5 --n_walkers 50 --time_horizon 15` | |
| **LunarLander-v3** | richiede `pip install swig && pip install "gymnasium[box2d]"` | |
| **DM Control walker** | richiede `pip install dm_control mujoco` | |

## Cos'è "il loro stack"

Lo stesso che gira nel video-seminario di Sergio e in [`fragile/app/_plangym.py`](../../repos/fragile/src/fragile/app/_plangym.py):

```
+----------------------+      +-----------------------+
|   plangym.PlanEnv    | <--- |  fmc_swarm.FMCSwarm   |
|   (env state save/   |      |  (clones+virtual rew) |
|   restore + step)    |      |  ~120 LOC NumPy       |
+----------------------+      +-----------------------+
   ^                                ^
   |                                |
plangym.make(...)              FMCConfig(n_walkers=...)
```

- `plangym` (cloned in [`repos/plangym/`](../../repos/plangym/)) fornisce gli env.
- [`rocket_hook/fmc_swarm.py`](rocket_hook/fmc_swarm.py) implementa lo swarm in NumPy puro (~200 LOC, paper §4.4-§4.5). Non richiede torch/panel/holoviews — basta numpy + plangym.
- Per il path "completo" con `fragile.FractalTree` (torch + dashboard live), vedi commenti in `run_plangym_fmc.py`.

## Layout

```
work/08_simulators/
├── README.md                  ← questo file
├── run_plangym_fmc.py         ← entrypoint principale (FMC su un built-in env)
├── cartpole.gif               ← run output (CartPole)
├── boxing.gif                 ← run output (Boxing)
└── rocket_hook/               ← env custom Sergio's F23 demo (rocket + uncino + sasso)
    ├── env.py                 ← PlanEnv subclass, ~250 LOC fisica esplicita
    ├── render.py              ← rasterizer 64×64 RGB
    ├── fmc_swarm.py           ← FMC swarm NumPy (riusato da run_plangym_fmc.py)
    ├── run_fmc.py             ← demo standalone su RocketHookEnv
    └── tests/                 ← 15 pytest verdi
        └── test_env.py
```

## Stato del rocket_hook custom env

L'env [`rocket_hook/env.py`](rocket_hook/env.py) è una **interpretazione fedele al video F23 di Sergio** — fisica documentata in [`work/02_deep_dives/08_video_seminar_extracted_insights.md`](../02_deep_dives/08_video_seminar_extracted_insights.md) §F23. Test passano (15/15), reward bipartito esattamente come Sergio descrive (`R = 1/dist(hook, stone)` poi `R = 1/dist(hook, target)`).

> ⚠️ **Onestà**: Sergio non ha mai pubblicato il codice di quella demo specifica. La fisica è ricostruita dal video. Per replicare visivamente "la roba che funziona perfettamente come nel video" servirebbe:
> - 1 000–10 000 walker (vedi `_plangym.py`: `n_walkers=10000`), non 100
> - `fragile.FractalTree` con `dt_sampler` adattivo, non il NumPy minimal
> - costanti fisiche tunate dal team Hernández-Cerezo (mai pubblicate)
>
> Per il **path canonico FragileTech** validato (= roba che gira in 2 secondi e funziona): usa `run_plangym_fmc.py --env CartPole-v1` o uno degli env built-in elencati sopra.

## Test

```bash
cd work/08_simulators/rocket_hook
python3 -m pytest tests/ -v
# → 15 passed in ~0.3s
```

Test coverage:
- **PlanEnv contract**: setup spaces, reset, get/set_state round-trip, set_state determinism, step_batch signature
- **Physics**: gravità sasso, thrust solleva razzo, auto-grab, auto-deposit, crash, truncation, reward positivo
- **Rendering**: shape e dtype dell'immagine, img_shape property
- **Packing**: pack/unpack dello stato

## Estendere — ricetta nuovo simulatore

1. **Erediti `PlanEnv`** in `work/08_simulators/<nome>/env.py` (~80-150 LOC). I 4 metodi obbligatori:
   ```python
   def apply_reset(self, **kwargs) -> tuple[obs, info]: ...
   def apply_action(self, action) -> tuple[obs, reward, terminal, truncated, info]: ...
   def get_state(self) -> np.ndarray: ...
   def set_state(self, state: np.ndarray) -> None: ...
   ```
2. **Definisci** `observation_space` e `action_space` dentro `setup()`.
3. **Test** copiando `rocket_hook/tests/test_env.py` come template.
4. **Run FMC** con:
   ```python
   from fmc_swarm import FMCSwarm, FMCConfig, run_episode
   env = MyEnv(...)
   result = run_episode(env, FMCConfig(n_walkers=200, time_horizon=30), seed=42, verbose=True)
   ```
5. **Bonus**: dashboard live via `shaolin.StreamingPlot` (vedi `repos/fragile/src/fragile/dataviz.py` `MontezumaDisplay`).

## Riferimenti

- Teardown architetturale: [`docs/architecture/tier1_repos_teardown.md`](../../docs/architecture/tier1_repos_teardown.md)
- Plangym docs: [`repos/plangym/CLAUDE.md`](../../repos/plangym/CLAUDE.md)
- Fragile FractalTree: [`repos/fragile/src/fragile/core.py`](../../repos/fragile/src/fragile/core.py) (linea 458 — `class FractalTree`)
- Esempio idiomatico Sergio: [`repos/fragile/src/fragile/app/_plangym.py`](../../repos/fragile/src/fragile/app/_plangym.py)
- Demo F23 (rocket-uncino) descritta nel seminar: [`work/02_deep_dives/08_video_seminar_extracted_insights.md`](../02_deep_dives/08_video_seminar_extracted_insights.md) §F23
- FMC algoritmo NumPy alternativo (Atari): [`work/03_atari_replication/scripts/fmc_minimal.py`](../03_atari_replication/scripts/fmc_minimal.py)
