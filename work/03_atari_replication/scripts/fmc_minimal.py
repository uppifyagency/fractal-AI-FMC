"""fmc_minimal.py — implementazione standalone di Fractal Monte Carlo per Atari.

Algoritmo dal paper Hernández-Cerezo & Duran-Ballester (2020), §4.3.
Reimplementato da zero usando gymnasium + ALE state cloning, per evitare il
porting del FractalAI_old (basato su gym vecchio) e per essere verificabile.

Caratteristiche:
- Funziona su qualsiasi ALE/<game>-v5 di gymnasium
- Usa ALE.cloneState() per snapshot/restore degli stati dei walker
- Distanza: L2 sulla RAM (128 bytes) — il paper sezione 5.1.3.3 mostra +61% vs IMG
- Single-threaded, NumPy puro — basta per smoke test

Uso:
    python fmc_minimal.py --game ALE/Boxing-v5 --n_walkers 30 --time_horizon 15 \\
                          --max_samples 5000 --seed 42

Output (JSON su stdout):
    {"reward": 99.0, "samples": 4123, "wall_time_s": 287.4, "n_steps": 850}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

import numpy as np


def relativize(x: np.ndarray) -> np.ndarray:
    """Trasforma vettore in valori positivi conservando ordinamento.

    Implementa §2.2.3 del paper:
        R_N = (R - μ) / σ
        R = exp(R_N)         se R_N ≤ 0
        R = 1 + ln(1 + R_N)  se R_N > 0
    """
    std = float(x.std())
    if std == 0:
        return np.ones_like(x, dtype=np.float32)
    standard = (x - x.mean()) / std
    out = np.where(
        standard <= 0,
        np.exp(np.clip(standard, -50, 0)),
        1 + np.log1p(np.maximum(standard, 0)),
    )
    return out.astype(np.float32)


@dataclass
class FMCConfig:
    n_walkers: int = 30
    time_horizon: int = 15           # M tick per pianificazione
    max_samples_per_action: int = 300
    balance: float = 1.0             # alpha = beta = balance
    fixed_steps: int = 5             # skipframe (frame stack)


class FMCAgent:
    """Implementazione minima di FMC per Atari via gymnasium ALE."""

    def __init__(self, env, config: FMCConfig):
        self.env = env
        self.ale = env.unwrapped.ale
        self.cfg = config
        self.n_actions = env.action_space.n
        self.rng = np.random.default_rng()

    def _snapshot_state(self):
        """Salva lo stato corrente dell'ALE (clone deep)."""
        return self.ale.cloneState()

    def _restore_state(self, state):
        """Ripristina lo stato dell'ALE."""
        self.ale.restoreState(state)

    def _ram_obs(self) -> np.ndarray:
        """RAM 128-byte come osservazione (paper: più informativa di IMG)."""
        return self.ale.getRAM().astype(np.float32)

    def _step_skipframe(self, action: int) -> tuple[float, bool]:
        """Esegui `action` per fixed_steps frame (skipframe), return (reward_sum, terminal)."""
        total = 0.0
        terminal = False
        for _ in range(self.cfg.fixed_steps):
            r = self.ale.act(int(action))
            total += float(r)
            if self.ale.game_over():
                terminal = True
                break
        return total, terminal

    def decide(self, root_state) -> int:
        """Calcola la decisione FMC dato lo stato attuale dell'env.

        Algoritmo (paper §4.3):
        1. Inizializza N walker copiando root_state, ognuno con initial_decision random
        2. Per M tick:
           a. Ogni walker prende la propria initial_decision (al primo tick) o random
           b. Step the simulator
           c. Calcola virtual reward = R · Dist (entrambi relativized)
           d. Cloning: ogni walker confronta sé con un random partner, decide se clonare
        3. Decisione finale = bincount(initial_decisions).argmax()
        """
        N = self.cfg.n_walkers
        M = self.cfg.time_horizon

        # Inizializzazione: tutti i walker partono da root_state
        init_actions = self.rng.integers(0, self.n_actions, size=N)
        cum_rewards = np.zeros(N, dtype=np.float32)
        is_dead = np.zeros(N, dtype=bool)

        # Snapshot iniziale: clona lo stato N volte
        walker_states = [self._snapshot_state() for _ in range(N)]
        # ^ root_state è già attivo nell'ALE, gli snapshot sono identici
        # ma servono come "memoria" per ogni walker dopo che evolveranno

        # Loop di pianificazione
        for t in range(M):
            for i in range(N):
                if is_dead[i]:
                    continue
                self._restore_state(walker_states[i])
                # Tick 0: usa initial_decision; tick > 0: random
                a = init_actions[i] if t == 0 else self.rng.integers(0, self.n_actions)
                r, term = self._step_skipframe(a)
                cum_rewards[i] += r
                if term:
                    is_dead[i] = True
                else:
                    walker_states[i] = self._snapshot_state()

            # Calcola reward + distanza + virtual reward
            obs_ram = np.zeros((N, 128), dtype=np.float32)
            for i in range(N):
                if not is_dead[i]:
                    self._restore_state(walker_states[i])
                    obs_ram[i] = self._ram_obs()

            # Distanza: ogni walker confronta con un compagno random
            partners = self.rng.permutation(N)
            same = partners == np.arange(N)
            partners[same] = (partners[same] + 1) % N  # evita j == i

            distances = np.linalg.norm(obs_ram - obs_ram[partners], axis=1)

            # Relativize entrambi
            R_norm = relativize(cum_rewards)
            D_norm = relativize(distances)

            # Penalizza i morti azzerando la virtual reward
            R_norm[is_dead] = 0
            D_norm[is_dead] = 0

            # Virtual reward
            VR = (R_norm ** self.cfg.balance) * (D_norm ** self.cfg.balance)

            # Cloning: confronta con un altro partner random
            clone_partners = self.rng.permutation(N)
            same = clone_partners == np.arange(N)
            clone_partners[same] = (clone_partners[same] + 1) % N

            VR_self = VR
            VR_other = VR[clone_partners]
            denom = np.where(VR_self > 1e-8, VR_self, 1e-8)
            clone_prob = np.clip((VR_other - VR_self) / denom, 0, 1)
            # Walker morti clonano sempre
            clone_prob[is_dead] = 1.0

            random_draws = self.rng.random(N)
            will_clone = random_draws < clone_prob

            # Applica i clone
            for i in np.where(will_clone)[0]:
                k = clone_partners[i]
                if not is_dead[k]:
                    walker_states[i] = walker_states[k]  # ALEState è copiabile per ref
                    # Ri-snapshot per evitare aliasing (ALEState è mutevole su restore)
                    self._restore_state(walker_states[i])
                    walker_states[i] = self._snapshot_state()
                    init_actions[i] = init_actions[k]
                    cum_rewards[i] = cum_rewards[k]
                    is_dead[i] = False

        # Decisione: argmax bincount sulle initial_decision dei walker vivi
        alive_ids = init_actions[~is_dead]
        if len(alive_ids) == 0:
            return int(self.rng.integers(0, self.n_actions))
        counts = np.bincount(alive_ids, minlength=self.n_actions)
        return int(counts.argmax())


def run_episode(game: str, config: FMCConfig, seed: int, max_steps: int = 27000,
                reward_limit: float | None = None, verbose: bool = False) -> dict:
    """Esegui un episodio completo con FMC."""
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)

    env = gym.make(game, full_action_space=False)  # action_space ridotto al gioco
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)

    agent = FMCAgent(env, config)
    agent.rng = np.random.default_rng(seed)

    cum_reward = 0.0
    samples_total = 0
    n_decisions = 0
    t_start = time.time()

    for step in range(max_steps):
        # Salva lo stato attuale del "vero" gioco
        root_state = agent._snapshot_state()
        # Pianifica
        action = agent.decide(root_state)
        # Ripristina e applica l'azione decisa nel gioco vero
        agent._restore_state(root_state)
        r, term = agent._step_skipframe(action)
        cum_reward += r

        # Stima dei sample usati: walker × tick × skipframe
        samples_total += config.n_walkers * config.time_horizon * config.fixed_steps
        n_decisions += 1

        if verbose and n_decisions % 10 == 0:
            print(f"  step {n_decisions}: action={action} reward={cum_reward:.0f} "
                  f"samples={samples_total} elapsed={time.time()-t_start:.1f}s",
                  file=sys.stderr)

        if term or (reward_limit is not None and cum_reward >= reward_limit):
            break

    wall = time.time() - t_start
    env.close()

    return {
        "reward": float(cum_reward),
        "samples": int(samples_total),
        "wall_time_s": float(wall),
        "n_steps": n_decisions,
        "samples_per_action_avg": samples_total / max(1, n_decisions),
        "terminated": bool(term),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ALE/Boxing-v5")
    ap.add_argument("--n_walkers", type=int, default=30)
    ap.add_argument("--time_horizon", type=int, default=15)
    ap.add_argument("--max_samples", type=int, default=300, help="cap di sample per azione")
    ap.add_argument("--balance", type=float, default=1.0)
    ap.add_argument("--fixed_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=27000)
    ap.add_argument("--reward_limit", type=float, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    config = FMCConfig(
        n_walkers=args.n_walkers,
        time_horizon=args.time_horizon,
        max_samples_per_action=args.max_samples,
        balance=args.balance,
        fixed_steps=args.fixed_steps,
    )

    result = run_episode(
        game=args.game,
        config=config,
        seed=args.seed,
        max_steps=args.max_steps,
        reward_limit=args.reward_limit,
        verbose=args.verbose,
    )
    result["game"] = args.game
    result["seed"] = args.seed
    result["config"] = config.__dict__

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
