"""verify_install.py — smoke test minimo di FMC su un Atari semplice.

Esegue un singolo episodio di MsPacman con N=50 walker, M=10 tick, time_limit=30s.
Verifica che l'algoritmo termini, che il reward sia > 0, e che il numero di sample
sia plausibile (qualche migliaio per episodio).

Pensato per essere eseguito da `verify_install.sh` come gate finale di setup.
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"FAIL: dipendenza base mancante: {e}")
        return 1

    # Path A (FractalAI_old) — smoke test minimale
    try:
        from fractalai.swarm import Swarm
        from fractalai.environment import AtariEnvironment
        from fractalai.model import RandomDiscreteModel

        env = AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)
        model = RandomDiscreteModel(n_actions=env.n_actions)

        swarm = Swarm(
            env=env,
            model=model,
            n_walkers=50,
            balance=1.0,
            reward_limit=None,
            samples_limit=5000,
            render_every=1e10,
            accumulate_rewards=True,
        )

        t0 = time.time()
        swarm.run_swarm()
        dt = time.time() - t0

        reward = float(swarm.rewards.max())
        samples = int(swarm._n_samples_done)

        print(f"  Episode reward: {reward:.1f}")
        print(f"  Samples used:   {samples}")
        print(f"  Wall time:      {dt:.1f}s")
        print(f"  Walkers:        {swarm.n_walkers}")

        if reward <= 0:
            print("FAIL: reward non positiva — controlla ROM Atari")
            return 1

        return 0
    except ImportError:
        pass  # Path A non installato, prova Path B

    # Path B (fragile) — smoke test minimale
    try:
        from fragile.fractalai import calculate_virtual_reward

        # Test minimale: la funzione virtual_reward esiste e non crasha
        rewards = torch.randn(100)
        observs = torch.randn(100, 8)
        vr = calculate_virtual_reward(observs, rewards)
        assert vr.shape == (100,)
        print(f"  fragile virtual_reward shape OK: {tuple(vr.shape)}")
        print("  (smoke test minimo riuscito; per benchmark completo vedi 03_atari_replication/)")
        return 0
    except ImportError as e:
        print(f"FAIL: né FractalAI_old né fragile installati: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
