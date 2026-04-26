"""run_single.py — esegue un singolo (game, seed) di FMC e salva un JSON di risultato.

Uso:
    python run_single.py --config ../configs/boxing.yaml --seed 42 --output ../results/boxing_seed42.json

Il JSON di output contiene:
    {
      "game": "boxing",
      "seed": 42,
      "config": {...},
      "result": {
        "reward": 99.0,
        "samples_used": 4123,
        "wall_time_s": 287.4,
        "n_steps": 850,
        "samples_per_action_avg": 4.85
      },
      "hardware": {...},
      "fragile_version": "...",
      "timestamp": "2026-04-26T18:50:00"
    }
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    with path.open() as fh:
        return yaml.safe_load(fh)


def hardware_info() -> dict[str, str]:
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": str(__import__("os").cpu_count()),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "absent"
    return info


def rom_checksum() -> dict[str, str]:
    """Best-effort checksum delle ROM Atari per replicabilità."""
    try:
        import ale_py
        rom_dir = Path(ale_py.__file__).parent / "roms"
        if rom_dir.exists():
            out = subprocess.run(
                ["md5sum", *list(rom_dir.glob("*.bin"))[:3]],
                capture_output=True, text=True
            )
            return {"sample_md5": out.stdout.strip().split("\n")[:3]}
    except Exception:
        pass
    return {"sample_md5": "unavailable"}


def run_fmc_episode(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Esegue un episodio FMC. Astrazione su FractalAI_old e fragile (autodetect)."""

    # Preferisci fragile (Path B) se installato
    try:
        return _run_with_fragile(config, seed)
    except (ImportError, ModuleNotFoundError):
        pass

    # Fallback su FractalAI_old (Path A)
    try:
        return _run_with_fractalai_old(config, seed)
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "Né `fragile` né `fractalai` (FractalAI_old) sono installati. "
            "Esegui prima 01_setup_environment/verify_install.sh"
        ) from e


def _run_with_fractalai_old(config: dict[str, Any], seed: int) -> dict[str, Any]:
    import numpy as np
    from fractalai.environment import AtariEnvironment
    from fractalai.fractalmc import FractalMC
    from fractalai.model import RandomDiscreteModel

    np.random.seed(seed)

    env_cfg = config["env"]
    plan_cfg = config["planner"]

    env = AtariEnvironment(
        name=env_cfg["name"].replace("ALE/", "").replace("-v5", "-v0"),
        clone_seeds=True,
        autoreset=True,
    )
    model = RandomDiscreteModel(n_actions=env.n_actions)

    fmc = FractalMC(
        env=env,
        model=model,
        n_walkers=plan_cfg["n_walkers"],
        balance=plan_cfg["balance"],
        time_horizon=plan_cfg["time_horizon"],
        max_samples_step=plan_cfg["max_samples_step"],
        reward_limit=plan_cfg.get("reward_limit"),
        can_win=plan_cfg.get("can_win", False),
    )

    t0 = time.time()
    history = fmc.run_agent()  # API specifica del repo
    wall = time.time() - t0

    return {
        "reward": float(np.sum([h.reward for h in history])),
        "samples_used": int(fmc._n_samples_done),
        "wall_time_s": wall,
        "n_steps": len(history),
        "samples_per_action_avg": fmc._n_samples_done / max(1, len(history)),
        "backend": "fractalai_old",
    }


def _run_with_fragile(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Stub: implementare quando fragile API è confermata.

    fragile usa `BaseFractalTree` con interfaccia diversa.
    Da specificare dopo verify_install.sh.
    """
    raise NotImplementedError("Implementazione fragile da completare in fase Step 2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    config = load_config(args.config)

    print(f"Running {args.config.stem} with seed {args.seed}...")
    result = run_fmc_episode(config, args.seed)
    print(f"  reward={result['reward']}  samples={result['samples_used']}  wall={result['wall_time_s']:.1f}s")

    output_data = {
        "game": args.config.stem,
        "seed": args.seed,
        "config": config,
        "result": result,
        "hardware": hardware_info(),
        "rom": rom_checksum(),
        "timestamp": dt.datetime.now().isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"Saved → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
