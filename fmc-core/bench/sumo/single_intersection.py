"""Bet 1 — single-intersection SUMO benchmark, first-pass.

Three controllers compared on the same Poisson-arrivals scenario:

  1. actuated   — SUMO default (gap-based actuated traffic light, tuned).
  2. static     — fixed 30s green N/S then 30s green E/W (no adaptation).
  3. fmc        — FMC planner: at each decision step, simulate M futures using
                  a simple queue model (deterministic linear-saturated),
                  pick the action that maximizes throughput-derived reward.

The FMC planner does NOT use SUMO for forward simulation (would need 64*M
state clones — not feasible). It uses a Python-level queue dynamics model
for planning, and applies the action on SUMO via traci.

This is a *first pass*: validates the wiring + produces real numbers.
A production run would need (a) richer queue model, (b) GBR/MaxPressure
baseline, (c) RESCO Cologne/Hangzhou scenarios, (d) more seeds.

Honesty: the original Bet 1 spec called for 3-4 weeks of work. This pass
is ~2 hours. Output is a smoke test, not a publication-grade evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import traci
import sumolib

from fmc.core import virtual_reward, clone_step
from bench.runner import _hardware_info
import fmc as _fmc


HERE = Path(__file__).parent
SCENARIO_DIR = HERE / "scenarios" / "single_intersection"

# Two phases for the FMC controller.
# 0 = green N/S, red E/W
# 1 = green E/W, red N/S
PHASES = [
    "GGGggrrrrrGGGggrrrrr",  # NS green (greens for N->S, S->N + their right turns)
    "rrrrrGGGggrrrrrGGGgg",  # EW green
]

MIN_GREEN = 5      # seconds, prevent flicker
YELLOW = 3         # seconds yellow when switching


# ---------------------------------------------------------------------------
# SUMO helpers.
# ---------------------------------------------------------------------------

def _start_sumo(controller: str, seed: int, gui: bool = False) -> None:
    """Start SUMO via traci. controller selects how the traffic light is run."""
    binary = "sumo-gui" if gui else "sumo"
    cfg = str(SCENARIO_DIR / "sumo.sumocfg")
    cmd = [binary, "-c", cfg, "--seed", str(seed),
           "--no-step-log", "--no-warnings",
           "--tripinfo-output", str(SCENARIO_DIR / f"trips_{controller}_{seed}.xml")]
    traci.start(cmd)


def _close_sumo() -> None:
    try:
        traci.close()
    except Exception:
        pass


def _set_phase(tls_id: str, phase_idx: int) -> None:
    """Set the traffic light to a specific phase string."""
    traci.trafficlight.setRedYellowGreenState(tls_id, PHASES[phase_idx])


def _queue_lengths() -> np.ndarray:
    """Return current queue length (halted vehicles) per incoming lane."""
    incoming = ["n_to_c_0", "n_to_c_1", "s_to_c_0", "s_to_c_1",
                "e_to_c_0", "e_to_c_1", "w_to_c_0", "w_to_c_1"]
    return np.array([traci.lane.getLastStepHaltingNumber(l) for l in incoming], dtype=np.float64)


def _vehicles_in_network() -> int:
    return traci.simulation.getMinExpectedNumber()


def _arrived_count() -> int:
    return traci.simulation.getArrivedNumber()


# ---------------------------------------------------------------------------
# Controllers.
# ---------------------------------------------------------------------------

def run_actuated(seed: int) -> Dict[str, float]:
    """Run with SUMO's built-in actuated traffic light (already in the network)."""
    _start_sumo("actuated", seed)
    tls_id = traci.trafficlight.getIDList()[0]
    total_arrived = 0
    total_waiting_time = 0.0
    step = 0
    end_time = 600
    while traci.simulation.getTime() < end_time:
        traci.simulationStep()
        total_arrived += _arrived_count()
        # Sum of waiting time of all vehicles in network this step.
        for lane in ["n_to_c_0", "n_to_c_1", "s_to_c_0", "s_to_c_1",
                     "e_to_c_0", "e_to_c_1", "w_to_c_0", "w_to_c_1"]:
            total_waiting_time += traci.lane.getWaitingTime(lane)
        step += 1
    _close_sumo()
    return {
        "controller": "actuated",
        "seed": seed,
        "throughput": total_arrived,
        "total_waiting_time": total_waiting_time,
        "avg_waiting_per_step": total_waiting_time / max(1, step),
    }


def run_static(seed: int) -> Dict[str, float]:
    """Fixed 30s NS green then 30s EW green, with 3s yellow transitions."""
    _start_sumo("static", seed)
    tls_id = traci.trafficlight.getIDList()[0]
    total_arrived = 0
    total_waiting_time = 0.0
    step = 0
    end_time = 600
    cycle_len_ns = 30
    cycle_len_ew = 30
    yellow = YELLOW

    last_phase = 0
    _set_phase(tls_id, 0)
    phase_clock = 0

    while traci.simulation.getTime() < end_time:
        traci.simulationStep()
        step += 1
        phase_clock += 1
        total_arrived += _arrived_count()
        for lane in ["n_to_c_0", "n_to_c_1", "s_to_c_0", "s_to_c_1",
                     "e_to_c_0", "e_to_c_1", "w_to_c_0", "w_to_c_1"]:
            total_waiting_time += traci.lane.getWaitingTime(lane)
        # Phase transition (with yellow, but simplified: skip yellow).
        if last_phase == 0 and phase_clock >= cycle_len_ns:
            last_phase = 1
            _set_phase(tls_id, 1)
            phase_clock = 0
        elif last_phase == 1 and phase_clock >= cycle_len_ew:
            last_phase = 0
            _set_phase(tls_id, 0)
            phase_clock = 0
    _close_sumo()
    return {
        "controller": "static",
        "seed": seed,
        "throughput": total_arrived,
        "total_waiting_time": total_waiting_time,
        "avg_waiting_per_step": total_waiting_time / max(1, step),
    }


# ---------------------------------------------------------------------------
# FMC planner — simple queue model for forward simulation.
# ---------------------------------------------------------------------------

# Saturation flow per green-direction (vehicles / second) — calibrated
# rough estimate for 2-lane direction.
SAT_FLOW = 1.2  # per direction
ARRIVAL_RATE = 0.15  # per direction per second (matches gen_routes default)


def _queue_simulate(q0: np.ndarray, action: int, dt: float) -> np.ndarray:
    """Simple deterministic queue model for FMC forward simulation.

    State: q in R^4 = [N, S, E, W] queue lengths.
    Action: 0 = green NS, 1 = green EW.
    Dynamics over dt seconds:
      - all directions accumulate ARRIVAL_RATE * dt vehicles.
      - active direction discharges at SAT_FLOW * dt (capped by queue length).
    """
    q = q0.copy()
    arrivals = ARRIVAL_RATE * dt
    discharge = SAT_FLOW * dt
    q += arrivals  # all 4 directions get arrivals
    if action == 0:  # NS green
        q[0] = max(0.0, q[0] - discharge)
        q[1] = max(0.0, q[1] - discharge)
    else:  # EW green
        q[2] = max(0.0, q[2] - discharge)
        q[3] = max(0.0, q[3] - discharge)
    return q


def _plan_fmc(q_state: np.ndarray, N: int = 64, M: int = 6, alpha: float = 0.1, beta: float = 0.0,
              dt: float = 5.0, seed: int = 0) -> int:
    """FMC planner over the queue model. Returns action index (0 or 1)."""
    rng = np.random.default_rng(seed)
    actions = (0, 1)
    states = [q_state.copy() for _ in range(N)]
    labels = np.array(
        [actions[rng.integers(0, len(actions))] for _ in range(N)],
        dtype=np.int64,
    )
    for t in range(M):
        for i in range(N):
            a = labels[i] if t == 0 else int(rng.integers(0, 2))
            states[i] = _queue_simulate(states[i], a, dt)
        # Reward: -total_queue (lower queue = better).
        rewards = -np.array([s.sum() for s in states])
        obs = np.stack(states)
        partners = rng.permutation(N)
        for i in range(N):
            if partners[i] == i:
                partners[i] = (i + 1) % N
        vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
        idx = clone_step(vr, rng)
        states = [states[k].copy() for k in idx]
        labels = labels[idx]
    return int(Counter(labels.tolist()).most_common(1)[0][0])


def run_fmc(seed: int, N_walkers: int = 32, M_horizon: int = 6, dt_decision: float = 10.0) -> Dict[str, float]:
    """FMC controller: every dt_decision seconds, plan with FMC over the queue model."""
    _start_sumo("fmc", seed)
    tls_id = traci.trafficlight.getIDList()[0]

    total_arrived = 0
    total_waiting_time = 0.0
    step = 0
    end_time = 600
    last_phase = 0
    _set_phase(tls_id, last_phase)
    phase_clock = 0

    next_decision_at = MIN_GREEN  # first decision after MIN_GREEN seconds
    plans = 0

    while traci.simulation.getTime() < end_time:
        traci.simulationStep()
        step += 1
        phase_clock += 1
        total_arrived += _arrived_count()
        for lane in ["n_to_c_0", "n_to_c_1", "s_to_c_0", "s_to_c_1",
                     "e_to_c_0", "e_to_c_1", "w_to_c_0", "w_to_c_1"]:
            total_waiting_time += traci.lane.getWaitingTime(lane)

        if traci.simulation.getTime() >= next_decision_at and phase_clock >= MIN_GREEN:
            # Read queue state (4-dim: aggregate per direction).
            ql = _queue_lengths()
            q4 = np.array([ql[0]+ql[1], ql[2]+ql[3], ql[4]+ql[5], ql[6]+ql[7]])
            action = _plan_fmc(q4, N=N_walkers, M=M_horizon, dt=dt_decision,
                               seed=seed * 100000 + step)
            plans += 1
            if action != last_phase:
                last_phase = action
                _set_phase(tls_id, last_phase)
                phase_clock = 0
            next_decision_at = traci.simulation.getTime() + dt_decision

    _close_sumo()
    return {
        "controller": "fmc",
        "seed": seed,
        "throughput": total_arrived,
        "total_waiting_time": total_waiting_time,
        "avg_waiting_per_step": total_waiting_time / max(1, step),
        "fmc_plans": plans,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rate", type=float, default=0.15)
    parser.add_argument("--rate-heavy", type=float, default=None,
                        help="If set, use asymmetric scenario with this N/S rate")
    parser.add_argument("--rate-light", type=float, default=0.05,
                        help="E/W rate when --rate-heavy is set")
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    asymmetric = args.rate_heavy is not None
    scenario_tag = "asymmetric" if asymmetric else "symmetric"
    output = args.output or str(HERE.parent / "results" / f"sumo_single_intersection_{scenario_tag}.jsonl")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if asymmetric:
        print(f"Bet 1 — ASYMMETRIC traffic (N/S={args.rate_heavy}, E/W={args.rate_light}, T={args.duration}s)")
    else:
        print(f"Bet 1 — symmetric Poisson (rate={args.rate}/s/dir, T={args.duration}s)")
    print(f"Methods: actuated, static, fmc(N={args.N}, M={args.M}); seeds={args.seeds}\n")

    results = []
    for seed in range(args.seeds):
        # Regenerate routes deterministically per seed.
        if asymmetric:
            os.system(
                f"cd '{SCENARIO_DIR}' && python3 gen_routes_asymmetric.py "
                f"--rate-heavy {args.rate_heavy} --rate-light {args.rate_light} "
                f"--duration {args.duration} --seed {seed} "
                f"--output routes.rou.xml > /dev/null"
            )
        else:
            os.system(
                f"cd '{SCENARIO_DIR}' && python3 gen_routes.py "
                f"--rate {args.rate} --duration {args.duration} --seed {seed} "
                f"--output routes.rou.xml > /dev/null"
            )
        # Patch sumo.sumocfg end value.
        cfg_path = SCENARIO_DIR / "sumo.sumocfg"
        cfg = cfg_path.read_text()
        cfg = cfg.replace("<end value=\"600\"/>", f"<end value=\"{args.duration}\"/>")
        cfg_path.write_text(cfg)

        for runner_fn in [run_actuated, run_static, run_fmc]:
            t0 = time.time()
            if runner_fn is run_fmc:
                r = runner_fn(seed, N_walkers=args.N, M_horizon=args.M)
            else:
                r = runner_fn(seed)
            r["duration_s"] = time.time() - t0
            r["scenario"] = scenario_tag
            r["arrival_rate_per_s_per_dir"] = args.rate if not asymmetric else None
            r["arrival_rate_heavy"] = args.rate_heavy
            r["arrival_rate_light"] = args.rate_light if asymmetric else None
            r["sim_duration_s"] = args.duration
            print(f"  seed={seed} {r['controller']:<10} throughput={r['throughput']}  "
                  f"avg_wait={r['avg_waiting_per_step']:.1f}  {r['duration_s']:.1f}s")
            results.append(r)
        print()

    # Summary by controller.
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    by_ctrl: Dict[str, List[Dict]] = {}
    for r in results:
        by_ctrl.setdefault(r["controller"], []).append(r)
    for ctrl, rs in by_ctrl.items():
        thr = np.array([r["throughput"] for r in rs])
        wait = np.array([r["avg_waiting_per_step"] for r in rs])
        print(f"  {ctrl:<10} throughput = {thr.mean():.1f} ± {thr.std(ddof=1) if len(thr)>1 else 0:.1f}  "
              f"avg_wait = {wait.mean():.1f}")

    # Persist.
    with open(output, "w") as f:
        for r in results:
            r["hardware"] = _hardware_info()
            r["fmc_core_version"] = _fmc.__version__
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} records to {output}")


if __name__ == "__main__":
    main()
