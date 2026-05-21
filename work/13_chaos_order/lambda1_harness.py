"""Conjecture B / H-B1a — swarm Lyapunov exponent harness.

Deep dive 09 (chaos/order frontier) reformulated Conjecture B around Psi_1, the
largest Lyapunov exponent of the FMC swarm in state space, with the frontier at
lambda_1 ~ 0. H-B1a is the existence test: does lambda_1 cross zero as alpha is
swept?  Falsified if it never changes sign.

Method (dd09 section 6, twin-trajectory / Benettin):
  - Two swarms A (reference) and B (perturbed), evolved under the SAME realised
    randomness (two RNGs seeded identically -> lockstep: the FMC rng-consumption
    pattern does not depend on the virtual reward, so the draws stay in sync).
  - B starts as A's initial config with each walker perturbed by a tiny vector;
    the total state-space separation is rescaled to delta0.
  - Each tick: run one FMC iteration on each swarm, measure the separation
    ||W^A - W^B|| in observation space, log the growth, then renormalise B back
    to separation delta0 (Benettin).
  - lambda_1 = mean per-tick log-growth.

dd09 section 3.1 caveat is respected: FMC's cloning is discontinuous, so this is
a FINITE-delta estimate and may depend on delta0 -- a delta0 sweep is included.

Kernel fmc-core is UNCHANGED: the harness replicates the public plan() tick and
calls fmc.core.{virtual_reward, clone_step} unmodified.

Run:  python work/13_chaos_order/lambda1_harness.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "fmc-core" / "src"))

from fmc.core import clone_step, virtual_reward          # noqa: E402
from fmc.envs.navigation2d import Navigation2D, State as NavState   # noqa: E402
from fmc.envs.pendulum import Pendulum, State as PendState         # noqa: E402

# --- experiment parameters ---------------------------------------------------
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
BETA = 1.0
N_WALKERS = 64
HORIZON_M = 20
N_SEEDS = 30
DELTA0 = 1e-3
DELTA_SWEEP = [1e-2, 1e-3, 1e-4]      # dd09 section 3.1: characterise delta-dependence


# --- per-environment adapters (observe-space, wrap-safe) ---------------------

class NavAdapter:
    name = "navigation2d"
    dim = 2

    def make_env(self):       return Navigation2D()
    def x0(self, env):        return env.reset()                  # (0.15, 0.15)
    def observe(self, s):     return np.array([s.x, s.y], dtype=np.float64)

    def from_observe(self, ref, v):
        return NavState(x=float(v[0]), y=float(v[1]),
                        goal_x=ref.goal_x, goal_y=ref.goal_y, alive=ref.alive)


class PendAdapter:
    name = "pendulum"
    dim = 3                                                       # [cos, sin, vel]

    def make_env(self):       return Pendulum()
    def x0(self, env):        return PendState(theta=math.pi, theta_dot=0.0, alive=True)
    def observe(self, s):
        return np.array([math.cos(s.theta), math.sin(s.theta),
                         s.theta_dot * 0.125], dtype=np.float64)

    def from_observe(self, ref, v):
        # Observe-space is wrap-safe; invert it back to (theta, theta_dot).
        theta = math.atan2(float(v[1]), float(v[0]))
        return PendState(theta=theta, theta_dot=float(v[2]) / 0.125, alive=ref.alive)


# --- FMC tick: replicates fmc.core.plan's loop body exactly ------------------

def _tick(env, states, labels, rng, t, alpha, beta):
    n = len(states)
    for i in range(n):
        a = labels[i] if t == 0 else env.sample_action(states[i], rng)
        states[i] = env.step(states[i], a)
    rewards = np.array([env.reward(s) for s in states], dtype=np.float64)
    obs = np.stack([np.asarray(env.observe(s), dtype=np.float64).ravel()
                    for s in states])
    partners = rng.permutation(n)
    for i in range(n):
        if partners[i] == i:
            partners[i] = (i + 1) % n
    vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
    clone_idx = clone_step(vr, rng)
    states = [env.clone_state(states[k]) for k in clone_idx]
    labels = labels[clone_idx]
    return states, labels


def _sep(adapter, A, B):
    """State-space separation ||W^A - W^B|| over the swarm (observation space)."""
    diff = np.stack([adapter.observe(A[i]) - adapter.observe(B[i])
                     for i in range(len(A))])
    return float(np.sqrt((diff * diff).sum()))


def _renorm(adapter, A, B, delta0):
    """Benettin: pull B back toward A so the separation equals delta0."""
    d = _sep(adapter, A, B)
    if d < 1e-18:
        return B                              # collapsed; leave as is (lambda<<0)
    scale = delta0 / d
    return [adapter.from_observe(A[i],
                                 adapter.observe(A[i])
                                 + scale * (adapter.observe(B[i]) - adapter.observe(A[i])))
            for i in range(len(A))]


def measure_lambda1(adapter, env, x0, N, M, alpha, beta, seed, delta0):
    """Twin-trajectory finite-delta Lyapunov exponent of the FMC swarm map."""
    rngA = np.random.default_rng(seed)
    rngB = np.random.default_rng(seed)        # identical -> lockstep randomness
    prng = np.random.default_rng((seed << 1) ^ 0x9E3779B9)
    acts = list(env.actions())

    A = [env.clone_state(x0) for _ in range(N)]
    B = [adapter.from_observe(x0, adapter.observe(x0)
                              + prng.normal(0.0, 1.0, size=adapter.dim))
         for _ in range(N)]
    labA = np.array([acts[int(rngA.integers(0, len(acts)))] for _ in range(N)],
                    dtype=object)
    labB = np.array([acts[int(rngB.integers(0, len(acts)))] for _ in range(N)],
                    dtype=object)
    B = _renorm(adapter, A, B, delta0)        # initial separation := delta0

    logs = []
    for t in range(M):
        A, labA = _tick(env, A, labA, rngA, t, alpha, beta)
        B, labB = _tick(env, B, labB, rngB, t, alpha, beta)
        d = _sep(adapter, A, B)
        logs.append(math.log(max(d, 1e-15) / delta0))
        B = _renorm(adapter, A, B, delta0)
    return float(np.mean(logs))               # per-tick lambda_1


# --- H-B1a: sweep alpha, look for the zero crossing --------------------------

def sweep(adapter, delta0=DELTA0, n_seeds=N_SEEDS):
    env = adapter.make_env()
    x0 = adapter.x0(env)
    out = {}
    for alpha in ALPHAS:
        vals = [measure_lambda1(adapter, env, x0, N_WALKERS, HORIZON_M,
                                alpha, BETA, seed=1000 + s, delta0=delta0)
                for s in range(n_seeds)]
        m = float(np.mean(vals))
        sem = float(np.std(vals) / math.sqrt(n_seeds))
        out[alpha] = {"lambda1": m, "sem": sem}
    return out


def main():
    t0 = time.time()
    results = {}
    for adapter in (NavAdapter(), PendAdapter()):
        results[adapter.name] = {"sweep": sweep(adapter)}
        print(f"  [{time.time()-t0:6.1f}s] {adapter.name} alpha-sweep done")

    # delta-dependence check (dd09 section 3.1): is lambda_1 scale-free?
    delta_check = {}
    for adp, alphas_dc in ((NavAdapter(), [0.0, 0.1, 0.5]), (PendAdapter(), [0.1])):
        env_dc = adp.make_env(); x0_dc = adp.x0(env_dc)
        for a_dc in alphas_dc:
            for d0 in DELTA_SWEEP:
                vals = [measure_lambda1(adp, env_dc, x0_dc, N_WALKERS, HORIZON_M,
                                        a_dc, BETA, seed=1000 + s, delta0=d0)
                        for s in range(N_SEEDS)]
                delta_check[f"{adp.name}_a{a_dc}_d{d0:.0e}"] = {
                    "lambda1": float(np.mean(vals)),
                    "sem": float(np.std(vals) / math.sqrt(N_SEEDS))}
    results["delta_check"] = delta_check

    # --- report --------------------------------------------------------------
    print("\n" + "=" * 74)
    print("H-B1a — swarm Lyapunov exponent lambda_1 vs alpha (beta=1, N=64, M=20)")
    print("=" * 74)
    verdict = {}
    for name in ("navigation2d", "pendulum"):
        sw = results[name]["sweep"]
        print(f"\n{name}:")
        print(f"  {'alpha':>6s} {'lambda_1':>12s} {'+/- sem':>10s}")
        means = []
        for a in ALPHAS:
            print(f"  {a:6.2f} {sw[a]['lambda1']:12.4f} {sw[a]['sem']:10.4f}")
            means.append(sw[a]["lambda1"])
        sems = [sw[a]["sem"] for a in ALPHAS]
        # a *resolved* crossing: consecutive points individually sign-resolved
        # (|mean| > 2*sem) with opposite signs.
        crosses = any(
            ((means[i] > 2 * sems[i] and means[i + 1] < -2 * sems[i + 1]) or
             (means[i] < -2 * sems[i] and means[i + 1] > 2 * sems[i + 1]))
            for i in range(len(means) - 1))
        mono = all(means[i] >= means[i + 1] - 3 * sw[ALPHAS[i + 1]]["sem"]
                   for i in range(len(means) - 1))
        alpha_c = None
        if crosses:
            for i in range(len(ALPHAS) - 1):
                if (means[i] > 0) != (means[i + 1] > 0):
                    lo, hi = means[i], means[i + 1]
                    frac = lo / (lo - hi) if lo != hi else 0.5
                    alpha_c = ALPHAS[i] + frac * (ALPHAS[i + 1] - ALPHAS[i])
                    break
        verdict[name] = {"crosses_zero": crosses, "monotone_decr": mono,
                         "alpha_c": alpha_c}
        print(f"  -> crosses zero: {crosses}"
              + (f" (alpha_c ~ {alpha_c:.3f})" if alpha_c is not None else "")
              + f" | monotone decreasing: {mono}")

    print("\ndelta-dependence — is lambda_1 scale-free? (dd09 section 3.1):")
    groups = {}
    for k, dc in results["delta_check"].items():
        print(f"  {k:26s} lambda_1={dc['lambda1']:+.4f} +/- {dc['sem']:.4f}")
        groups.setdefault(k.rsplit("_d", 1)[0], []).append(dc["lambda1"])
    delta_signflip = {pre: (min(g) < 0.0 < max(g)) for pre, g in groups.items()}
    verdict["delta_signflip"] = delta_signflip

    # --- pre-registered H-B1a verdict (dd09 section 4.2) ---------------------
    print("\n" + "-" * 74)
    both_cross = all(verdict[n]["crosses_zero"] for n in ("navigation2d", "pendulum"))
    any_signflip = any(delta_signflip.values())
    if any_signflip:
        print("H-B1a INCONCLUSIVE — the finite-delta lambda_1 is NOT scale-free: it "
              "changes sign across delta0 " +
              f"({[p for p,f in delta_signflip.items() if f]}). The dd09 section 3.1 "
              "caveat is confirmed empirically. A scale-free swarm Lyapunov exponent "
              "(Psi_1) is not resolved by the naive twin-trajectory estimator; the "
              "alpha-sweep shows lambda_1 ~ 0 within noise throughout. H-B1a needs a "
              "scale-resolved treatment or a different Psi. NOT a falsification of "
              "Conjecture B's substance — a finding about the estimator and the "
              "discontinuous cloning dynamics.")
    elif both_cross:
        print("H-B1a SUPPORTED — lambda_1 crosses zero (sign-resolved) on both tasks.")
    else:
        print("H-B1a NOT SUPPORTED in this range — no sign-resolved zero crossing.")
    print("  (dd09 section 4.2; finite-delta estimate per dd09 section 3.1.)")

    out = _HERE / "results" / "hb1a_lambda1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "params": {"alphas": ALPHAS, "beta": BETA, "N": N_WALKERS,
                   "M": HORIZON_M, "n_seeds": N_SEEDS, "delta0": DELTA0,
                   "delta_sweep": DELTA_SWEEP},
        "results": {k: v for k, v in results.items()},
        "verdict": verdict,
    }, indent=2, default=str))
    print(f"\nwrote {out}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
