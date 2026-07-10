"""E2 divergence smoke test — a reusable "fit gate" for FMC candidate domains.

WHY THIS EXISTS
---------------
FMC (Fractal Monte Carlo) only beats random search when the *free* swarm —
N walkers started from the same state s0, each following an independent random
rollout — DIVERGES within the planning horizon M. The cloning kernel selects
among divergent trajectories; if there is nothing to select among, FMC ~= random.

This is the exact cause of the plasma failure (M18, 2026-05-05): the linear TCV
shape-control sim is quasi-deterministic, so all walkers collapse onto the same
gradient trajectory regardless of the action. Rewards become near-identical,
`relativize` maps a near-constant vector to ~all-ones (its std->0 branch), the
virtual reward is ~uniform, and the cloning argmax carries no information.

E2 measures that collapse BEFORE we invest in an adapter. It reuses the fmc-core
math layer (relativize / virtual_reward / effective_sample_size /
effective_branching_factor) — it does not reimplement FMC.

METRICS (all measured at horizon M, averaged over seeds)
--------------------------------------------------------
Free-swarm channel (raw dynamical divergence, no cloning):
  * disp_ratio   = meanPairwiseDist(obs @ M) / meanPairwiseDist(obs @ 1)
                   > 1 : the cloud grows (chaotic/expanding dynamics)
                  <= 1 : the cloud contracts (linear-stable dynamics)  <-- plasma
  * reward_cv_M  = std / (|mean|+eps) of the raw reward vector @ M
                   this is EXACTLY the input whose CV controls whether
                   relativize(reward) retains signal.

Relativize / VR channel (what FMC actually sees):
  * ess_ratio    = ESS(VR @ M) / N.  ~1.0 => VR uniform => cloning null.
  * b_eff        = effective branching factor of surviving labels (Def 6),
                   reported for context (not a primary gate; per MATH_CANON it
                   "looks at the wrong space" — monotone-contractive under cloning).

VERDICT
-------
"DIVERGE (FMC-fit)"  vs  "COLLAPSE (no-fit, FMC~random)".
See e2_verdict() for the rule and the empirically calibrated threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from fmc.core import (
    relativize,
    virtual_reward,
    effective_sample_size,
    effective_branching_factor,
    clone_step,
)

EPS = 1e-12

# --------------------------------------------------------------------------- #
# Empirically calibrated threshold (see W34_e2_smoke_test.md).                #
# Calibrated on 6 envs (4 diverge, 2 collapse) + a contraction-strength sweep. #
#                                                                             #
# PRIMARY GATE = disp_ratio only. Real separation is wide and clean:           #
#   collapse: {1.94, 2.39}       diverge: {4.66, 5.52, 8.84, 24.9}             #
#   -> gate 3.0 sits in the empty gap [2.39, 4.66] with margin on both sides.  #
# The contraction sweep confirms disp_ratio rises monotonically with the       #
# spectral radius A and crosses the gate exactly at the stability boundary     #
# (A ~ 0.93), i.e. it tracks the true dynamical property, not an arbitrary cut. #
#                                                                             #
# ess_ratio and reward_cv are REPORTED but NOT gated:                          #
#   * ess_ratio false-negatives CartPole (0.649 despite disp_ratio 5.5):       #
#     survival-reward + mass death inflate VR uniformity even when trajectories #
#     scatter widely. Thin/unreliable margin.                                  #
#   * reward_cv fails to separate at all (Rocket 0.36 < LinContractive 0.86):  #
#     dying walkers (reward=0) compress a divergent env's reward CV.           #
#   A near-zero reward_cv WOULD flag the distinct 'reward-degenerate' failure   #
#   mode (dispersion fine, but reward can't discriminate) -> soft warning only. #
# --------------------------------------------------------------------------- #
DISP_RATIO_GATE = 3.0        # free-swarm dispersion (in one-step-authority units)
REWARD_CV_WARN = 0.02        # below this: reward channel degenerate (soft warn)


# --------------------------------------------------------------------------- #
# Dispersion helpers                                                           #
# --------------------------------------------------------------------------- #

def _mean_pairwise_dist(obs: np.ndarray) -> float:
    """Mean off-diagonal pairwise Euclidean distance of an (N, F) cloud."""
    n = len(obs)
    if n < 2:
        return 0.0
    diff = obs[:, None, :] - obs[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    return float(d.sum() / (n * (n - 1)))


def _obs_matrix(env, states) -> np.ndarray:
    return np.stack(
        [np.asarray(env.observe(s), dtype=np.float64).ravel() for s in states]
    )


# --------------------------------------------------------------------------- #
# Free swarm (no cloning) — measures intrinsic dynamical divergence            #
# --------------------------------------------------------------------------- #

def _free_swarm(env, s0, N, M, rng):
    """N independent random rollouts from s0. Returns per-step obs/reward stats."""
    states = [env.clone_state(s0) for _ in range(N)]
    disp_curve = []
    rew_final = None
    obs_final = None
    for t in range(M):
        for i in range(N):
            a = env.sample_action(states[i], rng)
            states[i] = env.step(states[i], a)
        obs = _obs_matrix(env, states)
        disp_curve.append(_mean_pairwise_dist(obs))
        if t == M - 1:
            rew_final = np.array([env.reward(s) for s in states], dtype=np.float64)
            obs_final = obs
    return np.array(disp_curve), obs_final, rew_final


# --------------------------------------------------------------------------- #
# Instrumented FMC rollout — reuses core primitives, returns labels + ESS.     #
# (core.plan() returns only the action; we need the surviving-label dist and   #
#  the horizon ESS to score the gate, so we run the same loop and capture them)#
# --------------------------------------------------------------------------- #

def _fmc_rollout(env, s0, N, M, alpha, beta, rng):
    actions = list(env.actions())
    states = [env.clone_state(s0) for _ in range(N)]
    labels = np.array(
        [actions[rng.integers(0, len(actions))] for _ in range(N)], dtype=object
    )
    ess_final = float(N)
    for t in range(M):
        for i in range(N):
            a = labels[i] if t == 0 else env.sample_action(states[i], rng)
            states[i] = env.step(states[i], a)
        rewards = np.array([env.reward(s) for s in states], dtype=np.float64)
        obs = _obs_matrix(env, states)
        partners = rng.permutation(N)
        for i in range(N):
            if partners[i] == i:
                partners[i] = (i + 1) % N
        vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
        if t == M - 1:
            ess_final = effective_sample_size(vr)
        clone_idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in clone_idx]
        labels = labels[clone_idx]
    return labels, ess_final


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def e2_divergence(env, s0, N=64, M=30, alpha=1.0, beta=1.0, seeds=(0, 1, 2, 3, 4)):
    """Measure the E2 divergence signature of an environment at horizon M.

    Parameters
    ----------
    env   : object implementing the fmc.envs.base.Environment protocol.
    s0    : initial state (as returned by env.reset()).
    N, M  : swarm size and planning horizon.
    alpha, beta : virtual-reward exponents (FMC defaults 1.0, 1.0).
    seeds : iterable of RNG seeds; metrics are averaged across them.

    Returns
    -------
    dict with the E2 metrics (means over seeds) plus a 'verdict' string.
    """
    n_actions = len(list(env.actions()))
    disp_ratios, reward_cvs, ess_ratios, b_effs = [], [], [], []
    disp1s, dispMs = [], []

    for sd in seeds:
        rng = np.random.default_rng(sd)
        disp_curve, _obs_M, rew_M = _free_swarm(env, s0, N, M, rng)
        disp1, dispM = disp_curve[0], disp_curve[-1]
        disp1s.append(disp1)
        dispMs.append(dispM)
        disp_ratios.append(dispM / (disp1 + EPS))
        reward_cvs.append(rew_M.std() / (abs(rew_M.mean()) + EPS))

        rng2 = np.random.default_rng(1000 + sd)
        labels, ess = _fmc_rollout(env, s0, N, M, alpha, beta, rng2)
        ess_ratios.append(ess / N)
        b_effs.append(effective_branching_factor(labels))

    out = {
        "N": N, "M": M, "alpha": alpha, "beta": beta, "K": n_actions,
        "disp_1": float(np.mean(disp1s)),
        "disp_M": float(np.mean(dispMs)),
        "disp_ratio": float(np.mean(disp_ratios)),
        "reward_cv_M": float(np.mean(reward_cvs)),
        "ess_ratio": float(np.mean(ess_ratios)),
        "b_eff": float(np.mean(b_effs)),
        "b_eff_over_K": float(np.mean(b_effs)) / n_actions,
    }
    out["verdict"] = e2_verdict(out)
    return out


def e2_verdict(m: dict) -> str:
    """Gate rule. FMC-fit requires the free swarm to DIVERGE by horizon M:
    terminal dispersion must be >= DISP_RATIO_GATE x the one-step dispersion.
    If the swarm collapses onto a common trajectory, relativize zeroes the
    signal and FMC degenerates to random search (the M18 plasma failure).

    A separate soft warning fires when the reward channel is degenerate even
    though trajectories diverge (dispersion OK but reward can't discriminate)."""
    if m["disp_ratio"] < DISP_RATIO_GATE:
        return "COLLAPSE (no-fit)"
    if m["reward_cv_M"] < REWARD_CV_WARN:
        return "DIVERGE but reward-degenerate (weak-fit)"
    return "DIVERGE (FMC-fit)"


# =========================================================================== #
# Toy LINEAR / CONVEX environments — the FMC no-fit regime (plasma mimic).     #
# Implement the Environment protocol inline; they are validation fixtures,     #
# not shipped adapters.                                                        #
# =========================================================================== #

@dataclass
class _LinState:
    x: np.ndarray
    alive: bool = True


class LinearContractive:
    """2D linear-stable integrator with weak actuation and concave reward.

    Mimics the M18 plasma regime: x_{t+1} = A * x_t + B * nudge(a), with a
    strongly contracting A (eigenvalue 0.85) that pulls every walker toward the
    origin regardless of action, and a tiny actuation gain B. Reward is the
    concave quadratic r = -||x||^2 (peak at the origin). All walkers collapse
    onto the same attractor trajectory -> reward CV -> 0 -> relativize dies.
    """

    # 9 nudges = {-1,0,+1}^2 (same K=9 arity as rocket/nav/pendulum).
    _NUDGES = tuple(
        (dx, dy) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)
    )

    def __init__(self, A: float = 0.85, B: float = 0.01):
        self.A = A    # spectral radius: <1 stable/contracting, >1 expanding
        self.B = B    # actuation gain (how much actions matter)

    def reset(self):
        return _LinState(x=np.array([1.0, 1.0], dtype=np.float64), alive=True)

    def actions(self):
        return tuple(range(9))

    def clone_state(self, s):
        return _LinState(x=s.x.copy(), alive=s.alive)

    def step(self, s, a):
        dx, dy = self._NUDGES[a]
        nx = self.A * s.x + self.B * np.array([dx, dy], dtype=np.float64)
        return _LinState(x=nx, alive=True)

    def reward(self, s):
        return float(-(s.x @ s.x))

    def observe(self, s):
        return s.x

    def sample_action(self, s, rng):
        return int(rng.integers(0, 9))


class LinearIntegrator1D:
    """1D marginally-stable integrator, convex bowl reward. Second collapse case.

    x_{t+1} = 0.9 * x_t + 0.02 * a_dir. Reward r = -(x-target)^2. Actions nudge
    left/none/right (K=3). Still linear-stable: walkers pulled to a common band.
    """

    A = 0.9
    B = 0.02
    TARGET = 0.0
    _DIRS = (-1.0, 0.0, 1.0)

    def reset(self):
        return _LinState(x=np.array([1.0], dtype=np.float64), alive=True)

    def actions(self):
        return (0, 1, 2)

    def clone_state(self, s):
        return _LinState(x=s.x.copy(), alive=s.alive)

    def step(self, s, a):
        nx = self.A * s.x + self.B * np.array([self._DIRS[a]], dtype=np.float64)
        return _LinState(x=nx, alive=True)

    def reward(self, s):
        return float(-((s.x[0] - self.TARGET) ** 2))

    def observe(self, s):
        return s.x

    def sample_action(self, s, rng):
        return int(rng.integers(0, 3))


# =========================================================================== #
# Validation driver                                                            #
# =========================================================================== #

def fmc_vs_random(env, reset_fn, ep_len=30, N=48, M=15, seeds=(0, 1, 2)):
    """Closed-loop confirmation that the E2 verdict PREDICTS FMC's advantage.

    Runs an episode under (a) the FMC controller (core.plan each step) and
    (b) a uniform-random controller, from the same start, and returns mean
    episode return for each. Expectation: on DIVERGE envs FMC >> random; on
    COLLAPSE envs FMC ~= random (nothing to select among)."""
    from fmc.core import plan

    fmc_returns, rnd_returns = [], []
    for sd in seeds:
        # FMC controller
        s = reset_fn()
        tot = 0.0
        for k in range(ep_len):
            a = plan(env, s, N=N, M=M, alpha=1.0, beta=1.0, seed=sd * 100 + k)
            s = env.step(s, a)
            tot += env.reward(s)
        fmc_returns.append(tot)
        # Random controller
        rng = np.random.default_rng(sd)
        s = reset_fn()
        tot = 0.0
        acts = list(env.actions())
        for k in range(ep_len):
            a = acts[rng.integers(0, len(acts))]
            s = env.step(s, a)
            tot += env.reward(s)
        rnd_returns.append(tot)
    return float(np.mean(fmc_returns)), float(np.mean(rnd_returns))


def _fmt(m: dict) -> str:
    return (
        f"disp_ratio={m['disp_ratio']:8.3f}  "
        f"reward_cv_M={m['reward_cv_M']:8.4f}  "
        f"ess_ratio={m['ess_ratio']:6.3f}  "
        f"b_eff={m['b_eff']:5.2f}/{m['K']:<2d}  "
        f"|  disp_1={m['disp_1']:.4f} disp_M={m['disp_M']:.4f}"
    )


def main():
    from fmc.envs.rocket import Rocket
    from fmc.envs.navigation2d import Navigation2D
    from fmc.envs.pendulum import Pendulum
    from fmc.envs.cartpole import CartPole

    N, M = 64, 30
    SEEDS = tuple(range(8))
    cases = []

    # --- Envs that MUST diverge (FMC works here) ---
    r = Rocket();          cases.append(("Rocket (nonlinear)",       r, r.reset(), "diverge"))
    nav = Navigation2D();  cases.append(("Navigation2D (nonlinear)", nav, nav.reset(), "diverge"))
    pen = Pendulum();      cases.append(("Pendulum (nonlinear)",     pen, pen.reset(seed=0), "diverge"))
    cp = CartPole();       cases.append(("CartPole (nonlinear)",     cp, cp.reset(seed=0), "diverge"))

    # --- Envs that MUST collapse (FMC ~= random) ---
    lc = LinearContractive();  cases.append(("LinearContractive 2D (plasma mimic)", lc, lc.reset(), "collapse"))
    li = LinearIntegrator1D(); cases.append(("LinearIntegrator1D",                  li, li.reset(), "collapse"))

    print(f"E2 divergence smoke test  |  N={N} M={M} alpha=1.0 beta=1.0  "
          f"seeds={len(SEEDS)}  |  gate: disp_ratio >= {DISP_RATIO_GATE}\n")
    print(f"{'env':<38s} {'expected':<9s} {'verdict':<20s} metrics")
    print("-" * 140)

    n_correct = 0
    results = []
    for name, env, s0, expected in cases:
        m = e2_divergence(env, s0, N=N, M=M, alpha=1.0, beta=1.0, seeds=SEEDS)
        m["env"] = name
        m["expected"] = expected
        results.append(m)
        got = "diverge" if m["verdict"].startswith("DIVERGE") else "collapse"
        ok = "OK " if got == expected else "XX "
        n_correct += (got == expected)
        print(f"{name:<38s} {expected:<9s} {m['verdict']:<20s} {_fmt(m)}   [{ok}]")

    print("-" * 140)
    print(f"separation: {n_correct}/{len(cases)} correct")

    # --- Calibration sweep: dial the spectral radius A of the linear env from    ---
    # --- strongly contracting (collapse) through marginal to expanding (diverge).---
    # --- Proves disp_ratio tracks the underlying property, not an arbitrary cut. ---
    print("\ncontraction-strength calibration sweep (LinearContractive, B=0.01):")
    print(f"{'A (spectral radius)':<22s} {'disp_ratio':>12s} {'ess_ratio':>10s} {'verdict':>20s}")
    print("-" * 66)
    for A in (0.5, 0.7, 0.85, 0.95, 1.0, 1.02, 1.05):
        env_a = LinearContractive(A=A, B=0.01)
        m = e2_divergence(env_a, env_a.reset(), N=N, M=M, alpha=1.0, beta=1.0, seeds=SEEDS)
        print(f"{A:<22.2f} {m['disp_ratio']:>12.3f} {m['ess_ratio']:>10.3f} {m['verdict']:>20s}")

    # --- Predictive-validity check: does the E2 verdict predict FMC > random? ---
    print("\nFMC vs random closed-loop (ep_len=30, N=48, M=15, 3 seeds):")
    print(f"{'env':<38s} {'E2 verdict':<12s} {'FMC ret':>10s} {'rand ret':>10s} {'FMC-rand':>10s}")
    print("-" * 84)
    checks = [
        ("Rocket",             Rocket(),             lambda: Rocket().reset(),            "diverge"),
        ("Navigation2D",       Navigation2D(),       lambda: Navigation2D().reset(),      "diverge"),
        ("Pendulum",           Pendulum(),           lambda: Pendulum().reset(seed=0),    "diverge"),
        ("LinearContractive",  LinearContractive(),  lambda: LinearContractive().reset(), "collapse"),
    ]
    for name, env_c, reset_fn, grp in checks:
        fmc_r, rnd_r = fmc_vs_random(env_c, reset_fn)
        tag = "diverge" if grp == "diverge" else "collapse"
        print(f"{name:<38s} {tag:<12s} {fmc_r:>10.3f} {rnd_r:>10.3f} {fmc_r - rnd_r:>10.3f}")

    return results


if __name__ == "__main__":
    main()
