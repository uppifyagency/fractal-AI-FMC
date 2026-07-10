"""W4B — FMC applied to logic-synthesis operator sequencing (phase-ordering).

Second applicative spike of Fractal Monte Carlo (FMC): given an And-Inverter
Graph (AIG), choose the *sequence* of technology-independent optimization
operators (rewrite / resub / refactor / balance / cleanup) that minimizes the
final node (AND-gate) count. This is the classic ABC phase-ordering problem.
Strong baselines: resyn2 (ABC's canonical fixed script) and a size-greedy loop.

WHY FMC MIGHT FIT (the a-priori case)
-------------------------------------
Phase-ordering is NP-hard, actions are discrete (which operator to apply next),
the reward is a scalar (node reduction), the simulator is reversible (an AIG
snapshot = clone == get_state/set_state), and planning is per-instance.

KNOWN RISK (to be *measured*, not argued away): reward plateau. If node-count is
flat over the horizon M, the reward channel degenerates -> relativize maps a
near-constant vector to ~ones -> the cloning argmax carries no information ->
FMC ~= random. The E2 gate (disp_ratio + reward_cv_M) measures this BEFORE the
head-to-head, exactly as designed in wave3.

ENGINE = fmc-core (installed). We reuse fmc.core.plan and the wave3 E2 gate; we
do NOT reimplement FMC.

TOOLING = aigverse 0.1.1 (mockturtle bindings). Operators return a *new* Aig
(inplace=False), so state handling is clean. equivalence_checking (SAT) verifies
every method's final output preserves the Boolean function.

Run:  python3 w4b_logic_synthesis.py            # full run
      python3 w4b_logic_synthesis.py --pilot    # quick smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# --- fmc-core (installed) + the wave3 E2 gate ------------------------------- #
_WAVE3 = Path(__file__).resolve().parent.parent / "wave3_validation"
sys.path.insert(0, str(_WAVE3))
from fmc.core import plan                                    # noqa: E402
from w34_e2_smoke import e2_divergence                       # noqa: E402

# --- aigverse -------------------------------------------------------------- #
from aigverse.networks import Aig, DepthAig                  # noqa: E402
from aigverse import generators as G                         # noqa: E402
from aigverse import algorithms as A                         # noqa: E402

from scipy.stats import wilcoxon                             # noqa: E402


# =========================================================================== #
# Operators (the task's action set + zero-gain variants)                       #
# =========================================================================== #
# Full discrete action set exposed to FMC / greedy / E2. All six are
# equivalence-preserving on the structured arithmetic circuits used here
# (verified per-op; the final SAT check is the backstop).
OPS = ["b", "rw", "rwz", "rf", "rs", "cl"]
OP_LONG = {
    "b": "balancing",
    "rw": "cut_rewriting",
    "rwz": "cut_rewriting(zero-gain)",
    "rf": "sop_refactoring",
    "rs": "resubstitution",
    "cl": "cleanup_dangling",
}


def apply_op(name: str, aig: Aig) -> Aig:
    """Apply one operator, returning a NEW Aig. Never mutates the input."""
    x = aig.clone()
    if name == "b":
        return A.balancing(x)
    if name == "rw":
        return A.aig_cut_rewriting(x, allow_zero_gain=False)
    if name == "rwz":
        return A.aig_cut_rewriting(x, allow_zero_gain=True)
    if name == "rf":
        r = A.sop_refactoring(x, allow_zero_gain=False)
        return r if r is not None else x
    if name == "rfz":  # used only inside resyn2, not an FMC action
        r = A.sop_refactoring(x, allow_zero_gain=True)
        return r if r is not None else x
    if name == "rs":
        r = A.aig_resubstitution(x)
        return r if r is not None else x
    if name == "cl":
        return A.cleanup_dangling(x)
    raise ValueError(name)


# =========================================================================== #
# Synthesis environment (fmc-core Environment protocol)                        #
# =========================================================================== #

class SynthesisEnv:
    """State = an AIG. Immutable/atomic via Aig.clone (get_state/set_state).

    reward(s) = g0 - num_gates(s)  (accumulated node reduction from the start;
    higher = better). observe(s) = [gates, levels, size] structural projection
    used for the FMC diversity (distance) term and for E2 dispersion.
    """

    def __init__(self, base_aig: Aig):
        self.base = base_aig.clone()
        self.g0 = base_aig.num_gates
        self.n_steps = 0        # operator-application counter (cost accounting)

    def actions(self):
        return tuple(range(len(OPS)))

    def reset(self):
        return self.base.clone()

    def clone_state(self, s):
        return s.clone()

    def step(self, s, a):
        self.n_steps += 1
        return apply_op(OPS[a], s)

    def observe(self, s):
        return np.array([s.num_gates, DepthAig(s).num_levels, s.size],
                        dtype=np.float64)

    def reward(self, s):
        return float(self.g0 - s.num_gates)

    def sample_action(self, s, rng):
        return int(rng.integers(0, len(OPS)))


def levels(aig: Aig) -> int:
    return DepthAig(aig).num_levels


# =========================================================================== #
# Baselines                                                                    #
# =========================================================================== #

def run_resyn2(env: SynthesisEnv):
    """ABC's canonical resyn2 script, mapped onto aigverse operators:
        b; rw; rf; b; rw; rwz; b; rfz; rwz; b
    Returns (final_gates, best_gates_along_traj, best_state, traj)."""
    seq = ["b", "rw", "rf", "b", "rw", "rwz", "b", "rfz", "rwz", "b"]
    x = env.reset()
    traj = [x.num_gates]
    best_g, best_s = x.num_gates, x.clone()
    for s in seq:
        x = apply_op(s, x)
        traj.append(x.num_gates)
        if x.num_gates < best_g:
            best_g, best_s = x.num_gates, x.clone()
    return x.num_gates, best_g, best_s, traj


def run_greedy(env: SynthesisEnv, max_steps: int = 20):
    """Size-greedy: each step apply the operator with the best immediate node
    reduction; stop when no operator reduces (a strict local optimum).
    Returns (best_gates, best_state, n_op_evals, traj)."""
    x = env.reset()
    g = x.num_gates
    traj = [g]
    evals = 0
    for _ in range(max_steps):
        best_g, best_x = g, None
        for a in env.actions():
            r = apply_op(OPS[a], x)
            evals += 1
            if r.num_gates < best_g:
                best_g, best_x = r.num_gates, r
        if best_x is None:      # plateau: no immediate improvement
            break
        x, g = best_x, best_g
        traj.append(g)
    return g, x, evals, traj


def run_fmc(env: SynthesisEnv, H: int, N: int, M: int, seed: int):
    """Closed-loop FMC-base: at each of H decisions, fmc.core.plan runs an
    N-walker x M-horizon swarm and returns the chosen operator; we apply it and
    record the running-best node count along the executed trajectory.
    Returns (best_gates, best_state, n_op_evals, traj)."""
    env.n_steps = 0
    x = env.reset()
    best_g, best_s = x.num_gates, x.clone()
    traj = [x.num_gates]
    for k in range(H):
        a = plan(env, x, N=N, M=M, alpha=1.0, beta=1.0, seed=seed * 10_000 + k)
        x = env.step(x, a)
        traj.append(x.num_gates)
        if x.num_gates < best_g:
            best_g, best_s = x.num_gates, x.clone()
    return best_g, best_s, env.n_steps, traj


# =========================================================================== #
# Benchmark suite (structured arithmetic — all six operators equiv-safe)       #
# random_aig is EXCLUDED: aigverse `balancing` returns non-equivalent circuits #
# on some random AIGs (verified: seeds 1 and 3 at pis=8,gates=100). On the      #
# structured suite balancing is sound.                                         #
# =========================================================================== #

def build_suite():
    return [
        ("mult_3",  G.ripple_carry_multiplier(3)),
        ("mult_4",  G.ripple_carry_multiplier(4)),
        ("mult_5",  G.ripple_carry_multiplier(5)),
        ("rca_8",   G.ripple_carry_adder(8)),
        ("rca_16",  G.ripple_carry_adder(16)),
        ("cla_8",   G.carry_lookahead_adder(8)),
        ("cla_16",  G.carry_lookahead_adder(16)),
        ("cla_32",  G.carry_lookahead_adder(32)),
        ("mux_8",   G.multiplexer(8)),
        ("dec_5",   G.binary_decoder(5)),
    ]


# =========================================================================== #
# Drivers                                                                      #
# =========================================================================== #

def run_e2(suite, N=32, M=8, seeds=(0, 1, 2, 3)):
    print(f"\n=== E2 GATE (free-swarm divergence)  N={N} M={M} seeds={len(seeds)} ===")
    print(f"{'circuit':10s} {'g0':>5s} {'lev':>4s} {'disp_ratio':>11s} "
          f"{'reward_cv_M':>12s} {'ess_ratio':>10s}  verdict")
    print("-" * 78)
    rows = []
    for name, aig in suite:
        env = SynthesisEnv(aig)
        m = e2_divergence(env, env.reset(), N=N, M=M, seeds=seeds)
        rows.append((name, aig.num_gates, levels(aig), m))
        print(f"{name:10s} {aig.num_gates:>5d} {levels(aig):>4d} "
              f"{m['disp_ratio']:>11.3f} {m['reward_cv_M']:>12.4f} "
              f"{m['ess_ratio']:>10.3f}  {m['verdict']}")
    return rows


def run_headtohead(suite, H, N, M, seeds):
    print(f"\n=== FMC vs resyn2 vs greedy   FMC(H={H},N={N},M={M},seeds={len(seeds)}) ===")
    print(f"{'circuit':9s} {'g0':>4s} | {'resyn2_f':>8s} {'resyn2_b':>8s} | "
          f"{'greedy':>6s} {'g_ev':>5s} | {'FMC_med':>7s} {'FMC_best':>8s} "
          f"{'f_ev':>6s} | {'FMCvsGrd%':>9s} {'eq':>3s} {'t_s':>5s}")
    print("-" * 110)
    results = []
    for name, aig in suite:
        env = SynthesisEnv(aig)
        g0 = aig.num_gates

        r_final, r_best, r_state, _ = run_resyn2(env)
        gr_best, gr_state, gr_ev, _ = run_greedy(env)

        t0 = time.time()
        fmc_bests, fmc_states, fmc_evs = [], [], []
        for sd in seeds:
            b, st, ev, _ = run_fmc(env, H=H, N=N, M=M, seed=sd)
            fmc_bests.append(b); fmc_states.append(st); fmc_evs.append(ev)
        dt = time.time() - t0

        fmc_med = float(np.median(fmc_bests))
        fmc_min = int(np.min(fmc_bests))
        fmc_min_state = fmc_states[int(np.argmin(fmc_bests))]

        # correctness: SAT-verify each method's best output vs the original
        eq_r = A.equivalence_checking(aig, r_state)
        eq_g = A.equivalence_checking(aig, gr_state)
        eq_f = A.equivalence_checking(aig, fmc_min_state)
        eq_all = bool(eq_r) and bool(eq_g) and bool(eq_f)

        # FMC (median over seeds) vs greedy, in % of g0
        fmc_vs_greedy = 100.0 * (gr_best - fmc_med) / g0

        results.append(dict(
            name=name, g0=g0,
            resyn2_final=r_final, resyn2_best=r_best,
            greedy=gr_best, greedy_evals=gr_ev,
            fmc_median=fmc_med, fmc_min=fmc_min,
            fmc_evals=float(np.mean(fmc_evs)),
            fmc_vs_greedy_pct=fmc_vs_greedy,
            eq_r=bool(eq_r), eq_g=bool(eq_g), eq_f=bool(eq_f),
            time_s=dt,
        ))
        print(f"{name:9s} {g0:>4d} | {r_final:>8d} {r_best:>8d} | "
              f"{gr_best:>6d} {gr_ev:>5d} | {fmc_med:>7.1f} {fmc_min:>8d} "
              f"{np.mean(fmc_evs):>6.0f} | {fmc_vs_greedy:>+8.2f}% "
              f"{'Y' if eq_all else 'N':>3s} {dt:>5.1f}")
    return results


def summarize(results):
    print("\n=== PAIRED STATISTICS (across circuits) ===")
    g0 = np.array([r["g0"] for r in results], dtype=float)
    greedy = np.array([r["greedy"] for r in results], dtype=float)
    fmc = np.array([r["fmc_median"] for r in results], dtype=float)
    resyn2f = np.array([r["resyn2_final"] for r in results], dtype=float)
    resyn2b = np.array([r["resyn2_best"] for r in results], dtype=float)

    def red(x):  # mean % reduction vs g0
        return float(np.mean(100.0 * (g0 - x) / g0))

    print(f"mean node reduction vs g0:  resyn2_final={red(resyn2f):+.2f}%  "
          f"resyn2_best={red(resyn2b):+.2f}%  greedy={red(greedy):+.2f}%  "
          f"FMC={red(fmc):+.2f}%")

    # Paired Wilcoxon FMC vs greedy (final gate counts); needs nonzero diffs.
    diff = fmc - greedy
    nz = diff[diff != 0]
    print(f"\nFMC(median) - greedy  per circuit: {diff.astype(int).tolist()}")
    if len(nz) >= 1:
        try:
            w, p = wilcoxon(fmc, greedy)
            print(f"Wilcoxon FMC vs greedy: W={w:.3f} p={p:.4f} "
                  f"(n_nonzero={len(nz)}/{len(diff)})")
        except Exception as e:
            print(f"Wilcoxon FMC vs greedy: undefined ({e})")
    else:
        print("Wilcoxon FMC vs greedy: all pairs identical -> no difference "
              "to test (FMC ties greedy on every circuit).")

    # Bootstrap 95% CI on mean(FMC - greedy) in raw gates.
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(diff, size=len(diff), replace=True))
             for _ in range(10_000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"Bootstrap 95% CI of mean(FMC - greedy) gates: "
          f"[{lo:.2f}, {hi:.2f}]  (0 => tie; <0 => FMC better)")

    # cost ratio
    tot_g = sum(r["greedy_evals"] for r in results)
    tot_f = sum(r["fmc_evals"] for r in results)
    print(f"\nOperator-eval cost: greedy total={tot_g:.0f}, "
          f"FMC total (per seed avg)={tot_f:.0f}  "
          f"-> FMC/greedy = {tot_f / max(tot_g,1):.0f}x")

    n_eq = sum(1 for r in results if r["eq_r"] and r["eq_g"] and r["eq_f"])
    print(f"correctness (SAT equiv, all 3 methods): {n_eq}/{len(results)} circuits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    suite = build_suite()
    if args.pilot:
        suite = suite[:4]
        H, N, M, seeds = 6, 12, 3, (0, 1)
        e2_seeds = (0, 1)
        e2_N, e2_M = 16, 6
    else:
        H, N, M, seeds = 12, 16, 4, (0, 1, 2, 3)
        e2_seeds = (0, 1, 2, 3)
        e2_N, e2_M = 32, 8

    print("W4B — FMC logic-synthesis operator sequencing")
    print(f"operators: {[OP_LONG[o] for o in OPS]}")
    print(f"suite: {[n for n, _ in suite]}")

    t0 = time.time()
    e2_rows = run_e2(suite, N=e2_N, M=e2_M, seeds=e2_seeds)
    results = run_headtohead(suite, H=H, N=N, M=M, seeds=seeds)
    summarize(results)
    print(f"\ntotal wall time: {time.time() - t0:.1f}s")
    return e2_rows, results


if __name__ == "__main__":
    main()
