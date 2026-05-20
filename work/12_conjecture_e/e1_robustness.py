"""Conjecture E — test E1-robustness (adversarial geometry).

Pre-registered design: E1_ROBUSTNESS_DESIGN.md  (read it first).

Closes the geometry caveat flagged in RESULT.md / E2_RESULT.md:

    E1-base used 3 layouts with lava in COMPACT regions. At alpha=0 the FMC
    virtual reward is relativize(distance)^beta -- it rewards spatial OUTLIERS.
    A walker that reaches ISOLATED, DISTANT lava is a max-diversity outlier ->
    high VR -> its frozen/dead state (and t=0 label) could propagate by cloning
    -> the swarm is pulled ONTO the lava. If so, E1's emergent self-preservation
    INVERTS on adversarial geometry. Empirical question -> test it.

Three adversarial layouts, all targeting that exact mechanism at different
scales: island (one 2x2 block), spur (far-corner cluster, off the start->goal
path), archipelago (several lone cells, each a strong outlier).

Kernel fmc-core unchanged. N=64, M=20, beta=1 (identical to E1-base).
n=60 episodes/cell. alpha in {0, 0.1, 1.0}. Machinery reused from e1_base.py.

Run:  python work/12_conjecture_e/e1_robustness.py
"""

from __future__ import annotations

import json
import time
from math import sqrt
from pathlib import Path

from e1_base import (  # noqa: E402  (e1_base sets up sys.path for fmc-core)
    BETA,
    HORIZON_M,
    MAX_STEPS,
    N_WALKERS,
    greedy_policy,
    make_fmc_policy,
    random_policy,
    run_episode,
    two_prop_z,
)
from gridworld_terminal import parse_layout  # noqa: E402

_HERE = Path(__file__).resolve().parent

# --- experiment parameters (alpha set + n differ from e1_base) ---------------
ALPHAS = [0.0, 0.1, 1.0]
N_EPISODES = 60  # E2's n; E1-base's n=20 too weak for a small nonzero rate.

# Pre-registered verdict thresholds (E1_ROBUSTNESS_DESIGN.md).
PASS_MAX_DEATH = 0.10   # death rate <= 10% AND sig < random  -> PASS
FAIL_MIN_DEATH = 0.25   # death rate >= 25%                   -> FAIL (caveat)

# --- adversarial layouts (15x15) ---------------------------------------------
# All three put lava in ISOLATED cells, far from the swarm's natural mass, so a
# walker reaching one is a strong spatial outlier (the registered caveat).
LAYOUTS = {
    # one isolated 2x2 block, off the SW->NE corridor.
    "island": """
S..............
...............
...............
.........LL....
.........LL....
...............
...............
...............
...............
...............
...............
...............
...............
...............
..............G
""",
    # lava cluster in the top-right corner; path runs down the LEFT edge, so
    # the lava is irrelevant to the route -- any death is pure attractor.
    "spur": """
S..........LLLL
...........L..L
...........LLLL
...............
...............
...............
...............
...............
...............
...............
...............
...............
...............
...............
G..............
""",
    # several lone lava cells, each ringed by free space -> each a strong,
    # independent outlier (scatter taken to the isolated extreme).
    "archipelago": """
S..............
...............
.....L.........
...............
...............
...........L...
...............
...............
....L..........
...............
...............
..........L....
...............
...............
..............G
""",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Robust at k=0."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    policies = {"random": random_policy, "greedy": greedy_policy}
    for a in ALPHAS:
        policies[f"fmc_a{a}"] = make_fmc_policy(a)

    results = {}
    t0 = time.time()
    for li, (lname, ltext) in enumerate(LAYOUTS.items()):
        env, start = parse_layout(ltext)
        results[lname] = {}
        for pi, (pname, policy) in enumerate(policies.items()):
            outcomes = []
            for e in range(N_EPISODES):
                base = 5_000_000 + 1_000_000 * li + 10_000 * pi + e
                outcomes.append(run_episode(env, start, policy, base))
            deaths = outcomes.count("lava")
            goals = outcomes.count("goal")
            results[lname][pname] = {
                "n": N_EPISODES,
                "deaths": deaths,
                "goals": goals,
                "timeouts": outcomes.count("timeout"),
                "death_rate": deaths / N_EPISODES,
                "goal_rate": goals / N_EPISODES,
            }
            print(f"  [{time.time()-t0:6.1f}s] {lname:12s} {pname:10s} "
                  f"death={deaths:2d}/{N_EPISODES}  goal={goals:2d}/{N_EPISODES}")

    # --- report ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("E1-ROBUSTNESS RESULTS — adversarial isolated-lava geometry")
    print("=" * 78)
    print(f"{'layout':12s} {'policy':10s} {'death%':>8s} {'goal%':>8s} "
          f"{'vs random':>16s} {'vs greedy':>16s}")
    layout_verdict = {}
    for lname, lr in results.items():
        rand_d = lr["random"]["deaths"]
        grd_d = lr["greedy"]["deaths"]
        for pname, st in lr.items():
            line = (f"{lname:12s} {pname:10s} {st['death_rate']*100:7.1f}% "
                    f"{st['goal_rate']*100:7.1f}%")
            if pname.startswith("fmc"):
                zr, pr = two_prop_z(st["deaths"], st["n"], rand_d, st["n"])
                zg, pg = two_prop_z(st["deaths"], st["n"], grd_d, st["n"])
                line += f"   z={zr:+.2f} p={pr:.3f}   z={zg:+.2f} p={pg:.3f}"
            print(line)

        # --- pre-registered verdict on fmc_a0.0 ------------------------------
        a0 = lr["fmc_a0.0"]
        dr = a0["death_rate"]
        lo, hi = wilson_ci(a0["deaths"], a0["n"])
        zr, pr = two_prop_z(a0["deaths"], a0["n"], rand_d, a0["n"])
        sig_below_random = (a0["deaths"] < rand_d) and (pr / 2.0 < 0.05)
        if dr <= PASS_MAX_DEATH and sig_below_random:
            verdict = "PASS"
        elif dr >= FAIL_MIN_DEATH:
            verdict = "FAIL"
        else:
            verdict = "MARGINAL"
        layout_verdict[lname] = verdict
        print(f"  -> fmc_a0.0 death={dr*100:.1f}% "
              f"(Wilson 95% CI [{lo*100:.1f}%, {hi*100:.1f}%]) "
              f"| sig<random={sig_below_random} (1-sided p={pr/2:.3f}) "
              f"| verdict: {verdict}")
        print()

    # --- overall verdict ------------------------------------------------------
    n_pass = sum(v == "PASS" for v in layout_verdict.values())
    n_fail = sum(v == "FAIL" for v in layout_verdict.values())
    print("-" * 78)
    print("E1-ROBUSTNESS verdict (pre-registered, E1_ROBUSTNESS_DESIGN.md): " +
          ", ".join(f"{k}:{v}" for k, v in layout_verdict.items()))
    if n_pass == len(layout_verdict):
        print("  => CAVEAT REJECTED — E1 self-preservation is robust to "
              "isolated-lava geometry. Update MATH_CANON (downgrade caveat).")
    elif n_fail >= 1:
        print("  => CAVEAT CONFIRMED — E1 is geometry-dependent. Register the "
              "limit in MATH_CANON as a bound on Conjecture E.")
    else:
        print("  => PARTIAL robustness (MARGINAL, no FAIL). Report honestly.")

    out = _HERE / "results" / "e1_robustness.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "params": {"N": N_WALKERS, "M": HORIZON_M, "beta": BETA,
                   "alphas": ALPHAS, "n_episodes": N_EPISODES,
                   "max_steps": MAX_STEPS,
                   "pass_max_death": PASS_MAX_DEATH,
                   "fail_min_death": FAIL_MIN_DEATH},
        "results": results,
        "layout_verdict": layout_verdict,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
