"""Conjecture E — E1-robustness mechanism diagnostic.

The geometry caveat is a MECHANISM claim, in two links:

  (1) a walker on isolated lava is a spatial outlier -> high VR at alpha=0;
  (2) therefore its frozen/dead state + t=0 label propagate by cloning and the
      swarm is pulled onto the lava.

e1_robustness.py showed the black-box outcome (0% death). This script opens the
box: it replicates the fmc-core plan() loop (WITHOUT modifying the kernel -- it
only re-runs the public fmc.core functions to observe) from a probe state placed
DIRECTLY ADJACENT to an isolated lava cell, the worst case for the caveat.

It records, per horizon step: how many walkers sit on lava, the mean VR of
lava-walkers vs free-walkers (tests link 1), and the swarm fraction whose t=0
label is the into-lava action (tests link 2 -> what decide() votes).

Run:  python work/12_conjecture_e/e1_robustness_diag.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from e1_base import HORIZON_M, N_WALKERS  # noqa: E402  (sets up fmc-core path)
from fmc.core import decide, virtual_reward  # noqa: E402
from gridworld_terminal import State, parse_layout  # noqa: E402
from e1_robustness import LAYOUTS  # noqa: E402

ALPHA, BETA = 0.0, 1.0
INTO_LAVA_ACTION = 3  # 'right': from the probe cell, steps onto the lava cell.
N_PROBES = 12


def instrument_plan(env, x0, n, m, alpha, beta, seed):
    """Replicate fmc-core plan() and observe lava-walker VR + label dynamics.

    Returns (decided_action, per-step trace). Logic mirrors fmc/core.py:plan
    exactly; nothing in fmc-core is modified.
    """
    rng = np.random.default_rng(seed)
    actions = list(env.actions())
    states = [env.clone_state(x0) for _ in range(n)]
    labels = np.array([actions[rng.integers(0, len(actions))] for _ in range(n)],
                      dtype=object)
    trace = []
    for t in range(m):
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

        on_lava = np.array([env.cell(s) == 1 for s in states])
        trace.append({
            "t": t,
            "n_lava": int(on_lava.sum()),
            "vr_lava": float(vr[on_lava].mean()) if on_lava.any() else float("nan"),
            "vr_free": float(vr[~on_lava].mean()) if (~on_lava).any() else float("nan"),
            "lava_label_frac": float(np.mean(labels == INTO_LAVA_ACTION)),
        })

        # RESAMPLE (clone_step replicated; same kernel logic).
        cp = rng.integers(0, n, size=n)
        same = cp == np.arange(n)
        while same.any():
            cp[same] = rng.integers(0, n, size=int(same.sum()))
            same = cp == np.arange(n)
        vr_safe = np.where(vr > 0, vr, 1e-10)
        p_clone = np.clip((vr[cp] - vr) / vr_safe, 0.0, 1.0)
        p_clone[vr == 0] = 1.0
        will = rng.random(n) < p_clone
        clone_idx = np.where(will, cp, np.arange(n))
        states = [env.clone_state(states[k]) for k in clone_idx]
        labels = labels[clone_idx]
    return decide(labels), trace


def main():
    env, _ = parse_layout(LAYOUTS["archipelago"])
    # Probe: cell directly LEFT of the isolated lava cell at (2,5).
    # action 'right' (3) from here steps straight onto lava. Worst case.
    probe = State(2, 4, done=False)
    assert env.cell(State(2, 5, False)) == 1, "expected lava at (2,5)"

    print("E1-ROBUSTNESS mechanism diagnostic — archipelago, probe at (2,4),")
    print("directly adjacent to isolated lava (2,5). alpha=0, beta=1, N=64, M=20.")
    print("=" * 74)

    into_lava_votes = 0
    agg = []
    for p in range(N_PROBES):
        act, trace = instrument_plan(env, probe, N_WALKERS, HORIZON_M,
                                     ALPHA, BETA, seed=7_000_000 + p)
        into_lava_votes += (act == INTO_LAVA_ACTION)
        agg.append(trace)
        last = trace[-1]
        print(f"  probe {p:2d}: decided action={act} "
              f"({'INTO LAVA' if act == INTO_LAVA_ACTION else 'safe'})  "
              f"| final: {last['n_lava']:2d}/64 walkers on lava, "
              f"into-lava label frac={last['lava_label_frac']:.2f}")

    # --- aggregate the two mechanism links over horizon ----------------------
    ts, n_lava_a, ratio_a, frac_a = [], [], [], []
    for t in range(HORIZON_M):
        rows = [tr[t] for tr in agg]
        n_lava = np.mean([r["n_lava"] for r in rows])
        vl = np.nanmean([r["vr_lava"] for r in rows])
        vf = np.nanmean([r["vr_free"] for r in rows])
        ratio = vl / vf if vf and not np.isnan(vf) and not np.isnan(vl) else float("nan")
        frac = np.mean([r["lava_label_frac"] for r in rows])
        ts.append(t); n_lava_a.append(n_lava); ratio_a.append(ratio); frac_a.append(frac)

    print("\nHorizon-averaged over %d probes (link 1 = is a lava-walker a"
          " high-VR outlier?):" % N_PROBES)
    print(f"  {'t':>3s} {'n_lava':>7s} {'VR ratio':>9s} {'lava-label%':>12s}")
    for t in ts:
        if t % 4 == 0 or t == HORIZON_M - 1:
            print(f"  {t:3d} {n_lava_a[t]:7.1f} {ratio_a[t]:9.2f} "
                  f"{frac_a[t]*100:11.1f}%")

    # --- mechanism figure ----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    c1, c2 = "#c0392b", "#2c3e50"
    ax1.axhline(1.0, color="gray", ls=":", lw=1)
    ax1.plot(ts, ratio_a, "o-", color=c1, lw=2, label="VR ratio (lava / free)")
    ax1.set_xlabel("FMC planning horizon step  t")
    ax1.set_ylabel("VR ratio  (lava-walker / free-walker)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(0.0, 1.3)
    ax2 = ax1.twinx()
    ax2.plot(ts, [f * 100 for f in frac_a], "s-", color=c2, lw=2,
             label="swarm with into-lava t=0 label")
    ax2.set_ylabel("into-lava t=0 label  (% of swarm)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.set_ylim(0, 25)
    ax1.set_title("Why E1 survives isolated lava: the absorbing cell is a VR SINK\n"
                  "caveat predicted ratio > 1 (attractor) — measured: ratio < 1, "
                  "into-lava lineage selected out", fontsize=9.5)
    lines = ax1.get_lines()[1:] + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right",
               fontsize=8.5)
    fig.tight_layout()
    figpath = _HERE / "results" / "e1_robustness_mechanism.png"
    figpath.parent.mkdir(exist_ok=True)
    fig.savefig(figpath, dpi=130)
    print(f"\nwrote {figpath}")

    print("\n" + "=" * 74)
    print(f"Decided 'into-lava' in {into_lava_votes}/{N_PROBES} probes "
          f"(adjacent to lava, no goal pull at alpha=0).")
    print("Link 1 (lava-walker = high-VR outlier): see VR ratio column.")
    print("Link 2 (therefore the swarm is pulled in): see whether the into-lava")
    print("label fraction grows toward a majority and whether decide() votes it.")


if __name__ == "__main__":
    main()
