"""Conjecture E — visual demonstrations (non-technical).

Two figures that make the phenomenon legible without statistics:

  e2_trajectories.png — who walks into the fire and who walks around it.
       3 layouts x 4 policies, a representative episode each.
  e2_swarm.png        — the mechanism: the FMC walker swarm "imagining" the
       future. At low alpha the cloud avoids the lava; at high alpha the goal
       pull drags it into the fire.

Run:  python work/12_conjecture_e/e2_visuals.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "fmc-core" / "src"))

from fmc.core import virtual_reward, clone_step  # noqa: E402
from gridworld_terminal import FREE, LAVA, GOAL, parse_layout  # noqa: E402
from e1_base import LAYOUTS  # noqa: E402

N_WALKERS, HORIZON_M, MAX_STEPS = 64, 20, 60

# colours: free / lava / goal
_CELL_CMAP = ListedColormap([(0.96, 0.96, 0.93),
                             (0.93, 0.30, 0.13),    # lava = fire orange-red
                             (0.27, 0.72, 0.36)])   # goal = green


# --- policies -----------------------------------------------------------------

def random_policy(env, s, seed):
    return int(np.random.default_rng(seed).integers(0, len(list(env.actions()))))


def greedy_policy(env, s, seed):
    rng = np.random.default_rng(seed)
    acts = list(env.actions())
    vals = np.array([env.reward(env.step(s, a)) for a in acts])
    best = np.flatnonzero(vals == vals.max())
    return int(acts[best[rng.integers(0, len(best))]])


def make_fmc(alpha, beta=1.0):
    from fmc.core import plan

    def pol(env, s, seed):
        return plan(env, s, N=N_WALKERS, M=HORIZON_M,
                    alpha=alpha, beta=beta, seed=seed)
    return pol


# --- episode path recording ---------------------------------------------------

def record_path(env, start, policy, base_seed):
    s = env.reset(start)
    path = [(s.r, s.c)]
    for t in range(MAX_STEPS):
        if s.done:
            break
        a = policy(env, s, base_seed * 100 + t)
        s = env.step(s, a)
        path.append((s.r, s.c))
    cell = env.cell(s)
    outcome = {LAVA: "morto", GOAL: "goal"}.get(cell, "salvo")
    return path, outcome


def representative_episode(env, start, policy, n_try=11):
    """Run several seeds, return a path whose outcome is the modal outcome."""
    runs = [record_path(env, start, policy, seed) for seed in range(n_try)]
    modal = Counter(o for _, o in runs).most_common(1)[0][0]
    rate = sum(o == modal for _, o in runs) / len(runs)
    for path, outcome in runs:
        if outcome == modal:
            return path, outcome, rate
    return runs[0][0], runs[0][1], rate


# --- swarm tracer (mirrors fmc.core.plan, returns the final swarm) ------------

def plan_trace(env, x0, N, M, alpha, beta, seed):
    """Identical iteration to fmc.core.plan() but returns the final walker
    states instead of the decision — for visualising the swarm."""
    rng = np.random.default_rng(seed)
    actions = list(env.actions())
    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array([actions[rng.integers(0, len(actions))] for _ in range(N)],
                      dtype=object)
    for t in range(M):
        for i in range(N):
            a = labels[i] if t == 0 else env.sample_action(states[i], rng)
            states[i] = env.step(states[i], a)
        rewards = np.array([env.reward(s) for s in states], dtype=np.float64)
        obs = np.stack([np.asarray(env.observe(s), np.float64).ravel()
                        for s in states])
        partners = rng.permutation(N)
        for i in range(N):
            if partners[i] == i:
                partners[i] = (i + 1) % N
        vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
        clone_idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in clone_idx]
        labels = labels[clone_idx]
    return states


# --- drawing helpers ----------------------------------------------------------

def _draw_grid(ax, env):
    ax.imshow(env.grid, cmap=_CELL_CMAP, vmin=0, vmax=2, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.7")


# --- figure 1: trajectories ---------------------------------------------------

def fig_trajectories():
    policies = [
        ("Caso\n(random)", lambda: random_policy),
        ("Avido\n(greedy)", lambda: greedy_policy),
        ("FMC alpha=0\npreservazione", lambda: make_fmc(0.0)),
        ("FMC alpha=1\ndesiderio", lambda: make_fmc(1.0)),
    ]
    names = list(LAYOUTS.keys())
    fig, axes = plt.subplots(len(names), len(policies),
                             figsize=(13.5, 10.2))
    for ri, lname in enumerate(names):
        env, start = parse_layout(LAYOUTS[lname])
        for ci, (pname, pol_factory) in enumerate(policies):
            ax = axes[ri, ci]
            _draw_grid(ax, env)
            path, outcome, rate = representative_episode(env, start, pol_factory())
            ys, xs = zip(*path)
            ax.plot(xs, ys, "-", color="#1f4e8c", lw=2.0, alpha=0.85, zorder=3)
            ax.scatter(xs, ys, s=14, color="#1f4e8c", alpha=0.6, zorder=3)
            ax.scatter([xs[0]], [ys[0]], s=130, marker="o",
                       color="#1f4e8c", edgecolor="white", zorder=5)
            end_style = {"morto": ("X", "#b30000", "MORTO nel fuoco"),
                         "goal": ("*", "#d4a017", "GOAL raggiunto"),
                         "salvo": ("o", "0.35", "sopravvissuto")}
            mk, col, _ = end_style[outcome]
            ax.scatter([xs[-1]], [ys[-1]], s=340, marker=mk, color=col,
                       edgecolor="white", linewidth=1.3, zorder=6)
            if ri == 0:
                ax.set_title(pname, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(lname, fontsize=12, fontweight="bold")
            tag = {"morto": "morto", "goal": "goal", "salvo": "salvo"}[outcome]
            ax.text(0.5, -0.07, f"esito esempio: {tag}  ({rate:.0%} dei casi)",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                    color=col)
    fig.suptitle("Chi cammina nel fuoco e chi lo evita\n"
                 "linea blu = percorso dell'agente   "
                 "rosso = lava (morte)   verde = obiettivo   "
                 "cerchio blu = partenza",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(_HERE / "results" / "e2_trajectories.png", dpi=130)
    plt.close(fig)
    print("wrote e2_trajectories.png")


# --- figure 2: the swarm ------------------------------------------------------

def fig_swarm():
    env, start = parse_layout(LAYOUTS["lake"])
    x0 = env.reset(start)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for ax, alpha, label in [
            (axes[0], 0.0, "alpha = 0  (preservazione pura)"),
            (axes[1], 1.0, "alpha = 1  (desiderio: goal-seeking)")]:
        _draw_grid(ax, env)
        swarm = plan_trace(env, x0, N=N_WALKERS, M=HORIZON_M,
                           alpha=alpha, beta=1.0, seed=0)
        rs = np.array([s.r for s in swarm], float)
        cs = np.array([s.c for s in swarm], float)
        in_lava = np.array([env.grid[s.r, s.c] == LAVA for s in swarm])
        jit = lambda v: v + np.random.default_rng(1).uniform(-0.28, 0.28, len(v))
        ax.scatter(jit(cs[~in_lava]), jit(rs[~in_lava]), s=55,
                   color="#1f6fb3", edgecolor="white", linewidth=0.4,
                   alpha=0.85, zorder=4, label="walker salvo")
        ax.scatter(jit(cs[in_lava]), jit(rs[in_lava]), s=55,
                   color="#b30000", edgecolor="white", linewidth=0.4,
                   alpha=0.9, zorder=4, label="walker nel fuoco")
        ax.scatter([start[1]], [start[0]], s=170, marker="o",
                   color="black", edgecolor="white", zorder=5, label="partenza")
        n_lava = int(in_lava.sum())
        ax.set_title(f"{label}\n{n_lava}/{N_WALKERS} walker finiscono nel fuoco",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper center", fontsize=8, ncol=3,
                  bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Come 'pensa' FMC: 64 walker immaginano il futuro per UNA mossa\n"
                 "lo sciame che evita il fuoco vince il voto -> l'agente va al sicuro",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(_HERE / "results" / "e2_swarm.png", dpi=130)
    plt.close(fig)
    print("wrote e2_swarm.png")


if __name__ == "__main__":
    (_HERE / "results").mkdir(exist_ok=True)
    fig_trajectories()
    fig_swarm()
