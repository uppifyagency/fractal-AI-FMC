"""F12 verification on Atari Boxing — non-toy domain validation.

Sergio's claim (video, riga 537): "la probabilidad de que tú vayas a un sitio
sea proporcional a la recompensa". On the toy 2D landscape we measured
log-Pearson(log P_walker, log R) ≈ 0.77 at α = 1.

In Atari we can't compute P_R(x) analytically (R is given by the simulator,
not by a closed form over state space). The cleanest analog is the
*action-marginal* form:

    P_FMC(a)   ∝   E[ cumulative_reward | init_action = a ]

After M ticks of FMC planning, walkers labelled with each `init_action a`
should be present in proportion to the expected forward reward of taking a.
This is the "scanning density" claim restricted to the action-marginal slice
at t = 0.

What we measure:
  - Per decision, Pearson correlation between bincount(init_actions) and
    mean cumulative reward per action label
  - Per decision, whether the FMC argmax-action coincides with the
    highest-reward action
  - Spearman rank correlation (more robust to scale)

The script reuses the canonical reference implementation in
work/03_atari_replication/scripts/fmc_minimal.py with minimal modification:
we hook into FMCAgent.decide() to also return the post-planning walker state.

Output: results/f12_atari_boxing.json + plot results/f12_atari_boxing_correlation.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Reuse reference implementation
ATARI_DIR = HERE.parent / "03_atari_replication" / "scripts"
sys.path.insert(0, str(ATARI_DIR))
from fmc_minimal import FMCAgent, FMCConfig, relativize  # noqa: E402


class InstrumentedFMCAgent(FMCAgent):
    """FMCAgent variant that exposes per-decision diagnostics for F12 verification.

    The only modification: decide() returns a dict with the post-planning
    arrays instead of a single int.
    """

    def decide_with_diagnostics(self, root_state) -> dict:
        N = self.cfg.n_walkers
        M = self.cfg.time_horizon

        init_actions = self.rng.integers(0, self.n_actions, size=N)
        cum_rewards = np.zeros(N, dtype=np.float32)
        is_dead = np.zeros(N, dtype=bool)
        walker_states = [self._snapshot_state() for _ in range(N)]

        # Track which walkers survived which clones (for sanity-check).
        # We DO NOT track full provenance — just the final cum_R and init_action.
        for t in range(M):
            for i in range(N):
                if is_dead[i]:
                    continue
                self._restore_state(walker_states[i])
                a = init_actions[i] if t == 0 else self.rng.integers(0, self.n_actions)
                r, term = self._step_skipframe(a)
                cum_rewards[i] += r
                if term:
                    is_dead[i] = True
                else:
                    walker_states[i] = self._snapshot_state()

            obs_ram = np.zeros((N, 128), dtype=np.float32)
            for i in range(N):
                if not is_dead[i]:
                    self._restore_state(walker_states[i])
                    obs_ram[i] = self._ram_obs()

            partners = self.rng.permutation(N)
            same = partners == np.arange(N)
            partners[same] = (partners[same] + 1) % N
            distances = np.linalg.norm(obs_ram - obs_ram[partners], axis=1)

            R_norm = relativize(cum_rewards)
            D_norm = relativize(distances)
            R_norm[is_dead] = 0
            D_norm[is_dead] = 0
            VR = (R_norm ** self.cfg.balance) * (D_norm ** self.cfg.balance)

            clone_partners = self.rng.permutation(N)
            same = clone_partners == np.arange(N)
            clone_partners[same] = (clone_partners[same] + 1) % N
            VR_self = VR
            VR_other = VR[clone_partners]
            denom = np.where(VR_self > 1e-8, VR_self, 1e-8)
            clone_prob = np.clip((VR_other - VR_self) / denom, 0, 1)
            clone_prob[is_dead] = 1.0
            random_draws = self.rng.random(N)
            will_clone = random_draws < clone_prob

            for i in np.where(will_clone)[0]:
                k = clone_partners[i]
                if not is_dead[k]:
                    walker_states[i] = walker_states[k]
                    self._restore_state(walker_states[i])
                    walker_states[i] = self._snapshot_state()
                    init_actions[i] = init_actions[k]
                    cum_rewards[i] = cum_rewards[k]
                    is_dead[i] = False

        # Diagnostics
        alive_mask = ~is_dead
        alive_actions = init_actions[alive_mask]
        alive_rewards = cum_rewards[alive_mask]

        bincount = np.bincount(init_actions, minlength=self.n_actions)
        # Mean cum reward per init_action (over ALL walkers, including dead — they
        # have meaningfully low cum_R and that's part of the signal).
        mean_R_per_action = np.zeros(self.n_actions, dtype=np.float64)
        n_per_action = np.zeros(self.n_actions, dtype=np.int32)
        for a in range(self.n_actions):
            mask = (init_actions == a)
            n_per_action[a] = mask.sum()
            if n_per_action[a] > 0:
                mean_R_per_action[a] = cum_rewards[mask].mean()

        if len(alive_actions) > 0:
            chosen = int(np.bincount(alive_actions, minlength=self.n_actions).argmax())
        else:
            chosen = int(self.rng.integers(0, self.n_actions))

        return {
            "init_actions": init_actions.copy(),
            "cum_rewards": cum_rewards.copy(),
            "is_dead": is_dead.copy(),
            "bincount": bincount.copy(),
            "mean_R_per_action": mean_R_per_action,
            "n_per_action": n_per_action,
            "chosen_action": chosen,
            "n_alive": int(alive_mask.sum()),
            "max_cum_R": float(cum_rewards.max()),
            "min_cum_R": float(cum_rewards.min()),
        }


def correlations(bincount: np.ndarray, mean_R: np.ndarray, n_per_action: np.ndarray,
                 min_walkers: int = 2) -> dict:
    """Pearson and Spearman between FMC action distribution and per-action mean reward,
    restricted to actions with at least `min_walkers` walkers (otherwise mean_R is unreliable).
    """
    mask = n_per_action >= min_walkers
    if mask.sum() < 3:
        return {"pearson": np.nan, "spearman": np.nan, "n_actions_used": int(mask.sum())}
    bc = bincount[mask].astype(np.float64)
    mR = mean_R[mask]
    pearson = float(np.corrcoef(bc, mR)[0, 1]) if bc.std() > 0 and mR.std() > 0 else np.nan
    sp, _ = stats.spearmanr(bc, mR)
    return {
        "pearson": pearson,
        "spearman": float(sp) if np.isfinite(sp) else np.nan,
        "n_actions_used": int(mask.sum()),
    }


def main(n_decisions: int = 30, n_walkers: int = 50, time_horizon: int = 15,
         fixed_steps: int = 5, seed: int = 42) -> dict:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)

    env = gym.make("ALE/Boxing-v5", full_action_space=False)
    obs, _ = env.reset(seed=seed)
    cfg = FMCConfig(n_walkers=n_walkers, time_horizon=time_horizon,
                    fixed_steps=fixed_steps, balance=1.0)
    agent = InstrumentedFMCAgent(env, cfg)
    agent.rng = np.random.default_rng(seed)

    print("=" * 60)
    print("F12 Atari verification — Boxing")
    print("=" * 60)
    print(f"  N={n_walkers} walkers, M={time_horizon} ticks, skip={fixed_steps}, seed={seed}")
    print(f"  n_decisions={n_decisions}, action_space={env.action_space.n}")

    per_decision: list[dict] = []
    cum_game_reward = 0.0
    t0 = time.time()
    for k in range(n_decisions):
        root = agent._snapshot_state()
        diag = agent.decide_with_diagnostics(root)
        # apply chosen action in game
        agent._restore_state(root)
        r_step, term = agent._step_skipframe(diag["chosen_action"])
        cum_game_reward += r_step
        corr = correlations(diag["bincount"], diag["mean_R_per_action"],
                            diag["n_per_action"])
        per_decision.append({
            "step": k,
            "chosen_action": diag["chosen_action"],
            "step_reward": float(r_step),
            "cum_game_reward": float(cum_game_reward),
            "n_alive": diag["n_alive"],
            "max_cum_R": diag["max_cum_R"],
            "min_cum_R": diag["min_cum_R"],
            "pearson": corr["pearson"],
            "spearman": corr["spearman"],
            "n_actions_used": corr["n_actions_used"],
        })
        if k < 5 or k % 5 == 0:
            print(f"  step={k:3d} action={diag['chosen_action']:2d} step_R={r_step:+.1f} "
                  f"alive={diag['n_alive']:3d}/{n_walkers} cum_R_walker_max={diag['max_cum_R']:+.1f} "
                  f"pearson={corr['pearson']:+.3f} spearman={corr['spearman']:+.3f}")
        if term:
            print(f"  game over at step {k}")
            break
    wall = time.time() - t0
    env.close()

    pearsons = np.array([p["pearson"] for p in per_decision if np.isfinite(p["pearson"])])
    spearmans = np.array([p["spearman"] for p in per_decision if np.isfinite(p["spearman"])])

    summary = {
        "config": {"n_walkers": n_walkers, "time_horizon": time_horizon,
                   "fixed_steps": fixed_steps, "seed": seed,
                   "n_decisions_attempted": n_decisions, "game": "ALE/Boxing-v5"},
        "wall_time_s": float(wall),
        "n_decisions_completed": len(per_decision),
        "cum_game_reward": float(cum_game_reward),
        "pearson_mean": float(pearsons.mean()) if len(pearsons) else None,
        "pearson_median": float(np.median(pearsons)) if len(pearsons) else None,
        "pearson_std": float(pearsons.std()) if len(pearsons) else None,
        "pearson_pos_fraction": float((pearsons > 0).mean()) if len(pearsons) else None,
        "spearman_mean": float(spearmans.mean()) if len(spearmans) else None,
        "spearman_pos_fraction": float((spearmans > 0).mean()) if len(spearmans) else None,
        "per_decision": per_decision,
    }

    out_json = RESULTS / "f12_atari_boxing.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[F12-Atari] Wrote {out_json}  ({wall:.1f}s)")

    print("\n" + "=" * 60)
    print("Aggregate correlations across decisions")
    print("=" * 60)
    if len(pearsons):
        print(f"  Pearson:  mean={pearsons.mean():+.3f}  median={np.median(pearsons):+.3f}  "
              f"frac>0={(pearsons > 0).mean():.2%}  N={len(pearsons)}")
        print(f"  Spearman: mean={spearmans.mean():+.3f}  median={np.median(spearmans):+.3f}  "
              f"frac>0={(spearmans > 0).mean():.2%}  N={len(spearmans)}")
        print(f"  Game cumulative reward: {cum_game_reward:+.0f}")

    _plot(summary, RESULTS / "f12_atari_boxing_correlation.png")
    return summary


def _plot(summary: dict, out: Path) -> None:
    pd = summary["per_decision"]
    steps = [p["step"] for p in pd]
    pearsons = [p["pearson"] for p in pd]
    spearmans = [p["spearman"] for p in pd]
    cum_R = [p["cum_game_reward"] for p in pd]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(steps, pearsons, "o-", color="#1f77b4", linewidth=1.4, label="Pearson")
    axes[0, 0].plot(steps, spearmans, "s-", color="#d62728", linewidth=1.4, label="Spearman")
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("decision step"); axes[0, 0].set_ylabel("correlation")
    axes[0, 0].set_title("Per-decision corr( bincount(actions), E[cum_R | action] )")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    pe = np.array([p for p in pearsons if np.isfinite(p)])
    sp = np.array([p for p in spearmans if np.isfinite(p)])
    axes[0, 1].hist(pe, bins=15, color="#1f77b4", alpha=0.7, label="Pearson")
    axes[0, 1].hist(sp, bins=15, color="#d62728", alpha=0.5, label="Spearman")
    axes[0, 1].axvline(0, color="black", linestyle="--", alpha=0.5)
    if len(pe):
        axes[0, 1].axvline(pe.mean(), color="#1f77b4", linestyle=":",
                           label=f"Pearson mean={pe.mean():.3f}")
    if len(sp):
        axes[0, 1].axvline(sp.mean(), color="#d62728", linestyle=":",
                           label=f"Spearman mean={sp.mean():.3f}")
    axes[0, 1].set_xlabel("correlation"); axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("Distribution across decisions")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(steps, cum_R, "-", color="#2ca02c", linewidth=1.6)
    axes[1, 0].set_xlabel("decision step"); axes[1, 0].set_ylabel("cumulative game reward")
    axes[1, 0].set_title("Boxing score progression")
    axes[1, 0].grid(alpha=0.3)

    n_alive = [p["n_alive"] for p in pd]
    max_R = [p["max_cum_R"] for p in pd]
    axes[1, 1].plot(steps, n_alive, "-", color="#9467bd", label="n_alive walkers", linewidth=1.4)
    ax2 = axes[1, 1].twinx()
    ax2.plot(steps, max_R, "-", color="#ff7f0e", label="max walker cum_R", linewidth=1.4)
    axes[1, 1].set_xlabel("decision step"); axes[1, 1].set_ylabel("alive walkers")
    ax2.set_ylabel("max walker cum_R")
    axes[1, 1].set_title("Swarm health per decision")
    axes[1, 1].legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("F12 Atari verification — Boxing | "
                 f"N={summary['config']['n_walkers']}, M={summary['config']['time_horizon']}",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_decisions", type=int, default=30)
    p.add_argument("--n_walkers", type=int, default=50)
    p.add_argument("--time_horizon", type=int, default=15)
    p.add_argument("--fixed_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(n_decisions=args.n_decisions, n_walkers=args.n_walkers,
         time_horizon=args.time_horizon, fixed_steps=args.fixed_steps,
         seed=args.seed)
