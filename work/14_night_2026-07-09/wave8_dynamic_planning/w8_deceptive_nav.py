#!/usr/bin/env python3
"""
W8 -- the missing positive: a DYNAMICAL PLANNING domain with a WEAK incumbent
where FMC-base should actually WIN (not just tie). Mirror image of W7 (NK): there
we dialled ruggedness and FMC never beat greedy on a *static* landscape; here we
dial DECEPTION on a *dynamical* task and ask whether FMC beats standard planners.

Environment: DeceptiveNav -- a point mass with momentum in 2D must reach a goal
above a wall; the only gap in the wall is offset sideways. The reward is the
NAIVE -distance-to-goal, which is DECEPTIVE: it pulls straight into the wall
(a local optimum = pressed against the wall at min distance). Reaching the goal
requires temporarily moving AWAY (larger distance) to find the lateral gap.
- Momentum + wall collisions => rich divergent forward dynamics => E2 should FIRE.
- Deception knob = gap offset. offset=0 (gap on the straight path, easy) ...
  large offset (must detour far; greedy/shooting planners get trapped).

FMC's causal-entropy term (beta, the distance/anti-collapse term) keeps the swarm
spread and exploring laterally even though reward pulls to the wall -- the exact
mechanism that should escape the deceptive local optimum. Standard planners that
only maximise reward (greedy, random-shooting MPC, CEM) have no such pressure.

Baselines (weak, standard), all at MATCHED sim-call budget B_dec = N*M per decision:
  - greedy 1-step      : pick action with best next-state reward (naive floor)
  - random-shooting MPC: N sampled length-M action sequences, take best final reward
  - CEM                : cross-entropy method over action sequences (stronger MPC)
  - FMC                : fmc.core.plan(N, M) closed-loop
Golden rule respected: E2 divergence gate measured FIRST.
"""

import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from fmc.core import plan                       # noqa: E402
from w34_e2_smoke import e2_divergence          # noqa: E402

# 8 thrust directions + no-op
_TH = [(0, 0)] + [(np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)) for k in range(8)]
THRUSTS = np.array(_TH, dtype=np.float64)


class DeceptiveNav:
    """Dynamical deceptive navigation. State = np.array([x, y, vx, vy])."""

    def __init__(self, offset=0.0, W=10.0, Hgt=10.0, dt=0.2, drag=0.5,
                 a_mag=1.2, vmax=3.0, wall_y=5.0, gap_w=1.4, goal_r=0.6,
                 reward_mode="dense"):
        self.W, self.Hgt, self.dt, self.drag = W, Hgt, dt, drag
        self.a_mag, self.vmax = a_mag, vmax
        self.wall_y, self.gap_w = wall_y, gap_w
        self.goal_r = goal_r
        self.reward_mode = reward_mode      # 'dense' = -distance (deceptive);
        #                                     'sparse' = goal bonus only (hard-explore)
        self.start = np.array([W / 2, 1.0, 0.0, 0.0])
        self.goal = np.array([W / 2, Hgt - 1.0])
        self.gap_cx = np.clip(W / 2 + offset, gap_w, W - gap_w)  # gap centre

    # --- Environment protocol ---
    def actions(self):
        return list(range(len(THRUSTS)))

    def reset(self):
        return self.start.copy()

    def clone_state(self, s):
        return s.copy()

    def step(self, s, a):
        s = s.copy()
        x, y, vx, vy = s
        tx, ty = THRUSTS[a]
        vx += (tx * self.a_mag - self.drag * vx) * self.dt
        vy += (ty * self.a_mag - self.drag * vy) * self.dt
        sp = np.hypot(vx, vy)
        if sp > self.vmax:
            vx *= self.vmax / sp
            vy *= self.vmax / sp
        nx = np.clip(x + vx * self.dt, 0, self.W)
        ny = y + vy * self.dt
        # wall collision (crossing wall_y outside the gap => blocked)
        if (y - self.wall_y) * (ny - self.wall_y) < 0:      # crossing
            t = (self.wall_y - y) / (ny - y) if ny != y else 0.0
            xc = x + (nx - x) * t
            if abs(xc - self.gap_cx) > self.gap_w / 2:      # not in gap => block
                ny = self.wall_y - np.sign(vy) * 1e-2
                vy = 0.0
        ny = np.clip(ny, 0, self.Hgt)
        return np.array([nx, ny, vx, vy])

    def observe(self, s):
        return s.copy()

    def reward(self, s):
        d = np.hypot(s[0] - self.goal[0], s[1] - self.goal[1])
        if self.reward_mode == "sparse":
            # goal bonus only + a small "past-the-wall" landmark so the signal is
            # sparse but not a single point measure (still no gradient to the goal).
            r = 0.0
            if s[1] > self.wall_y:          # crossed the wall (rare under random play)
                r += 0.05
            if d < self.goal_r:
                r += 1.0
            return r
        r = -d                               # dense, deceptive
        if d < self.goal_r:
            r += 100.0
        return r

    def at_goal(self, s):
        return np.hypot(s[0] - self.goal[0], s[1] - self.goal[1]) < self.goal_r

    def sample_action(self, s, rng):
        return int(rng.integers(0, len(THRUSTS)))


# =========================================================================
# Planners -- each maps (env, state, N, M, rng) -> action, using ~N*M sim calls
# =========================================================================
def plan_greedy(env, s, N, M, rng):
    best_a, best_r = 0, -1e9
    for a in env.actions():
        r = env.reward(env.step(s, a))
        if r > best_r:
            best_r, best_a = r, a
    return best_a


def plan_random_shooting(env, s, N, M, rng):
    """N random length-M action sequences; pick first action of best final reward."""
    A = len(env.actions())
    best_a, best_val = 0, -1e9
    for _ in range(N):
        seq = rng.integers(0, A, size=M)
        st = s.copy()
        for a in seq:
            st = env.step(st, int(a))
        val = env.reward(st)
        if val > best_val:
            best_val, best_a = val, int(seq[0])
    return best_a


def plan_cem(env, s, N, M, rng, iters=3, elite_frac=0.25):
    """Cross-entropy method over action sequences (categorical dist per step).
    Total sim calls ~ iters * pop * M ; pop set so iters*pop = N."""
    A = len(env.actions())
    pop = max(N // iters, 4)
    logits = np.zeros((M, A))
    best_a, best_val = 0, -1e9
    for _ in range(iters):
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        seqs = np.stack([[rng.choice(A, p=probs[t]) for t in range(M)]
                         for _ in range(pop)])
        vals = np.empty(pop)
        for i in range(pop):
            st = s.copy()
            for a in seqs[i]:
                st = env.step(st, int(a))
            vals[i] = env.reward(st)
            if vals[i] > best_val:
                best_val, best_a = vals[i], int(seqs[i, 0])
        n_elite = max(int(pop * elite_frac), 2)
        elite = seqs[np.argsort(vals)[-n_elite:]]
        logits = np.zeros((M, A))
        for t in range(M):
            for a in range(A):
                logits[t, a] = np.log((elite[:, t] == a).mean() + 1e-2)
    return best_a


def plan_fmc(env, s, N, M, rng, alpha=1.0, beta=1.0):
    seed = int(rng.integers(0, 2**31 - 1))
    return plan(env, s, N=N, M=M, alpha=alpha, beta=beta, seed=seed)


# =========================================================================
# Closed-loop episode
# =========================================================================
def run_episode(env, planner, N, M, H, rng, **kw):
    s = env.reset()
    min_d = np.hypot(s[0] - env.goal[0], s[1] - env.goal[1])
    for t in range(H):
        if env.at_goal(s):
            return dict(success=True, steps=t, min_d=0.0)
        a = planner(env, s, N, M, rng, **kw)
        s = env.step(s, a)
        min_d = min(min_d, np.hypot(s[0] - env.goal[0], s[1] - env.goal[1]))
    return dict(success=bool(env.at_goal(s)), steps=H, min_d=float(min_d))


PLANNERS = {"greedy": plan_greedy, "rand-shoot": plan_random_shooting,
            "CEM": plan_cem, "FMC": plan_fmc}


def run(offsets=(0.0, 1.5, 2.5, 3.5), instances=12, N=48, M=12, H=45,
        reward_mode="dense"):
    Bdec = N * M
    print("=" * 92)
    print(f"W8 -- DeceptiveNav [{reward_mode}] | N={N} M={M} H={H}  "
          f"matched budget B_dec={Bdec} sim-calls/decision")
    print("=" * 92)
    print(f"{'offset':>7} {'E2 disp':>8} {'E2':>9} | "
          + " ".join(f"{k+' succ':>12}" for k in PLANNERS))
    print("-" * 92)
    rows = []
    for off in offsets:
        env0 = DeceptiveNav(offset=off, reward_mode=reward_mode)
        e2 = e2_divergence(env0, env0.reset(), N=48, M=M, seeds=(0, 1, 2))
        succ = {k: [] for k in PLANNERS}
        steps = {k: [] for k in PLANNERS}
        for inst in range(instances):
            for k, fn in PLANNERS.items():
                rng = np.random.default_rng(20260712 + inst * 17 + int(off * 10))
                env = DeceptiveNav(offset=off, reward_mode=reward_mode)
                res = run_episode(env, fn, N, M, H, rng)
                succ[k].append(res["success"])
                steps[k].append(res["steps"] if res["success"] else H)
        verdict = "DIVERGE" if e2["disp_ratio"] >= 3.0 else "collapse"
        srates = {k: float(np.mean(succ[k])) for k in PLANNERS}
        print(f"{off:>7.1f} {e2['disp_ratio']:>8.2f} {verdict:>9} | "
              + " ".join(f"{srates[k]:>12.2f}" for k in PLANNERS))
        rows.append((off, e2["disp_ratio"], srates,
                     {k: float(np.mean(steps[k])) for k in PLANNERS}))
    print("-" * 92)
    print("succ = fraction of instances that reached the goal within H steps (matched budget).")
    print("=" * 92)
    return rows


if __name__ == "__main__":
    run()
