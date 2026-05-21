"""Conjecture E / P13 — proxy experiment: does sparse interrogation degrade FMC search?

Pre-registered design: P13_DESIGN.md  (read it first, especially section 6).

P13 asks whether an O(N) sparse LLM-world-model interrogation degrades FMC search
(risk R2). R2 is testable WITHOUT any LLM: emulate each sparse schema on the true
gridworld kernel and measure whether the FMC decision holds vs the FULL control.

Arms (P13_DESIGN section 6.2):
  FULL          true kernel at every (walker, tick)        -- control = E1-robustness
  S1(eta,abs)   tick 0 true; ticks >=1 a corrupted surrogate:
                  - VR signal (reward, observation) + Gaussian(0, eta)
                  - abs=preserved: lava still absorbing inside the rollout
                  - abs=broken:    surrogate ignores absorption (lava passable)
  S2(B)         FMC runs entirely on M_hat, a table distilled from B offline
                samples of the true kernel; unseen (state,action) -> stay
  S3(k,stay)    FMC's action set restricted to a k-action macro-menu (top goal-ward),
                with ('s3_stay') or without ('s3_nostay') a guaranteed 'stay' action

Kernel fmc-core is UNCHANGED: proxy_plan replicates the public plan() loop and
calls fmc.core.{virtual_reward, clone_step, decide}; only the world-model is
swapped. proxy_plan(FullSchema) is asserted bit-identical to fmc.core.plan().

Metrics (P13_DESIGN section 6.3): death/goal rate (Wilson CI), decision-agreement
vs FULL, VR-rank Spearman vs FULL. alpha in {0.0, 0.1}, beta=1, N=64, M=20,
n=30 episodes/cell, 3 adversarial layouts (reused from e1_robustness).

Run:  python work/12_conjecture_e/p13_proxy.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from e1_base import BETA, HORIZON_M, MAX_STEPS, N_WALKERS  # noqa: E402 (sets fmc-core path)
from e1_robustness import LAYOUTS, wilson_ci  # noqa: E402
from gridworld_terminal import GOAL, LAVA, State, _DELTA, parse_layout  # noqa: E402
from fmc.core import clone_step, decide, plan, virtual_reward  # noqa: E402

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

# --- experiment parameters ---------------------------------------------------
ALPHAS = [0.0, 0.1]
N_EPISODES = 30
TABLE_SEED = 13
ETAS = [0.0, 1.0, 2.0, 4.0]
DISTILL_BUDGETS = [50, 150, 400, 1125]   # 1125 = 15*15*5 = full (cell,action) coverage
# Pre-registered R2 pass thresholds (P13_DESIGN section 6.5).
AGREE_MIN = 0.85          # decision-agreement vs FULL
DEATH_MARGIN = 0.05       # death rate must stay <= FULL + 5pp


# === world-model schemas =====================================================

def _lava_passable_step(env, s, a):
    """Surrogate that does NOT model absorption: lava passable, nothing terminal."""
    dr, dc = _DELTA[a]
    nr, nc = s.r + dr, s.c + dc
    if not (0 <= nr < env.H and 0 <= nc < env.W):
        nr, nc = s.r, s.c
    return State(nr, nc, done=False)


class FullSchema:
    """The control: true kernel everywhere."""
    def step(self, env, s, a, t, nrng):    return env.step(s, a)
    def reward(self, env, s, t, nrng):     return env.reward(s)
    def observe(self, env, s, t, nrng):    return env.observe(s)


class S1Schema:
    """Root true; ticks >=1 a noisy surrogate (P13_DESIGN S1)."""
    def __init__(self, eta: float, abs_preserved: bool):
        self.eta = eta
        self.abs_preserved = abs_preserved

    def step(self, env, s, a, t, nrng):
        if t == 0 or self.abs_preserved:
            return env.step(s, a)
        return _lava_passable_step(env, s, a)        # abs broken, t>=1

    def reward(self, env, s, t, nrng):
        r = env.reward(s)
        if t == 0 or self.eta == 0.0:
            return r
        return r + float(nrng.normal(0.0, self.eta))

    def observe(self, env, s, t, nrng):
        o = np.asarray(env.observe(s), dtype=np.float64)
        if t == 0 or self.eta == 0.0:
            return o
        return o + nrng.normal(0.0, self.eta, size=o.shape)


class S2Schema:
    """FMC runs entirely on a distilled table M_hat (P13_DESIGN S2)."""
    def __init__(self, table: dict):
        self.table = table

    def step(self, env, s, a, t, nrng):
        ns = self.table.get((s.r, s.c, a))
        if ns is None:
            return env.clone_state(s)                # unseen (x,a) -> stay
        return env.clone_state(ns)

    def reward(self, env, s, t, nrng):     return env.reward(s)
    def observe(self, env, s, t, nrng):    return env.observe(s)


def build_distill_table(env, budget: int, seed: int) -> dict:
    """Sample `budget` (state,action) transitions of the true kernel into a table."""
    rng = np.random.default_rng(seed)
    acts = list(env.actions())
    table = {}
    for _ in range(budget):
        r = int(rng.integers(0, env.H))
        c = int(rng.integers(0, env.W))
        a = acts[int(rng.integers(0, len(acts)))]
        s = State(r, c, done=bool(env._terminal(r, c)))
        table[(r, c, a)] = env.step(s, a)
    return table


class MacroEnv:
    """Wraps an env, restricting the action set to a macro-menu (P13_DESIGN S3)."""
    def __init__(self, base, menu):
        self.base = base
        self.menu = list(menu)

    def actions(self):                  return self.menu
    def clone_state(self, s):           return self.base.clone_state(s)
    def step(self, s, a):               return self.base.step(s, a)
    def observe(self, s):               return self.base.observe(s)
    def reward(self, s):                return self.base.reward(s)
    def sample_action(self, s, rng):    return self.menu[int(rng.integers(0, len(self.menu)))]


def macro_menu(env, root, k: int, include_stay: bool):
    """Top-k actions by goal-distance reduction (the 'oracle' macro-menu)."""
    acts = list(env.actions())
    gr, gc = env.goal

    def goalward(a):
        ns = env.step(root, a)
        return -(abs(ns.r - gr) + abs(ns.c - gc))    # higher = closer to goal

    ranked = sorted(acts, key=goalward, reverse=True)
    if include_stay:
        menu = [a for a in ranked if a != 4][:k - 1] + [4]
    else:
        menu = ranked[:k]
    return sorted(set(menu))


# === proxy plan: replicates fmc.core.plan, world-model swapped ================

def proxy_plan(env, x0, N, M, alpha, beta, seed, schema, vr_hook=None):
    """Replicate the public plan() loop of fmc.core with `schema` as world-model.

    Kernel math (virtual_reward, clone_step, decide) is the unmodified fmc-core.
    With schema=FullSchema and vr_hook=None this is bit-identical to fmc.core.plan
    (asserted in main). `vr_hook`, when given, is a callable (vr, rng) -> vr applied
    to each tick's VR vector before clone_step -- used by hP13-0 to degrade VR rank.
    Returns (decided_action, vr_trace) where vr_trace[t] is the tick-t VR vector.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng((0 if seed is None else int(seed)) ^ 0x5F3759DF)
    actions = list(env.actions())
    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array([actions[int(rng.integers(0, len(actions)))] for _ in range(N)],
                      dtype=object)
    vr_trace = []
    for t in range(M):
        for i in range(N):
            a = labels[i] if t == 0 else env.sample_action(states[i], rng)
            states[i] = schema.step(env, states[i], a, t, noise_rng)
        rewards = np.array([schema.reward(env, s, t, noise_rng) for s in states],
                           dtype=np.float64)
        obs = np.stack([np.asarray(schema.observe(env, s, t, noise_rng),
                                   dtype=np.float64).ravel() for s in states])
        partners = rng.permutation(N)
        for i in range(N):
            if partners[i] == i:
                partners[i] = (i + 1) % N
        vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
        if vr_hook is not None:
            vr = vr_hook(vr, noise_rng)
        vr_trace.append(vr.copy())
        clone_idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in clone_idx]
        labels = labels[clone_idx]
    return decide(labels), vr_trace


# === policies (env, state, seed) -> action ===================================

def full_policy(alpha):
    return lambda env, s, seed: plan(env, s, N=N_WALKERS, M=HORIZON_M,
                                     alpha=alpha, beta=BETA, seed=seed)

def s1_policy(alpha, eta, abs_preserved):
    sch = S1Schema(eta, abs_preserved)
    return lambda env, s, seed: proxy_plan(env, s, N_WALKERS, HORIZON_M,
                                           alpha, BETA, seed, sch)[0]

def s2_policy(alpha, table):
    sch = S2Schema(table)
    return lambda env, s, seed: proxy_plan(env, s, N_WALKERS, HORIZON_M,
                                           alpha, BETA, seed, sch)[0]

def s3_policy(alpha, k, include_stay):
    def pol(env, s, seed):
        menv = MacroEnv(env, macro_menu(env, s, k, include_stay))
        return proxy_plan(menv, s, N_WALKERS, HORIZON_M, alpha, BETA, seed,
                          FullSchema())[0]
    return pol


# === episode runner with inline decision-agreement ===========================

def run_episode_agree(env, start, arm_policy, ref_policy, base_seed):
    """Run one episode on the REAL env; record agreement of arm vs ref at each step."""
    s = env.reset(start)
    agree = []
    for t in range(MAX_STEPS):
        if s.done:
            break
        seed_t = base_seed * 100 + t
        a_arm = arm_policy(env, s, seed_t)
        if ref_policy is not None:
            agree.append(int(a_arm == ref_policy(env, s, seed_t)))
        s = env.step(s, a_arm)
    cell = env.cell(s)
    return {LAVA: "lava", GOAL: "goal"}.get(cell, "timeout"), agree


# === VR-rank diagnostic (hP13-0) =============================================

def vr_rank_diag(env, probes, alpha, eta):
    """Per-tick Spearman of S1(eta,abs-preserved) VR vs FULL VR, over probe states."""
    if spearmanr is None:
        return None
    full_sch, s1_sch = FullSchema(), S1Schema(eta, abs_preserved=True)
    per_tick = [[] for _ in range(HORIZON_M)]
    for pi, probe in enumerate(probes):
        seed = 8_200_000 + pi
        _, vf = proxy_plan(env, probe, N_WALKERS, HORIZON_M, alpha, BETA, seed, full_sch)
        _, vs = proxy_plan(env, probe, N_WALKERS, HORIZON_M, alpha, BETA, seed, s1_sch)
        for t in range(HORIZON_M):
            if np.std(vf[t]) < 1e-12 or np.std(vs[t]) < 1e-12:
                continue
            rho = float(spearmanr(vf[t], vs[t])[0])   # [0]: version-robust
            if not np.isnan(rho):
                per_tick[t].append(rho)
    means = [float(np.mean(x)) if x else float("nan") for x in per_tick]
    valid = [m for m in means if not np.isnan(m)]
    return {"mean": float(np.mean(valid)) if valid else float("nan"),
            "worst_tick": float(np.min(valid)) if valid else float("nan")}


def free_probes(env, n, seed):
    """Sample n free (non-terminal) cells as probe states."""
    rng = np.random.default_rng(seed)
    probes = []
    while len(probes) < n:
        r = int(rng.integers(0, env.H))
        c = int(rng.integers(0, env.W))
        if not env._terminal(r, c):
            probes.append(State(r, c, done=False))
    return probes


# === main ====================================================================

def main():
    t0 = time.time()

    # --- sanity: proxy_plan(FullSchema) must be bit-identical to fmc.core.plan ---
    env0, start0 = parse_layout(LAYOUTS["archipelago"])
    x0 = env0.reset(start0)
    ok = all(plan(env0, x0, N=N_WALKERS, M=HORIZON_M, alpha=0.1, beta=BETA, seed=sd)
             == proxy_plan(env0, x0, N_WALKERS, HORIZON_M, 0.1, BETA, sd,
                           FullSchema())[0]
             for sd in (1, 2, 3, 101, 777))
    print(f"sanity: proxy_plan(FULL) == fmc.core.plan() : {'OK' if ok else 'MISMATCH'}")
    assert ok, "proxy_plan FULL diverged from kernel plan() — control invalid"

    results = {}
    for li, (lname, ltext) in enumerate(LAYOUTS.items()):
        env, start = parse_layout(ltext)
        tables = {B: build_distill_table(env, B, TABLE_SEED) for B in DISTILL_BUDGETS}
        probes = free_probes(env, 10, seed=4242 + li)
        results[lname] = {}

        for ai, alpha in enumerate(ALPHAS):
            ref = full_policy(alpha)
            arms = {"FULL": (full_policy(alpha), None)}
            for eta in ETAS:
                arms[f"S1_eta{eta}_abs"] = (s1_policy(alpha, eta, True), ref)
                arms[f"S1_eta{eta}_brk"] = (s1_policy(alpha, eta, False), ref)
            for B in DISTILL_BUDGETS:
                arms[f"S2_B{B}"] = (s2_policy(alpha, tables[B]), ref)
            arms["S3_k3_stay"] = (s3_policy(alpha, 3, True), ref)
            arms["S3_k3_nostay"] = (s3_policy(alpha, 3, False), ref)

            cell = {}
            for arm_i, (aname, (pol, refp)) in enumerate(arms.items()):
                outcomes, agree = [], []
                for e in range(N_EPISODES):
                    base = 9_000_000 + 100_000 * li + 30_000 * ai + 1_000 * arm_i + e
                    oc, ag = run_episode_agree(env, start, pol, refp, base)
                    outcomes.append(oc)
                    agree.extend(ag)
                deaths = outcomes.count("lava")
                cell[aname] = {
                    "n": N_EPISODES, "deaths": deaths,
                    "goals": outcomes.count("goal"),
                    "timeouts": outcomes.count("timeout"),
                    "death_rate": deaths / N_EPISODES,
                    "goal_rate": outcomes.count("goal") / N_EPISODES,
                    "agreement": float(np.mean(agree)) if agree else 1.0,
                }
            # VR-rank diagnostic (S1 abs-preserved only).
            cell["_vr_rank"] = {f"eta{eta}": vr_rank_diag(env, probes, alpha, eta)
                                for eta in ETAS}
            results[lname][f"alpha{alpha}"] = cell
            print(f"  [{time.time()-t0:6.1f}s] {lname:12s} alpha={alpha} "
                  f"done ({len(arms)} arms)")

    _report(results)
    out = _HERE / "results" / "p13_proxy.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "params": {"N": N_WALKERS, "M": HORIZON_M, "beta": BETA, "alphas": ALPHAS,
                   "n_episodes": N_EPISODES, "etas": ETAS,
                   "distill_budgets": DISTILL_BUDGETS,
                   "agree_min": AGREE_MIN, "death_margin": DEATH_MARGIN},
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}   ({time.time()-t0:.1f}s total)")


def _passes_r2(arm, full, n_layouts):
    """Pre-registered R2 pass (P13_DESIGN 6.5), pooled across layouts."""
    death_ok = arm["death_rate"] <= full["death_rate"] + DEATH_MARGIN
    agree_ok = arm["agreement"] >= AGREE_MIN
    return death_ok and agree_ok


def _report(results):
    print("\n" + "=" * 86)
    print("P13 PROXY RESULTS — does sparse interrogation degrade FMC search? (risk R2)")
    print("=" * 86)
    layouts = list(results.keys())
    arm_names = [a for a in next(iter(results[layouts[0]].values())).keys()
                 if not a.startswith("_")]
    verdict = {}
    for alpha in ALPHAS:
        ak = f"alpha{alpha}"
        print(f"\n--- alpha = {alpha} "
              f"(pooled over {len(layouts)} layouts, n={N_EPISODES}/layout) ---")
        print(f"{'arm':16s} {'death%':>8s} {'goal%':>8s} {'agree':>7s} "
              f"{'Wilson95 death':>18s}  R2")
        pooled = {}
        for aname in arm_names:
            d = sum(results[L][ak][aname]["deaths"] for L in layouts)
            g = sum(results[L][ak][aname]["goals"] for L in layouts)
            n = sum(results[L][ak][aname]["n"] for L in layouts)
            ag = float(np.mean([results[L][ak][aname]["agreement"] for L in layouts]))
            pooled[aname] = {"death_rate": d / n, "goal_rate": g / n,
                             "agreement": ag, "deaths": d, "n": n}
        full = pooled["FULL"]
        for aname in arm_names:
            p = pooled[aname]
            lo, hi = wilson_ci(p["deaths"], p["n"])
            if aname == "FULL":
                tag = "ctrl"
            else:
                tag = "PASS" if _passes_r2(p, full, len(layouts)) else "FAIL"
                verdict.setdefault(ak, {})[aname] = tag
            print(f"{aname:16s} {p['death_rate']*100:7.1f}% {p['goal_rate']*100:7.1f}% "
                  f"{p['agreement']:7.2f}  [{lo*100:5.1f}%, {hi*100:5.1f}%]  {tag}")

    # --- VR-rank diagnostic (hP13-0) -----------------------------------------
    print("\n--- VR-rank diagnostic (hP13-0): Spearman of S1(abs-preserved) VR vs FULL ---")
    print(f"{'layout':12s} {'alpha':>6s} " +
          " ".join(f"eta{e}".rjust(11) for e in ETAS))
    for L in layouts:
        for alpha in ALPHAS:
            vr = results[L][f"alpha{alpha}"]["_vr_rank"]
            cells = []
            for e in ETAS:
                d = vr.get(f"eta{e}")
                cells.append("n/a".rjust(11) if not d
                             else f"{d['mean']:.2f}/{d['worst_tick']:.2f}".rjust(11))
            print(f"{L:12s} {alpha:6.1f} " + " ".join(cells)
                  + ("   (mean/worst)" if L == layouts[0] and alpha == ALPHAS[0] else ""))

    # --- pre-registered verdict (P13_DESIGN section 6.5) ---------------------
    print("\n" + "=" * 86)
    print("P13 VERDICT (pre-registered, P13_DESIGN section 6.5)")
    print("=" * 86)
    def any_pass(prefix):
        return {a: all(verdict[f"alpha{al}"].get(a) == "PASS" for al in ALPHAS)
                for a in arm_names if a.startswith(prefix)}
    s1_pre = any_pass("S1_") ; s1_pre = {k: v for k, v in s1_pre.items() if "_abs" in k}
    s2_pass = any_pass("S2_")
    s3_pass = any_pass("S3_")
    go_full = any(s1_pre.values())
    go_test = any(s2_pass.values())
    print(f"  S1 abs-preserved arms passing R2 on all alpha : "
          f"{[k for k,v in s1_pre.items() if v] or 'none'}")
    print(f"  S2 distillation arms passing R2 on all alpha  : "
          f"{[k for k,v in s2_pass.items() if v] or 'none'}")
    print(f"  S3 macro-menu arms passing R2 on all alpha    : "
          f"{[k for k,v in s3_pass.items() if v] or 'none'}")
    if go_full:
        print("  => GO-full : an S1 schema preserves the search on open-domain terms.")
    elif go_test:
        print("  => GO-test : S2 enough for E1-LLM as a closed-domain test (not P13-proper).")
    else:
        print("  => NO-GO : no schema passes R2 — sparsity degrades FMC search.")
    print("  (See P13_DESIGN section 6.5; full verdict written up in P13_RESULT.md.)")


if __name__ == "__main__":
    main()
