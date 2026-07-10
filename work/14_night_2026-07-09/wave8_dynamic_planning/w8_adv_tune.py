#!/usr/bin/env python3
"""ADVERSARIAL probe (A + B): the decisive test. Try HARD to make FMC beat the
MPC baselines at the deceptive point offset=1.5.

(A) alpha x beta sweep. Study fixes alpha=1,beta=1. Hypothesis: deceptive dense
    reward -d pulls to the wall, so LOW alpha (less reward-following) + HIGH beta
    (more causal-entropy dispersion) should let FMC escape the local optimum.
(B) decode fairness. core.plan uses majority-vote of first actions ("decide").
    MPC returns first action of the single BEST sequence. Test an argmax decode:
    first action of the surviving walker with max final reward.

Faithful replica of fmc.core.plan (same RNG stream) so decode='majority' is
bit-identical to core.plan; only the final read-out changes for 'argmax'.
Paired seeds identical to w8b_budget_sweep so the comparison is apples-to-apples.
"""
import sys, os
import numpy as np
from math import sqrt, erfc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from fmc.core import virtual_reward, clone_step, decide  # noqa: E402
from w8_deceptive_nav import DeceptiveNav, run_episode, plan_random_shooting, plan_cem  # noqa: E402


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, erfc(abs(z) / sqrt(2))


def plan_fmc_variant(env, x0, N, M, rng_seed, alpha, beta, decode="majority"):
    """Faithful replica of fmc.core.plan with configurable alpha,beta,decode."""
    rng = np.random.default_rng(rng_seed)
    actions = list(env.actions())
    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array([actions[rng.integers(0, len(actions))] for _ in range(N)],
                      dtype=object)
    for t in range(M):
        for i in range(N):
            a = labels[i] if t == 0 else env.sample_action(states[i], rng)
            states[i] = env.step(states[i], a)
        rewards = np.array([env.reward(s) for s in states], dtype=np.float64)
        obs = np.stack([np.asarray(env.observe(s), dtype=np.float64).ravel()
                        for s in states])
        partners = rng.permutation(N)
        for i in range(N):
            if partners[i] == i:
                partners[i] = (i + 1) % N
        vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
        clone_idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in clone_idx]
        labels = labels[clone_idx]
    if decode == "majority":
        return decide(labels)
    elif decode == "argmax":
        final_r = np.array([env.reward(s) for s in states], dtype=np.float64)
        return int(labels[int(np.argmax(final_r))])
    raise ValueError(decode)


def make_fmc_planner(alpha, beta, decode):
    def planner(env, s, N, M, rng, **kw):
        seed = int(rng.integers(0, 2**31 - 1))
        return plan_fmc_variant(env, s, N, M, seed, alpha, beta, decode)
    return planner


def eval_planner(planner, N, M, offset, instances, H, seed_base=90210):
    succ = 0
    for inst in range(instances):
        rng = np.random.default_rng(seed_base + inst * 31 + N * 7)
        env = DeceptiveNav(offset=offset, reward_mode="dense")
        succ += int(run_episode(env, planner, N, M, H, rng)["success"])
    return succ


def run(offset=1.5, instances=40, H=70, budget=(36, 11),
        alphas=(0.1, 0.3, 0.5, 1.0, 2.0, 5.0),
        betas=(0.0, 0.5, 1.0, 2.0, 4.0)):
    N, M = budget
    print("=" * 100)
    print(f"ADVERSARIAL TUNE | offset={offset} B={N*M} (N={N},M={M}) n={instances} H={H}")
    print("=" * 100)
    # Baselines on identical paired seeds
    rand_k = eval_planner(plan_random_shooting, N, M, offset, instances, H)
    cem_k = eval_planner(plan_cem, N, M, offset, instances, H)
    base_best_k = max(rand_k, cem_k)
    base_best_name = "rand" if rand_k >= cem_k else "CEM"
    print(f"BASELINES: rand-shoot={rand_k/instances:.3f}  CEM={cem_k/instances:.3f}"
          f"  -> best={base_best_k/instances:.3f} ({base_best_name})")
    print("-" * 100)
    print(f"{'alpha':>6} {'beta':>5} | {'maj succ':>9} {'z_maj':>7} {'p_maj':>7} |"
          f" {'arg succ':>9} {'z_arg':>7} {'p_arg':>7}   flag")
    print("-" * 100)
    best = {"succ": -1, "cfg": None}
    for a in alphas:
        for b in betas:
            km = eval_planner(make_fmc_planner(a, b, "majority"), N, M, offset, instances, H)
            ka = eval_planner(make_fmc_planner(a, b, "argmax"), N, M, offset, instances, H)
            zm, pm = two_prop_z(km, instances, base_best_k, instances)
            za, pa = two_prop_z(ka, instances, base_best_k, instances)
            for kk, cfgd in ((km, (a, b, "maj")), (ka, (a, b, "arg"))):
                if kk > best["succ"]:
                    best = {"succ": kk, "cfg": cfgd}
            flag = ""
            if km >= base_best_k:
                flag += "MAJ>=base "
            if ka >= base_best_k:
                flag += "ARG>=base "
            print(f"{a:>6.1f} {b:>5.1f} | {km/instances:>9.3f} {zm:>+7.2f} {pm:>7.3f} |"
                  f" {ka/instances:>9.3f} {za:>+7.2f} {pa:>7.3f}   {flag}")
    print("-" * 100)
    ba, bb, bd = best["cfg"]
    print(f"BEST FMC: alpha={ba} beta={bb} decode={bd} succ={best['succ']/instances:.3f}"
          f"  vs best baseline {base_best_k/instances:.3f} ({base_best_name})")
    zb, pb = two_prop_z(best["succ"], instances, base_best_k, instances)
    print(f"  best-FMC vs best-baseline: z={zb:+.2f} p={pb:.3f}")
    print("=" * 100)
    return best, base_best_k


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=36)
    ap.add_argument("--M", type=int, default=11)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--offset", type=float, default=1.5)
    args = ap.parse_args()
    run(offset=args.offset, instances=args.n, budget=(args.N, args.M))
