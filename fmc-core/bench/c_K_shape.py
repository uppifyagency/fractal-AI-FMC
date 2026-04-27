"""Shape of c_K — how Sergio's b_eff* scales with action arity K.

Run navigation2D parameterized by K at the Sergio config (alpha=0.1, beta=0)
for a sweep of K values, fit a simple model, write summary.

If c_K = b_eff*/K is constant -> "ratio universal".
If c_K = a*K^(-p) -> power law.
If c_K is something else -> document and propose follow-up.

Usage:
    python -m bench.c_K_shape
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fmc.core import virtual_reward, clone_step, effective_branching_factor
from fmc.envs.navigation2d_kN import Navigation2DKN
from bench.runner import run_cell, write_jsonl


def _measure_b_eff(seed, alpha, beta, N, M, K):
    rng = np.random.default_rng(seed)
    env = Navigation2DKN(K=K)
    x0 = env.reset()
    actions = list(env.actions())

    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array(
        [actions[rng.integers(0, len(actions))] for _ in range(N)],
        dtype=np.int64,
    )

    for t in range(M):
        for i in range(N):
            a = labels[i] if t == 0 else env.sample_action(states[i], rng)
            states[i] = env.step(states[i], int(a))
        rewards = np.array([env.reward(s) for s in states], dtype=np.float64)
        obs = np.stack([np.asarray(env.observe(s), dtype=np.float64).ravel() for s in states])
        partners = rng.permutation(N)
        for i in range(N):
            if partners[i] == i:
                partners[i] = (i + 1) % N
        vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
        idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in idx]
        labels = labels[idx]

    return effective_branching_factor(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[3, 4, 6, 9, 12, 16, 24, 32],
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results" / "c_K_shape.jsonl"),
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    N, M = 32, 15
    alpha, beta = 0.1, 0.0

    print(f"Running c_K shape sweep at Sergio config (α={alpha}, β={beta})")
    print(f"K values: {args.ks}, N={N}, M={M}, seeds={args.seeds}\n")

    results = []
    Ks_arr, b_arr, c_arr = [], [], []
    for K in args.ks:
        params = {"alpha": alpha, "beta": beta, "N": N, "M": M, "K": K}
        result = run_cell(
            benchmark="c_K_shape",
            env_name=f"navigation2d_K{K}",
            params=params,
            metric="b_eff",
            sample_fn=lambda s, k=K: _measure_b_eff(s, alpha, beta, N, M, k),
            seeds=seeds,
            notes=f"K={K} at Sergio config; measures shape of c_K = b_eff/K.",
        )
        results.append(result)
        c_K = result.mean / K
        Ks_arr.append(K)
        b_arr.append(result.mean)
        c_arr.append(c_K)
        print(
            f"K={K:>2d} → b_eff = {result.mean:.2f} "
            f"[{result.ci95_low:.2f}, {result.ci95_high:.2f}]   "
            f"c_K = b_eff/K = {c_K:.3f}"
        )

    write_jsonl(results, args.output)

    # Fit candidate models.
    Ks = np.array(Ks_arr, dtype=float)
    bs = np.array(b_arr)

    # Model 1: b_eff = constant (Sergio's claim) -> minimize SSE.
    const = bs.mean()
    sse_const = ((bs - const) ** 2).sum()

    # Model 2: b_eff = a * K^p -> log-log linear regression.
    log_K = np.log(Ks)
    log_b = np.log(bs)
    p, loga = np.polyfit(log_K, log_b, 1)
    a = np.exp(loga)
    pred = a * Ks ** p
    sse_power = ((bs - pred) ** 2).sum()

    # Model 3: b_eff = K * c with c constant -> ratio test.
    c_const = (bs / Ks).mean()
    sse_linK = ((bs - c_const * Ks) ** 2).sum()

    print("\n--- Model fits ---")
    print(f"Constant model (Sergio's '6'):      b_eff = {const:.2f}                  SSE = {sse_const:.2f}")
    print(f"Power law model (b_eff = a K^p):    b_eff = {a:.2f} K^{p:.3f}          SSE = {sse_power:.2f}")
    print(f"Linear in K (b_eff = c K):          b_eff = {c_const:.3f} K              SSE = {sse_linK:.2f}")

    # Save model summary.
    summary = {
        "models": {
            "constant": {"b_eff": float(const), "sse": float(sse_const)},
            "power_law": {"a": float(a), "p": float(p), "sse": float(sse_power)},
            "linear_in_K": {"c": float(c_const), "sse": float(sse_linK)},
        },
        "K_grid": Ks_arr,
        "b_eff_observed": [float(x) for x in b_arr],
        "c_K_observed": [float(x) for x in c_arr],
    }
    summary_path = Path(args.output).parent / "c_K_shape_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
