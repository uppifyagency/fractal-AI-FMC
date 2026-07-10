#!/usr/bin/env python3
"""
W3B (Weakness A) -- Multi-seed robustness of the Theorem 2' numerics under
PER-TICK STOCHASTIC fitness (the honest FMC model: VR resampled every tick
around a per-type mean, not frozen).

Closes the reviewer objection that W31's numbers (q, p) used frozen per-type
VR (Part A) or a single seed (Parts B/C), whereas real FMC has fitness that
fluctuates every tick.

Three deliverables, every number produced here (no invented values):

  (1) q  = heterozygosity-decay exponent  lambda(N) ~ N^q   (WF predicts -1)
      under per-tick fluctuating fitness, bootstrap CI over >=20 seeds,
      N in {32,64,128,256}, robustness across noise level sigma_v.

  (2) p  = neutral fixation-time exponent  T_fix(N) ~ N^p   (WF/Moran predict +1)
      under per-tick fluctuating fitness, bootstrap CI over >=20 seeds.

  (3) C  = alpha_eff population constant  bar_alpha_eff = C * alpha / sigma_R
      with multi-seed CI, on Gaussian AND uniform reward populations.

Exact FMC acceptance used throughout: a_FMC(r) = clip(r-1, 0, 1).
Fitness model: each walker draws VR = exp(g), g ~ N(m_type, sigma_v^2), every
tick (type-independent mean for the NEUTRAL tests -> genuine drift; the per-type
mean only enters Weakness B).
"""

import numpy as np

# --------------------------------------------------------------------------
def a_fmc(r):
    return np.clip(r - 1.0, 0.0, 1.0)


def relativize(r):
    """Def. 2 (reward channel, beta=0). Population z-score + two-branch map."""
    mu = r.mean()
    sd = r.std()
    if sd == 0.0:
        return np.ones_like(r)
    z = (r - mu) / sd
    out = np.where(z <= 0.0, np.exp(z), 1.0 + np.log1p(np.clip(z, 0.0, None)))
    return out


# --------------------------------------------------------------------------
# (1) Heterozygosity decay under per-tick fluctuating fitness
# --------------------------------------------------------------------------
def lambda_of_N(N, rng, ticks=40, runs=400, sigma_v=0.5):
    """Synchronous exact FMC kernel, tick = generation, 2 types at p=0.5, NO
    systematic selection (both type means equal -> neutral). Each tick every
    walker draws VR=exp(N(0,sigma_v^2)) INDEPENDENTLY (per-tick stochastic).
    Returns lambda from H(t)/H0 = (1-lambda)^t."""
    typ = np.zeros((runs, N), dtype=np.int8)
    typ[:, : N // 2] = 1
    p = typ.mean(axis=1)
    H = np.empty(ticks + 1)
    H[0] = np.mean(2 * p * (1 - p))
    for t in range(1, ticks + 1):
        g = rng.normal(0.0, sigma_v, size=(runs, N))
        vr = np.exp(g)
        offset = rng.integers(1, N, size=(runs, N))
        idx = (np.arange(N)[None, :] + offset) % N
        r = np.take_along_axis(vr, idx, axis=1) / vr
        accept = rng.random((runs, N)) < a_fmc(r)
        typ = np.where(accept, np.take_along_axis(typ, idx, axis=1), typ)
        p = typ.mean(axis=1)
        H[t] = np.mean(2 * p * (1 - p))
    t = np.arange(ticks + 1)
    use = H > H[0] * 0.02
    slope = np.polyfit(t[use], np.log(H[use] / H[0]), 1)[0]
    return 1.0 - np.exp(slope)


def q_one_seed(seed, Ns, sigma_v=0.5, runs=400):
    rng = np.random.default_rng(seed)
    lams = np.array([lambda_of_N(N, rng, runs=runs, sigma_v=sigma_v) for N in Ns])
    q = np.polyfit(np.log(Ns), np.log(lams), 1)[0]
    return q, lams


# --------------------------------------------------------------------------
# (2) Neutral fixation time under per-tick fluctuating fitness
# --------------------------------------------------------------------------
def tfix_of_N(N, rng, runs=150, sigma_v=0.5, max_ticks=None):
    if max_ticks is None:
        max_ticks = int(40 * N)
    typ = np.zeros((runs, N), dtype=np.int8)
    typ[:, : N // 2] = 1
    tfix = np.full(runs, -1, dtype=np.int64)
    active = np.ones(runs, dtype=bool)
    for t in range(1, max_ticks + 1):
        na = int(active.sum())
        if na == 0:
            break
        vr = np.exp(rng.normal(0.0, sigma_v, size=(na, N)))
        offset = rng.integers(1, N, size=(na, N))
        idx = (np.arange(N)[None, :] + offset) % N
        r = np.take_along_axis(vr, idx, axis=1) / vr
        accept = rng.random((na, N)) < a_fmc(r)
        sub = typ[active]
        sub = np.where(accept, np.take_along_axis(sub, idx, axis=1), sub)
        typ[active] = sub
        p = sub.mean(axis=1)
        done = (p == 0.0) | (p == 1.0)
        act_idx = np.where(active)[0]
        tfix[act_idx[done]] = t
        active[act_idx[done]] = False
    return tfix[tfix > 0].mean()


def p_one_seed(seed, Ns, sigma_v=0.5, runs=150):
    rng = np.random.default_rng(seed)
    ts = np.array([tfix_of_N(N, rng, runs=runs, sigma_v=sigma_v) for N in Ns])
    p = np.polyfit(np.log(Ns), np.log(ts), 1)[0]
    return p, ts


# --------------------------------------------------------------------------
# (3) alpha_eff constant C, multi-seed, Gaussian + uniform
# --------------------------------------------------------------------------
def C_one_seed(seed, dist, alpha=1.0, sigma_R=1.0, npop=200_000):
    rng = np.random.default_rng(seed)
    if dist == "gauss":
        R = rng.normal(0.0, sigma_R, size=npop)
    elif dist == "unif":
        half = np.sqrt(3.0) * sigma_R          # Var(U[-h,h]) = h^2/3
        R = rng.uniform(-half, half, size=npop)
    else:
        raise ValueError(dist)
    logVR = alpha * np.log(relativize(R))
    sR = R.std()
    beta_hat = np.cov(logVR, R, bias=True)[0, 1] / R.var()
    return beta_hat * sR / alpha             # C_hat


# --------------------------------------------------------------------------
def bootstrap_ci(x, B=20000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return x.mean(), lo, hi, x.std(ddof=1)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    Ns = [32, 64, 128, 256]
    N_SEEDS = 25
    seeds = 20260709 + np.arange(N_SEEDS)

    print("=" * 74)
    print("W3B (Weakness A) -- multi-seed robustness under per-tick stochastic")
    print(f"                    fitness.  {N_SEEDS} seeds, N in {Ns}")
    print("=" * 74)

    # ---- (1) q with CI, primary sigma_v=0.5 --------------------------------
    print("\n(1) HETEROZYGOSITY EXPONENT q   lambda(N) ~ N^q   [WF: -1]")
    print("    per-tick fluctuating VR = exp(N(0, sigma_v^2)),  sigma_v=0.5")
    qs = []
    lam_stack = []
    for sd in seeds:
        q, lams = q_one_seed(int(sd), Ns, sigma_v=0.5)
        qs.append(q)
        lam_stack.append(lams)
    qs = np.array(qs)
    lam_mean = np.mean(lam_stack, axis=0)
    m, lo, hi, sdv = bootstrap_ci(qs)
    print("    mean lambda(N) across seeds:")
    for N, lm in zip(Ns, lam_mean):
        print(f"        N={N:>4}:  lambda={lm:.5f}   lambda*N={lm*N:.3f}")
    print(f"    q = {m:.4f}   95% bootstrap CI [{lo:.4f}, {hi:.4f}]   sd={sdv:.4f}")

    # ---- robustness of q across noise levels ------------------------------
    print("\n    robustness of q across noise level sigma_v (10 seeds each):")
    for sv in [0.25, 0.5, 1.0]:
        qq = np.array([q_one_seed(int(sd), Ns, sigma_v=sv)[0]
                       for sd in seeds[:10]])
        mm, l2, h2, _ = bootstrap_ci(qq)
        print(f"        sigma_v={sv:<4}:  q = {mm:.4f}   CI [{l2:.4f}, {h2:.4f}]")

    # ---- (2) p with CI ----------------------------------------------------
    print("\n(2) FIXATION-TIME EXPONENT p   T_fix(N) ~ N^p   [WF/Moran: +1]")
    print("    per-tick fluctuating VR, sigma_v=0.5, neutral")
    ps = []
    t_stack = []
    for sd in seeds:
        p, ts = p_one_seed(int(sd), Ns, sigma_v=0.5, runs=150)
        ps.append(p)
        t_stack.append(ts)
    ps = np.array(ps)
    t_mean = np.mean(t_stack, axis=0)
    m, lo, hi, sdv = bootstrap_ci(ps)
    print("    mean T_fix(N) across seeds:")
    for N, tt in zip(Ns, t_mean):
        print(f"        N={N:>4}:  T_fix={tt:8.1f} ticks   T_fix/N={tt/N:.3f}")
    print(f"    p = {m:.4f}   95% bootstrap CI [{lo:.4f}, {hi:.4f}]   sd={sdv:.4f}")

    # ---- (3) C with CI, Gaussian + uniform --------------------------------
    print("\n(3) alpha_eff CONSTANT C   bar_alpha_eff = C * alpha / sigma_R")
    print("    (multi-seed, npop=2e5 per seed)")
    for dist, ref in [("gauss", 0.7223), ("unif", 0.7383)]:
        Cs = np.array([C_one_seed(int(sd), dist) for sd in seeds])
        m, lo, hi, sdv = bootstrap_ci(Cs)
        # also stress the scaling law: vary alpha, sigma_R -> C must be invariant
        grid = []
        rng_g = np.random.default_rng(999)
        for al in [0.5, 1.0, 2.0]:
            for sR in [0.2, 1.0, 5.0]:
                grid.append(C_one_seed(int(rng_g.integers(1e9)), dist,
                                       alpha=al, sigma_R=sR))
        grid = np.array(grid)
        print(f"    {dist:>6}:  C = {m:.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
              f"   sd={sdv:.4f}   (W32 ref {ref})")
        print(f"            scaling-law invariance across alpha x sigma_R grid:"
              f"  C in [{grid.min():.4f}, {grid.max():.4f}]"
              f"  (spread {grid.max()-grid.min():.4f})")

    print("\n" + "=" * 74)
    print("DONE. All numbers above are produced by this run (seeded).")
    print("=" * 74)
