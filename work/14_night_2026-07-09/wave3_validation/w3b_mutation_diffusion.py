#!/usr/bin/env python3
"""
W3B (Weakness B) -- The missing Theorem 2'.5: the non-degenerate stationary law
of the FMC cloning kernel WITH mutation, via a diffusion approximation whose
drift/diffusion coefficients are derived from the TRUE (uphill-only clip)
acceptance -- not from standard Moran selection.

------------------------------------------------------------------------------
DERIVATION (two types A,B; frequency x = #A / N; symmetric mutation rate mu;
per-tick stochastic fitness: each walker draws g ~ N(m_type, sigma_v^2),
VR = exp(g), with m_A = delta (mean log-advantage of A), m_B = 0).

Clip-acceptance mean over a fitness LOG-difference of mean m:
    Phi(m) = E_{u ~ N(m, 2*sigma_v^2)} [ clip(e^u - 1, 0, 1) ].

One synchronous tick, leading order in 1/N:
  P(B->A per walker) = x   * Phi(+delta)
  P(A->B per walker) = (1-x)* Phi(-delta)
  => E[dx]_sel = x(1-x)[Phi(delta)-Phi(-delta)] =: s_eff * x(1-x).   (DRIFT)
     s_eff is RENORMALIZED by the uphill-only clip + noise sigma_v; it is the
     corrected drift coefficient, NOT Moran's s.

  Var[dj] ~ 2 N x(1-x) phi0  (phi0=Phi(0), independent-flip leading order)
  => V(x) = x(1-x)/N_e,   N_e = N/(2 phi0).                          (DIFFUSION)

Mutation: drift mu(1-2x), negligible variance.
Effective WF SDE (per generation):
     dx = [ s_eff x(1-x) + mu(1-2x) ] dt + sqrt( x(1-x)/N_e ) dW.

Fokker-Planck stationary density (Wright 1931):
     phi_inf(x) ∝ x^{theta-1}(1-x)^{theta-1} e^{sigma x},
     theta = 2 N_e mu,   sigma = 2 N_e s_eff.
Limits: sigma_v->0 => phi0->0 => N_e->inf, sigma->inf => point mass x=1
        (fixation, Thm 2'.3); delta->0 => sigma=0 => Beta(theta,theta).
------------------------------------------------------------------------------
Verification strategy (honest separation of shape vs coefficient):
  (V0) DRIFT check: init x=0.5, mu=0, one tick, measure E[dx] vs s_eff*0.25.
  (V1) DIFFUSION calibration: neutral heterozygosity decay -> lambda ->
       N_e(measured); compare to analytic N_e=N/(2phi0).
  (V2) NEUTRAL density vs Beta(theta,theta): a-priori (analytic phi0) AND
       calibrated (measured N_e). Isolates shape-correctness from the O(1/N)
       coefficient error.
  (V3) SELECTION density vs Wright: TV, mean match, best-fit (theta,sigma).
  (V4) VALIDITY: TV vs N at FIXED theta,sigma (scale mu,delta ~ 1/N).
"""

import numpy as np
from scipy.optimize import minimize

RNG = np.random.default_rng(20260709)


def a_fmc(r):
    return np.clip(r - 1.0, 0.0, 1.0)


def Phi(m, sigma_v, n=6_000_000, rng=None):
    rng = rng or RNG
    u = rng.normal(m, np.sqrt(2.0) * sigma_v, size=n)
    return float(np.mean(a_fmc(np.exp(u))))


# --------------------------------------------------------------------------
def one_tick(typ, delta, sigma_v, mu, rng):
    chains, N = typ.shape
    ar = np.arange(N)
    g = delta * typ + rng.normal(0.0, sigma_v, size=(chains, N))
    vr = np.exp(g)
    offset = rng.integers(1, N, size=(chains, N))
    idx = (ar[None, :] + offset) % N
    r = np.take_along_axis(vr, idx, axis=1) / vr
    accept = rng.random((chains, N)) < a_fmc(r)
    typ = np.where(accept, np.take_along_axis(typ, idx, axis=1), typ)
    if mu > 0:
        mut = rng.random((chains, N)) < mu
        typ = np.where(mut, 1 - typ, typ)
    return typ.astype(np.int8)


def simulate_stationary(N, mu, delta, sigma_v, chains=220, burn=2000,
                        sample=6000, rng=None):
    rng = rng or RNG
    typ = (rng.random((chains, N)) < 0.5).astype(np.int8)
    out = np.empty((chains, sample))
    for t in range(burn + sample):
        typ = one_tick(typ, delta, sigma_v, mu, rng)
        if t >= burn:
            out[:, t - burn] = typ.mean(axis=1)
    return out


def measure_drift(N, delta, sigma_v, chains=40000, rng=None):
    """E[dx] over one tick starting exactly at x=0.5, no mutation."""
    rng = rng or RNG
    typ = np.zeros((chains, N), dtype=np.int8)
    typ[:, : N // 2] = 1
    x0 = typ.mean(axis=1)
    typ2 = one_tick(typ, delta, sigma_v, 0.0, rng)
    return float((typ2.mean(axis=1) - x0).mean())


def measure_lambda(N, sigma_v, ticks=40, runs=3000, rng=None):
    """Neutral heterozygosity decay rate lambda (delta=0)."""
    rng = rng or RNG
    typ = np.zeros((runs, N), dtype=np.int8)
    typ[:, : N // 2] = 1
    p = typ.mean(axis=1)
    H = np.empty(ticks + 1)
    H[0] = np.mean(2 * p * (1 - p))
    for t in range(1, ticks + 1):
        typ = one_tick(typ, 0.0, sigma_v, 0.0, rng)
        p = typ.mean(axis=1)
        H[t] = np.mean(2 * p * (1 - p))
    tt = np.arange(ticks + 1)
    use = H > H[0] * 0.02
    slope = np.polyfit(tt[use], np.log(H[use] / H[0]), 1)[0]
    return 1.0 - np.exp(slope)


# --------------------------------------------------------------------------
def wright_pdf(xg, theta, sigma):
    log = (theta - 1) * np.log(xg) + (theta - 1) * np.log1p(-xg) + sigma * xg
    log -= log.max()
    d = np.exp(log)
    return d / np.trapezoid(d, xg)


def tv_vs_pred(samples, theta, sigma, nbins=40):
    edges = np.linspace(0, 1, nbins + 1)
    xg = np.linspace(1e-4, 1 - 1e-4, 8000)
    dens = wright_pdf(xg, theta, sigma)
    pred = np.array([np.trapezoid(dens[(xg >= edges[i]) & (xg <= edges[i + 1])],
                                  xg[(xg >= edges[i]) & (xg <= edges[i + 1])])
                     for i in range(nbins)])
    pred /= pred.sum()
    emp, _ = np.histogram(samples.ravel(), bins=edges)
    emp = emp / emp.sum()
    tv = 0.5 * np.abs(emp - pred).sum()
    pred_mean = np.trapezoid(xg * dens, xg)
    return tv, emp.max() and float(np.abs(emp - pred).max()), pred_mean


def best_fit(samples, theta0, sigma0):
    def obj(p):
        th, sg = p
        if th <= 0.05 or sg < 0 or th > 20 or sg > 500:
            return 1e6
        return tv_vs_pred(samples, th, sg)[0]
    res = minimize(obj, [theta0, max(sigma0, 0.0)], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
    return res.x, res.fun


# --------------------------------------------------------------------------
if __name__ == "__main__":
    sv = 0.5
    print("=" * 76)
    print("W3B (Weakness B) -- Theorem 2'.5: stationary law of FMC + mutation")
    print("=" * 76)

    phi0 = Phi(0.0, sv)
    print(f"\n[coeff] sigma_v={sv}")
    print(f"  phi0 = Phi(0) = {phi0:.4f}   analytic N_e/N = 1/(2 phi0) = {1/(2*phi0):.3f}")

    # ---- V0: drift coefficient s_eff derived from acceptance -------------
    print("\n" + "-" * 76)
    print("V0 -- DRIFT check: measured E[dx] at x=0.5 vs  s_eff*0.25")
    print("-" * 76)
    for delta in [0.05, 0.10]:
        s_eff = Phi(delta, sv) - Phi(-delta, sv)
        meas = measure_drift(200, delta, sv)
        print(f"  delta={delta}:  s_eff=Phi(d)-Phi(-d)={s_eff:.5f}"
              f"   pred dx={s_eff*0.25:.5f}   measured dx={meas:.5f}"
              f"   rel.err={abs(meas-s_eff*0.25)/(s_eff*0.25):.2%}")

    # ---- V1: diffusion coefficient calibration ---------------------------
    print("\n" + "-" * 76)
    print("V1 -- DIFFUSION calibration: neutral lambda -> N_e(measured)")
    print("-" * 76)
    lamN = {}
    for N in [100, 200, 400]:
        lam = measure_lambda(N, sv)
        lamN[N] = lam * N
        print(f"  N={N:>4}:  lambda={lam:.5f}  lambda*N={lam*N:.4f}"
              f"   analytic 2*phi0={2*phi0:.4f}"
              f"   ratio meas/analytic={lam*N/(2*phi0):.3f}")
    phi0_eff = np.mean(list(lamN.values())) / 2.0
    print(f"  => phi0_eff (from lambda*N/2) = {phi0_eff:.4f}   "
          f"(analytic phi0={phi0:.4f}, ~{(phi0_eff/phi0-1):.0%} correction from"
          f" pairwise-resampling correlation)")

    # ---- V2: NEUTRAL density -- a-priori vs calibrated --------------------
    print("\n" + "-" * 76)
    print("V2 -- NEUTRAL density (delta=0) vs Beta(theta,theta)")
    print("-" * 76)
    for N in [200, 400]:
        theta_target = 1.5
        mu = theta_target * phi0_eff / N          # target theta wrt TRUE N_e
        theta_apriori = N * mu / phi0             # uses analytic phi0
        theta_calib = N * mu / phi0_eff           # uses measured phi0_eff
        samp = simulate_stationary(N, mu, 0.0, sv)
        tv_a, md_a, _ = tv_vs_pred(samp, theta_apriori, 0.0)
        tv_c, md_c, pm = tv_vs_pred(samp, theta_calib, 0.0)
        (thf, sgf), tvf = best_fit(samp, theta_calib, 0.0)
        print(f"  N={N:>4} mu={mu:.5f}:  emp mean={samp.mean():.4f}")
        print(f"        a-priori  theta={theta_apriori:.3f}: TV={tv_a:.4f} maxdev={md_a:.4f}")
        print(f"        calibrated theta={theta_calib:.3f}: TV={tv_c:.4f} maxdev={md_c:.4f}")
        print(f"        best-fit   theta={thf:.3f} (sigma={sgf:.2f}): TV={tvf:.4f}")

    # ---- V3: SELECTION density -------------------------------------------
    print("\n" + "-" * 76)
    print("V3 -- SELECTION density (delta>0) vs x^{th-1}(1-x)^{th-1}e^{sig x}")
    print("-" * 76)
    for N, delta in [(200, 0.05), (400, 0.04)]:
        s_eff = Phi(delta, sv) - Phi(-delta, sv)
        theta_target = 1.6
        mu = theta_target * phi0_eff / N
        theta = N * mu / phi0_eff
        sigma = N * s_eff / phi0_eff              # calibrated N_e
        samp = simulate_stationary(N, mu, delta, sv)
        tv, md, pm = tv_vs_pred(samp, theta, sigma)
        (thf, sgf), tvf = best_fit(samp, theta, sigma)
        print(f"  N={N:>4} delta={delta} s_eff={s_eff:.5f}: theta={theta:.3f} sigma={sigma:.3f}")
        print(f"        TV={tv:.4f} maxdev={md:.4f}  mean emp={samp.mean():.4f} pred={pm:.4f}")
        print(f"        best-fit theta={thf:.3f} sigma={sgf:.3f} (TV={tvf:.4f})"
              f"  vs predicted ({theta:.3f},{sigma:.3f})")

    # ---- V4: validity -- TV vs N at FIXED theta,sigma --------------------
    print("\n" + "-" * 76)
    print("V4 -- validity: TV vs N at FIXED theta=1.6, sigma=6 (mu,delta ~ 1/N)")
    print("-" * 76)
    # choose delta per N so that sigma = N s_eff/phi0_eff = 6  (weak selection)
    target_sigma = 6.0
    for N in [100, 200, 400, 800]:
        # solve s_eff = target_sigma * phi0_eff / N, then invert Phi to get delta
        s_target = target_sigma * phi0_eff / N
        # bisection on delta: s_eff(delta)=Phi(delta)-Phi(-delta)
        lo, hi = 0.0, 0.6
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if Phi(mid, sv, n=2_000_000) - Phi(-mid, sv, n=2_000_000) < s_target:
                lo = mid
            else:
                hi = mid
        delta = 0.5 * (lo + hi)
        s_eff = Phi(delta, sv) - Phi(-delta, sv)
        mu = 1.6 * phi0_eff / N
        theta = N * mu / phi0_eff
        sigma = N * s_eff / phi0_eff
        ch = 220 if N <= 400 else 160
        samp = simulate_stationary(N, mu, delta, sv, chains=ch,
                                   burn=2000, sample=6000)
        tv, md, pm = tv_vs_pred(samp, theta, sigma)
        print(f"  N={N:>4}: delta={delta:.4f} theta={theta:.3f} sigma={sigma:.3f}"
              f"   TV={tv:.4f} maxdev={md:.4f}")

    # ---- V5: frozen-limit sanity -----------------------------------------
    print("\n" + "-" * 76)
    print("V5 -- frozen limit: sigma_v -> 0 => phi0 -> 0 => fixation")
    print("-" * 76)
    for s in [0.5, 0.2, 0.05]:
        p0 = Phi(0.0, s, n=3_000_000)
        se = Phi(0.08, s, n=3_000_000) - Phi(-0.08, s, n=3_000_000)
        print(f"  sigma_v={s:<5}: phi0={p0:.5f}  N_e/N=1/(2phi0)={1/(2*p0):7.2f}"
              f"   s_eff(.08)={se:.5f}   => sigma=N s_eff/phi0 grows -> fixation")

    print("\n" + "=" * 76)
    print("DONE. All numbers produced by this run (seed 20260709).")
    print("=" * 76)
