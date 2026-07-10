#!/usr/bin/env python3
"""
W6C -- Closes Adversarial DEFECT 2: is sigma_v (the per-tick log-fitness noise
inside Phi) DETERMINED by relativize, or a free parameter?

Coupled relativize+clone simulation (nothing is a free input except the reward
model). Each tick:
  1. redraw reward R_i ~ N(m_type_i, sigma_within^2),  m_A=mu+dR/2, m_B=mu-dR/2
  2. relativize over the POOLED population: z=(R-mean)/std, logVR = alpha*log Rhat(z)
  3. pairwise clone: i adopts partner j's type with prob clip(VR_j/VR_i-1,0,1)
Then MEASURE the realised type-A drift at x=0.5:  s_eff_meas = 4 * E[dx].

The "sigma_v" is NOT an input: it is READ OFF the population as the within-type
std of logVR (s_A, s_B). The clip acceptance for a mixed A-B pair averages a
log-VR DIFFERENCE ~ N(delta, s_A^2+s_B^2), so the honest identification is
   tau^2 = s_A^2 + s_B^2   (= 2 sigma_v^2 only if s_A = s_B).

TESTS:
 (T1) s_A vs s_B: is the within-type spread a single constant? (weak vs strong dR)
 (T2) LINK A on the REAL population: delta_measured vs alpha_eff * dR
 (T3) IDENTIFICATION: s_eff_meas (from clone dynamics) vs Phi(delta;tau^2)-Phi(-delta;tau^2)
      with tau^2 PINNED from the population -- this is what DEFECT 2 demanded.
 (T4) BRIDGE: s_eff_meas vs 2 Phi'(0;tau^2) * alpha_eff * dR (weak selection).
"""

import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng(20260709)
LN2 = np.log(2.0)


def a_fmc(r):
    return np.clip(r - 1.0, 0.0, 1.0)


def log_Rhat(z):
    zpos = np.maximum(z, 0.0)
    return np.where(z <= 0, z, np.log1p(np.log1p(zpos)))


def Phi_tau(m, tau):
    """E_{u~N(m, tau^2)}[clip(e^u-1,0,1)], closed form (F = normal CDF, std tau)."""
    band = np.exp(m + 0.5 * tau**2) * (
        norm.cdf(LN2, m + tau**2, tau) - norm.cdf(0.0, m + tau**2, tau))
    return band + 1.0 - 2.0 * norm.cdf(LN2, m, tau) + norm.cdf(0.0, m, tau)


def Phi_prime0_tau(tau):
    """d/dm Phi_tau(m)|_0 = INT_0^ln2 e^u N(u;0,tau^2) du."""
    return np.exp(0.5 * tau**2) * (norm.cdf(LN2, tau**2, tau) - norm.cdf(0.0, tau**2, tau))


def coupled_tick_drift(N, alpha, dR, sigma_within, chains=60000, rng=None):
    """One coupled relativize+clone tick starting at x=0.5; return (dx, diagnostics)."""
    rng = rng or RNG
    typ = np.zeros((chains, N), dtype=np.int8)
    typ[:, : N // 2] = 1                                   # first half = type A
    # per-tick rewards
    m = np.where(typ == 1, dR / 2.0, -dR / 2.0)            # mu=0 wlog
    R = m + rng.normal(0.0, sigma_within, size=(chains, N))
    # relativize (pooled per chain)
    mu = R.mean(axis=1, keepdims=True)
    sd = R.std(axis=1, keepdims=True)
    z = (R - mu) / sd
    logvr = alpha * log_Rhat(z)
    vr = np.exp(logvr)
    # pairwise clone
    ar = np.arange(N)
    offset = rng.integers(1, N, size=(chains, N))
    idx = (ar[None, :] + offset) % N
    r = np.take_along_axis(vr, idx, axis=1) / vr
    accept = rng.random((chains, N)) < a_fmc(r)
    typ2 = np.where(accept, np.take_along_axis(typ, idx, axis=1), typ)
    dx = float((typ2.mean(axis=1) - typ.mean(axis=1)).mean())
    # population diagnostics (pinned sigma_v): within-type logVR std, cross-type gap
    A = typ == 1
    sA = float(logvr[A].std())
    sB = float(logvr[~A].std())
    delta = float(logvr[A].mean() - logvr[~A].mean())
    # alpha_eff = population OLS slope of logVR on R
    aeff = float(np.cov(logvr.ravel(), R.ravel())[0, 1] / R.var())
    return dx, dict(sA=sA, sB=sB, delta=delta, aeff=aeff)


if __name__ == "__main__":
    print("=" * 84)
    print("W6C -- coupled relativize+clone: is sigma_v pinned by relativize? (DEFECT 2)")
    print("=" * 84)
    N, alpha, sigma_within = 400, 1.0, 0.5
    sigma_R_approx = sigma_within                      # pooled std ~ sigma_within for small dR
    print(f"N={N} alpha={alpha} sigma_within={sigma_within}  (mu=0)")
    print(f"  reference alpha_eff = C*alpha/sigma_R ~ 0.7223*{alpha}/{sigma_R_approx} "
          f"= {0.7223*alpha/sigma_R_approx:.4f}\n")

    print(f"{'dR':>6} | {'s_A':>7} {'s_B':>7} {'sA=sB?':>7} | {'delta':>8} {'aeff*dR':>8} "
          f"{'T2err':>6} | {'tau':>6} | {'s_eff_meas':>10} {'Phi(pin)':>9} {'T3err':>6} "
          f"| {'bridge':>8} {'T4err':>6}")
    print("-" * 84)
    for dR in [0.25, 0.15, 0.10, 0.05, 0.025]:
        dx, d = coupled_tick_drift(N, alpha, dR, sigma_within)
        s_eff_meas = 4.0 * dx                          # E[dx] = s_eff * x(1-x) = s_eff/4
        tau = np.sqrt(d["sA"]**2 + d["sB"]**2)         # PINNED from population
        s_eff_pin = Phi_tau(d["delta"], tau) - Phi_tau(-d["delta"], tau)  # T3
        bridge = 2 * Phi_prime0_tau(tau) * d["aeff"] * dR                 # T4
        delta_pred = d["aeff"] * dR
        t2 = abs(d["delta"] - delta_pred) / abs(d["delta"])
        t3 = abs(s_eff_meas - s_eff_pin) / abs(s_eff_meas)
        t4 = abs(s_eff_meas - bridge) / abs(s_eff_meas)
        eq = "yes" if abs(d["sA"] - d["sB"]) / d["sA"] < 0.05 else "NO"
        print(f"{dR:>6} | {d['sA']:>7.4f} {d['sB']:>7.4f} {eq:>7} | "
              f"{d['delta']:>8.5f} {delta_pred:>8.5f} {t2:>5.1%} | {tau:>6.4f} | "
              f"{s_eff_meas:>10.5f} {s_eff_pin:>9.5f} {t3:>5.1%} | {bridge:>8.5f} {t4:>5.1%}")

    print("\nLegend: T2 = LINK A (delta = aeff*dR); T3 = IDENTIFICATION (clone drift vs")
    print("Phi with tau PINNED from population); T4 = weak-selection bridge.")
    print("If T3 is small, sigma_v is DETERMINED by relativize -> unification holds.")
    print("=" * 84)
