#!/usr/bin/env python3
"""
W6B -- The alpha_eff <-> s_eff bridge (closes the paper's Section 7.3 tension).

Two "effective" quantities were derived separately and both are correct:
  Thm 4    : alpha_eff = C * alpha / sigma_R   (inverse temperature felt in log-VR
             space per unit reward; C = E[g(z)]).
  Thm 2'.5 : s_eff = Phi(delta) - Phi(-delta)  (per-tick frequency drift), with
             Phi(m) = E_{u~N(m, 2 sigma_v^2)}[clip(e^u - 1, 0, 1)].

They are NOT two rival temperatures. They are the SAME selection mechanism,
linearised in two different coordinate systems, composed by the chain rule:

    reward gap  --[relativize; slope alpha_eff]-->  log-VR gap delta
    log-VR gap  --[clip drift;  slope 2*Phi'(0)]-->  selection drift s_eff

  LINK A (exact linearisation of relativize):   delta = alpha_eff * dR
  LINK B (weak-selection limit of the clip):    s_eff = Phi(delta)-Phi(-delta)
                                                       ~= 2*Phi'(0) * delta
  COMPOSITION:                                   s_eff ~= 2*Phi'(0) * alpha_eff * dR

CLOSED FORM for the clip's marginal transmission (Stein / integration by parts):
  Phi'(0) = (1/tau^2) E_{u~N(0,tau^2)}[ u * clip(e^u-1,0,1) ]      (tau^2 = 2 sigma_v^2)
          = E_{u~N(0,tau^2)}[ c'(u) ]  ,  c'(u)=e^u on 0<u<ln2 else 0    (Stein)
          = INT_0^{ln 2} e^u * N(u; 0, tau^2) du.
  Only the transition band 0<u<ln2 (accept prob strictly in (0,1)) transmits
  marginal selection: below 0 rejected, above ln2 saturated at 1.

This script verifies: (1) Phi'(0) closed form vs finite difference; (2) LINK B
weak-selection slope; (3) LINK A relativize slope; (4) end-to-end composition.
"""

import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng(20260709)
LN2 = np.log(2.0)


def clip_acc(r):
    return np.clip(r - 1.0, 0.0, 1.0)


def _band_integral(m, tau):
    """INT_0^{ln2} e^u N(u; m, tau^2) du  =  e^{m+tau^2/2} [F(ln2; m+tau^2) - F(0; m+tau^2)]
    (complete-the-square; F = CDF of N(.,tau^2))."""
    return np.exp(m + 0.5 * tau**2) * (
        norm.cdf(LN2, m + tau**2, tau) - norm.cdf(0.0, m + tau**2, tau))


def Phi(m, sigma_v):
    """Phi(m) = E_{u~N(m, 2 sigma_v^2)}[clip(e^u-1,0,1)]  -- FULLY CLOSED FORM.
       = INT_0^{ln2}(e^u-1) N du + INT_{ln2}^inf 1 * N du
       = band_integral - (F(ln2;m)-F(0;m)) + (1 - F(ln2;m))."""
    tau = np.sqrt(2.0) * sigma_v
    band = _band_integral(m, tau)
    return float(band + 1.0 - 2.0 * norm.cdf(LN2, m, tau) + norm.cdf(0.0, m, tau))


def Phi_prime0_closed(sigma_v):
    """Phi'(0) = INT_0^{ln2} e^u N(u;0,tau^2) du = band_integral(0, tau)  (closed form)."""
    tau = np.sqrt(2.0) * sigma_v
    return float(_band_integral(0.0, tau))


def Phi_prime0_stein_mc(sigma_v, n=40_000_000, rng=None):
    """Cross-check via Stein form (1/tau^2) E[u clip(e^u-1)]."""
    rng = rng or RNG
    tau = np.sqrt(2.0) * sigma_v
    u = rng.normal(0.0, tau, size=n)
    return float(np.mean(u * clip_acc(np.exp(u))) / tau**2)


# ---- relativize (canon Def. 2), reward channel only -------------------------
def log_Rhat(z):
    zpos = np.maximum(z, 0.0)                       # guard log1p domain for z<=0 branch
    return np.where(z <= 0, z, np.log1p(np.log1p(zpos)))


def alpha_eff_pop(alpha, sigma_R, rewards):
    """Population inverse temperature: OLS slope of log VR on reward = C*alpha/sigma_R."""
    z = (rewards - rewards.mean()) / rewards.std()
    logvr = alpha * log_Rhat(z)
    # slope of logvr on reward
    return np.cov(logvr, rewards)[0, 1] / np.var(rewards)


if __name__ == "__main__":
    print("=" * 78)
    print("W6B -- alpha_eff <-> s_eff bridge")
    print("=" * 78)

    # ---- (1) Phi'(0) closed form vs finite difference vs Stein-MC -----------
    print("\n(1) Clip marginal transmission Phi'(0)")
    print("-" * 78)
    for sv in [0.25, 0.5, 1.0]:
        cf = Phi_prime0_closed(sv)
        h = 1e-4
        fd = (Phi(h, sv) - Phi(-h, sv)) / (2 * h)
        st = Phi_prime0_stein_mc(sv)
        print(f"  sigma_v={sv}:  closed INT_0^ln2 = {cf:.6f}   "
              f"finite-diff = {fd:.6f} (err {abs(cf-fd)/cf:.2%})   "
              f"Stein-MC = {st:.6f} (err {abs(cf-st)/cf:.2%})")

    # ---- (2) LINK B: s_eff = Phi(delta)-Phi(-delta) ~= 2 Phi'(0) delta ------
    print("\n(2) LINK B: weak-selection slope  s_eff / delta -> 2 Phi'(0)")
    print("-" * 78)
    for sv in [0.5]:
        two_p0 = 2 * Phi_prime0_closed(sv)
        print(f"  sigma_v={sv}: 2*Phi'(0) = {two_p0:.5f}")
        for delta in [0.2, 0.1, 0.05, 0.02, 0.01]:
            s_eff = Phi(delta, sv) - Phi(-delta, sv)
            lin = two_p0 * delta
            print(f"    delta={delta:<5}: s_eff={s_eff:.6f}  2Phi'(0)*delta={lin:.6f}"
                  f"   rel.err={abs(s_eff-lin)/s_eff:.2%}")

    # ---- (3) LINK A: delta = alpha_eff * dR  (relativize linearisation) -----
    print("\n(3) LINK A: relativize turns reward gap dR into log-VR gap delta")
    print("-" * 78)
    alpha, sigma_R = 1.0, 1.0
    base = RNG.normal(0.0, sigma_R, size=4_000_000)   # pooled reward population
    a_eff = alpha_eff_pop(alpha, sigma_R, base)
    print(f"  alpha={alpha} sigma_R={sigma_R}:  alpha_eff (pop OLS) = {a_eff:.5f}"
          f"   (expected C*alpha/sigma_R, C~0.7223 => {0.7223*alpha/sigma_R:.5f})")
    for dR in [0.2, 0.1, 0.05, 0.02]:
        # type A rewards shifted +dR/2, type B -dR/2; relativize over the pool
        rA = base + dR / 2
        rB = base - dR / 2
        pool = np.concatenate([rA, rB])
        z = (pool - pool.mean()) / pool.std()
        logvr = alpha * log_Rhat(z)
        nA = len(rA)
        delta_meas = logvr[:nA].mean() - logvr[nA:].mean()
        delta_pred = a_eff * dR
        print(f"    dR={dR:<5}: delta_measured={delta_meas:.6f}  "
              f"alpha_eff*dR={delta_pred:.6f}   rel.err={abs(delta_meas-delta_pred)/delta_meas:.2%}")

    # ---- (4) END-TO-END: s_eff ~= 2 Phi'(0) alpha_eff dR --------------------
    print("\n(4) COMPOSITION: s_eff ~= 2 Phi'(0) * alpha_eff * dR")
    print("-" * 78)
    sv = 0.5
    two_p0 = 2 * Phi_prime0_closed(sv)
    for dR in [0.1, 0.05, 0.02]:
        # delta produced by relativize from this reward gap:
        rA, rB = base + dR / 2, base - dR / 2
        pool = np.concatenate([rA, rB]); z = (pool - pool.mean()) / pool.std()
        logvr = alpha * log_Rhat(z); nA = len(rA)
        delta = logvr[:nA].mean() - logvr[nA:].mean()
        s_eff_true = Phi(delta, sv) - Phi(-delta, sv)
        s_eff_bridge = two_p0 * a_eff * dR
        print(f"  dR={dR:<5}: delta={delta:.5f}  s_eff(true clip)={s_eff_true:.6f}"
              f"  bridge 2Phi'(0)*alpha_eff*dR={s_eff_bridge:.6f}"
              f"   rel.err={abs(s_eff_true-s_eff_bridge)/s_eff_true:.2%}")

    print("\n" + "=" * 78)
    print("DONE (seed 20260709).")
    print("=" * 78)
