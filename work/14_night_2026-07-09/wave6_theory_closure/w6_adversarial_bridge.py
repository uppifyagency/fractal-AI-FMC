#!/usr/bin/env python3
"""
ADVERSARIAL verification of CLAIM 2 (G2): the alpha_eff<->s_eff bridge.

Independent checks, NOT reusing w6b's closed forms as ground truth:
  (g) brute-force Phi(m) via scipy.quad over a RANGE of m (neg, 0, large pos),
      including the region split at u=0 and u=ln2, vs the closed form.
  (h) Phi'(0) by three independent methods: quad of e^u N over (0,ln2),
      high-order central difference of brute-force Phi, direct MC of c'(u).
  (i) THE LINK-A CLAIM as WRITTEN in the doc (line 91): delta ~= alpha*g(zbar)*dR/sigma_R
      with zbar=0 => g(0). Test whether the correct constant is g(0)=1 or the
      population-averaged slope C=E[g(z)]. Measure delta/dR directly.
  (j) show sigma_v is an independent free parameter of the bridge (change it,
      holding alpha,sigma_R fixed: bridge still 'works' -> not pinned by relativize).
"""
import numpy as np
from scipy import integrate
from scipy.stats import norm

RNG = np.random.default_rng(20260709)
LN2 = np.log(2.0)


def clip_acc(r):
    return np.clip(r - 1.0, 0.0, 1.0)


def clip_c(u):
    return np.clip(np.exp(u) - 1.0, 0.0, 1.0)


# ---- doc's closed form (target under test) ----------------------------------
def Phi_closed(m, sv):
    tau = np.sqrt(2.0) * sv
    band = np.exp(m + 0.5*tau**2)*(norm.cdf(LN2, m+tau**2, tau) -
                                   norm.cdf(0.0, m+tau**2, tau))
    return float(band + 1.0 - 2.0*norm.cdf(LN2, m, tau) + norm.cdf(0.0, m, tau))


# ---- brute-force Phi(m) via quad, explicit region split ---------------------
def Phi_brute(m, sv):
    tau = np.sqrt(2.0)*sv
    f = lambda u: clip_c(u) * norm.pdf(u, m, tau)
    # split at the kinks 0 and ln2 so quad sees smooth pieces
    lo, hi = m - 14*tau, m + 14*tau
    pts = [p for p in (0.0, LN2) if lo < p < hi]
    val = integrate.quad(f, lo, hi, points=pts, limit=200)[0]
    return val


# ---- relativize channel -----------------------------------------------------
def log_Rhat(z):
    zpos = np.maximum(z, 0.0)
    return np.where(z <= 0, z, np.log1p(np.log1p(zpos)))


def g_slope(z):
    """d/dz log_Rhat(z)."""
    return np.where(z <= 0, 1.0, 1.0/((1.0+np.log1p(np.maximum(z,0.0)))*(1.0+np.maximum(z,0.0))))


if __name__ == "__main__":
    print("="*80)
    print("(g) brute-force Phi(m) vs closed form, over a RANGE of m")
    print("="*80)
    for sv in [0.25, 0.5, 1.0]:
        print(f"\nsigma_v={sv}")
        for m in [-1.0, -0.3, -0.05, 0.0, 0.05, 0.3, 1.0, 2.5]:
            c = Phi_closed(m, sv); b = Phi_brute(m, sv)
            print(f"  m={m:+5.2f}: closed={c:.6f}  brute={b:.6f}  "
                  f"abs.err={abs(c-b):.2e}")

    print("\n" + "="*80)
    print("(h) Phi'(0): three independent methods")
    print("="*80)
    for sv in [0.25, 0.5, 1.0]:
        tau = np.sqrt(2.0)*sv
        band = integrate.quad(lambda u: np.exp(u)*norm.pdf(u,0.0,tau), 0.0, LN2)[0]
        # high-order 5-point central diff of brute-force Phi
        h = 1e-3
        vals = [Phi_brute(x*h, sv) for x in (-2,-1,1,2)]
        deriv = (vals[0] - 8*vals[1] + 8*vals[2] - vals[3])/(12*h)  # note sign order
        # fix ordering: f(-2h),f(-h),f(h),f(2h)
        fm2,fm1,fp1,fp2 = vals
        deriv = (fm2 - 8*fm1 + 8*fp1 - fp2)/(12*h)
        u = RNG.normal(0.0, tau, size=20_000_000)
        mc = np.mean(np.where((u>0)&(u<LN2), np.exp(u), 0.0))
        print(f"  sigma_v={sv}: quad(band)={band:.6f}  5ptdiff(brute)={deriv:.6f}  "
              f"MC(c')={mc:.6f}")

    print("\n" + "="*80)
    print("(i) LINK A: is the constant g(zbar)=g(0) or C=E[g(z)]?")
    print("="*80)
    z = RNG.normal(0.0, 1.0, size=8_000_000)
    C = g_slope(z).mean()
    print(f"  g(0) [doc line 91 uses g(zbar), zbar=0] = {float(g_slope(np.array([0.0]))[0]):.5f}")
    print(f"  C = E_z[g(z)], z~N(0,1)                  = {C:.5f}")
    print(f"  => doc's 'alpha g(zbar)/sigma_R' would give alpha_eff = {1.0*1.0/1.0:.5f}")
    print(f"     script's C*alpha/sigma_R gives         alpha_eff = {C:.5f}")
    alpha, sigma_R = 1.0, 1.0
    base = RNG.normal(0.0, sigma_R, size=6_000_000)
    print("  measured delta/dR (should match C, NOT g(0)):")
    for dR in [0.1, 0.05, 0.02, 0.01]:
        rA, rB = base + dR/2, base - dR/2
        pool = np.concatenate([rA, rB]); zz = (pool-pool.mean())/pool.std()
        lv = alpha*log_Rhat(zz); nA = len(rA)
        delta = lv[:nA].mean() - lv[nA:].mean()
        print(f"    dR={dR:<5}: delta/dR={delta/dR:.5f}   "
              f"(C={C:.5f}, g(0)=1.0)")

    print("\n" + "="*80)
    print("(j) is sigma_v a free parameter of the bridge? (vary it, fix alpha,sigma_R)")
    print("="*80)
    def Phi_prime0(sv):
        tau=np.sqrt(2.0)*sv
        return integrate.quad(lambda u: np.exp(u)*norm.pdf(u,0.0,tau),0.0,LN2)[0]
    dR=0.02
    rA,rB=base+dR/2,base-dR/2
    pool=np.concatenate([rA,rB]); zz=(pool-pool.mean())/pool.std()
    lv=log_Rhat(zz); nA=len(rA); delta=lv[:nA].mean()-lv[nA:].mean()
    print(f"  fixed alpha=1 sigma_R=1 -> delta={delta:.5f} (independent of sigma_v)")
    for sv in [0.2, 0.5, 1.0, 2.0]:
        s_true = Phi_closed(delta,sv)-Phi_closed(-delta,sv)
        s_bridge = 2*Phi_prime0(sv)*C*dR
        print(f"    sigma_v={sv}: 2Phi'(0)={2*Phi_prime0(sv):.5f}  "
              f"s_eff(true)={s_true:.6f}  bridge={s_bridge:.6f}  "
              f"err={abs(s_true-s_bridge)/s_true:.2%}")
    print("\nDONE.")
