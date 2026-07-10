#!/usr/bin/env python3
"""
W6A -- Closed-form co-ancestry correction to N_e for the neutral synchronous
pairwise FMC kernel. Closes the open item of Theorem 2'.5: the +13% diffusion-
coefficient correction that W3B measured but could not derive.

--------------------------------------------------------------------------
KERNEL (neutral, delta=0), exactly as w3b_mutation_diffusion.one_tick:
  each of N walkers i draws g_i ~ N(0, sigma_v^2), vr_i = exp(g_i);
  picks a uniform random partner j(i) != i (offset in {1..N-1});
  adopts partner's type with prob a_fmc(vr_j/vr_i) = clip(e^{g_j-g_i}-1, 0, 1).
  All updates synchronous (based on pre-update types).

HETEROZYGOSITY DECAY = PAIRWISE COALESCENCE (backward one tick).
  parent(i) = i           if walker i did not clone  (prob 1 - a_out(g_i))
            = j(i)         if walker i cloned         (prob   a_out(g_i))
  Two distinct offspring i1 != i2 coalesce iff they share a parent k.
  Complete enumeration (leading order in 1/N):
    (D) both clone onto the SAME k:      j(i1)=j(i2)=k, both accept
    (B) i1 is its own parent (=k), i2 clones onto i1
    (C) symmetric of B
  With
    a_in(t)  = E_{g~N(0,sv^2)}[clip(e^{t-g}-1,0,1)]  (prob a random walker clones ONTO fitness t)
    a_out(t) = E_{g~N(0,sv^2)}[clip(e^{g-t}-1,0,1)] = a_in(-t)
    phi0 = E_t[a_in(t)] = E_t[a_out(t)]   (t~N(0,sv^2))
  gives   lambda = 1/N_e = (1/N) * [ 2*phi0 - 2*E[a_in*a_out] + E[a_in^2] ]  (leading 1/N)
          => lambda*N = 2*phi0 + ( E[a_in^2] - 2*E[a_in*a_out] )   <-- co-ancestry correction
  Naive independent-flip baseline is 2*phi0; the bracket is the measured +13%.

VERIFICATION: closed-form lambda*N (Gauss-Hermite quadrature) vs directly
measured lambda*N from the exact kernel, across sigma_v and N.
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

RNG = np.random.default_rng(20260709)
LN2 = np.log(2.0)


def a_fmc(r):
    return np.clip(r - 1.0, 0.0, 1.0)


# ---- exact kernel one tick (neutral), copied from w3b_mutation_diffusion ----
def one_tick_neutral(typ, sigma_v, rng):
    chains, N = typ.shape
    ar = np.arange(N)
    g = rng.normal(0.0, sigma_v, size=(chains, N))
    vr = np.exp(g)
    offset = rng.integers(1, N, size=(chains, N))
    idx = (ar[None, :] + offset) % N
    r = np.take_along_axis(vr, idx, axis=1) / vr
    accept = rng.random((chains, N)) < a_fmc(r)
    typ = np.where(accept, np.take_along_axis(typ, idx, axis=1), typ)
    return typ.astype(np.int8)


def measure_lambdaN(N, sigma_v, ticks=40, runs=6000, rng=None):
    """Neutral heterozygosity decay rate lambda; returns lambda*N."""
    rng = rng or RNG
    typ = np.zeros((runs, N), dtype=np.int8)
    typ[:, : N // 2] = 1
    p = typ.mean(axis=1)
    H = np.empty(ticks + 1)
    H[0] = np.mean(2 * p * (1 - p))
    for t in range(1, ticks + 1):
        typ = one_tick_neutral(typ, sigma_v, rng)
        p = typ.mean(axis=1)
        H[t] = np.mean(2 * p * (1 - p))
    tt = np.arange(ticks + 1)
    use = H > H[0] * 0.02
    slope = np.polyfit(tt[use], np.log(H[use] / H[0]), 1)[0]
    lam = 1.0 - np.exp(slope)
    return lam * N


# ---- closed-form a_in via complete-the-square, then robust scipy.quad -------
def a_in(t, sigma_v):
    """a_in(t) = E_{g~N(0,sv^2)}[clip(e^{t-g}-1,0,1)] = E_{u~N(t,sv^2)}[clip(e^u-1,0,1)].
    Same closed form as w6b.Phi but with variance sv^2 (single walker), not 2 sv^2:
      = e^{t+sv^2/2}[F(ln2;t+sv^2,sv)-F(0;t+sv^2,sv)] + 1 - 2F(ln2;t,sv) + F(0;t,sv)."""
    v = sigma_v**2
    band = np.exp(t + 0.5 * v) * (norm.cdf(LN2, t + v, sigma_v) - norm.cdf(0.0, t + v, sigma_v))
    return band + 1.0 - 2.0 * norm.cdf(LN2, t, sigma_v) + norm.cdf(0.0, t, sigma_v)


def closed_form_lambdaN(sigma_v, deg=None):
    """lambda*N = 2 phi0 + (E[a_in^2] - 2 E[a_in a_out]); expectations over t~N(0,sv^2)
    computed with adaptive Gauss-Kronrod (scipy.quad) on the closed-form a_in.
    (deg kept for signature compatibility; ignored.)"""
    def Et(f):  # E_{t~N(0,sv^2)}[f(t)]
        g = lambda t: f(t) * norm.pdf(t, 0.0, sigma_v)
        return quad(g, -12 * sigma_v, 12 * sigma_v, limit=200)[0]
    phi0 = Et(lambda t: a_in(t, sigma_v))
    E_in2 = Et(lambda t: a_in(t, sigma_v) ** 2)
    E_in_out = Et(lambda t: a_in(t, sigma_v) * a_in(-t, sigma_v))
    corr = E_in2 - 2 * E_in_out
    lamN = 2 * phi0 + corr
    return dict(phi0=phi0, naive=2 * phi0, E_in2=E_in2, E_in_out=E_in_out,
                correction=corr, lamN=lamN, kappa=lamN / (2 * phi0))


if __name__ == "__main__":
    print("=" * 78)
    print("W6A -- closed-form co-ancestry N_e for neutral pairwise FMC kernel")
    print("=" * 78)

    for sv in [0.25, 0.5, 1.0]:
        cf = closed_form_lambdaN(sv)
        print(f"\nsigma_v = {sv}")
        print(f"  phi0                     = {cf['phi0']:.5f}")
        print(f"  naive 2*phi0 (indep)     = {cf['naive']:.5f}")
        print(f"  E[a_in^2]                = {cf['E_in2']:.5f}")
        print(f"  E[a_in*a_out]            = {cf['E_in_out']:.5f}")
        print(f"  co-ancestry correction   = {cf['correction']:+.5f}  "
              f"({cf['correction']/cf['naive']:+.1%} of naive)")
        print(f"  CLOSED-FORM lambda*N     = {cf['lamN']:.5f}")
        print(f"  kappa (inflation factor) = {cf['kappa']:.4f}   "
              f"=> N_e = N/(2 phi0 kappa) = N/{cf['lamN']:.4f}")

    print("\n" + "=" * 78)
    print("VALIDATION: closed-form lambda*N  vs  measured lambda*N (exact kernel)")
    print("=" * 78)
    for sv in [0.5, 1.0]:
        cf = closed_form_lambdaN(sv)
        print(f"\nsigma_v = {sv}:  closed-form lambda*N = {cf['lamN']:.4f}")
        meas = []
        for N in [100, 200, 400, 800]:
            m = measure_lambdaN(N, sv)
            meas.append(m)
            print(f"  N={N:>4}: measured lambda*N = {m:.4f}   "
                  f"rel.err vs closed-form = {abs(m-cf['lamN'])/cf['lamN']:+.1%}")
        mm = np.mean(meas)
        print(f"  mean measured = {mm:.4f}   vs closed-form {cf['lamN']:.4f}   "
              f"=> {abs(mm-cf['lamN'])/cf['lamN']:+.1%}")

    print("\n" + "=" * 78)
    print("DONE (seed 20260709). Closed form derived, not fitted.")
    print("=" * 78)
