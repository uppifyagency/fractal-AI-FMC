#!/usr/bin/env python3
"""
ADVERSARIAL verification of CLAIM 1 (G1): closed-form co-ancestry correction.

Independent of w6a: it does NOT use Gauss-Hermite. It uses
  (1) plain double Monte-Carlo for phi0, E[a_in^2], E[a_in*a_out];
  (2) scipy.quad semi-analytic for the same moments;
  (3) a DIRECT parent-map coalescence-counting simulation: build the actual
      parent(i) array from the exact kernel, count the fraction of distinct
      ordered pairs that share a parent -> p_coal, and compare p_coal*N to the
      closed form lambda*N. This is a completely different estimator from the
      heterozygosity-slope one in w6a/w3b.
  (4) EXACT finite-N coalescence formula (keeps (N-2)/(N-1)^2 and 1/(N-1))
      to test whether dropped terms are genuinely O(1/N^2).
"""
import numpy as np
from scipy import integrate
from scipy.stats import norm

RNG = np.random.default_rng(20260709)
LN2 = np.log(2.0)


def clip_acc(r):
    return np.clip(r - 1.0, 0.0, 1.0)


# ---- a_in(t) in closed form: a_in(t)=E_{g~N(0,sv^2)}[clip(e^{t-g}-1,0,1)] ----
# t-g ~ N(t, sv^2). Same band-integral structure as w3b Phi but with tau=sv.
def a_in_closed(t, sv):
    tau = sv
    band = np.exp(t + 0.5 * tau**2) * (norm.cdf(LN2, t + tau**2, tau) -
                                       norm.cdf(0.0, t + tau**2, tau))
    return band + 1.0 - 2.0 * norm.cdf(LN2, t, tau) + norm.cdf(0.0, t, tau)


def a_out_closed(t, sv):
    return a_in_closed(-t, sv)


# ---- moments by scipy quad over t ~ N(0,sv^2) -------------------------------
def moments_quad(sv):
    f_t = lambda t: norm.pdf(t, 0.0, sv)
    phi0 = integrate.quad(lambda t: a_in_closed(t, sv) * f_t(t), -12*sv, 12*sv)[0]
    E_in2 = integrate.quad(lambda t: a_in_closed(t, sv)**2 * f_t(t), -12*sv, 12*sv)[0]
    E_in_out = integrate.quad(lambda t: a_in_closed(t, sv)*a_out_closed(t, sv)*f_t(t),
                              -12*sv, 12*sv)[0]
    corr = E_in2 - 2*E_in_out
    return dict(phi0=phi0, E_in2=E_in2, E_in_out=E_in_out, corr=corr,
                lamN=2*phi0+corr)


# ---- moments by plain nested Monte-Carlo (no quadrature at all) --------------
def moments_mc(sv, n_t=100_000, n_g=4000, chunk=2000, rng=None):
    """Chunked double-MC to stay memory-safe."""
    rng = rng or RNG
    t = rng.normal(0.0, sv, size=n_t)
    a_in = np.empty(n_t); a_out = np.empty(n_t)
    for s in range(0, n_t, chunk):
        e = min(s + chunk, n_t)
        g = rng.normal(0.0, sv, size=(e - s, n_g))
        a_in[s:e] = clip_acc(np.exp(t[s:e, None] - g)).mean(axis=1)
        a_out[s:e] = clip_acc(np.exp(g - t[s:e, None])).mean(axis=1)
    return dict(phi0=a_in.mean(), E_in2=(a_in**2).mean(),
                E_in_out=(a_in*a_out).mean(),
                corr=(a_in**2).mean()-2*(a_in*a_out).mean())


# ---- DIRECT coalescence counting from the exact parent map -------------------
def measure_pcoal(N, sv, chains=40000, rng=None):
    """Build parent(i) for one tick; count fraction of distinct ordered pairs
       sharing a parent -> p_coal. Returns p_coal*N."""
    rng = rng or RNG
    ar = np.arange(N)
    g = rng.normal(0.0, sv, size=(chains, N))
    vr = np.exp(g)
    offset = rng.integers(1, N, size=(chains, N))
    idx = (ar[None, :] + offset) % N
    r = np.take_along_axis(vr, idx, axis=1) / vr
    accept = rng.random((chains, N)) < clip_acc(r)
    parent = np.where(accept, idx, ar[None, :])           # parent(i)
    # count coalescing distinct ordered pairs per chain, VECTORIZED:
    # bin over (chain, parent-value); m_{c,k}=#{i in chain c: parent=k};
    # pairs = sum_k m*(m-1). Offset each chain by c*N to separate bins.
    flat = (parent + (np.arange(chains)[:, None] * N)).ravel()
    counts = np.bincount(flat, minlength=chains * N).reshape(chains, N)
    pc = (counts * (counts - 1)).sum(axis=1) / (N * (N - 1))
    return pc.mean() * N


# ---- EXACT finite-N formula (keeps sub-leading coefficients) -----------------
def exact_finiteN_lamN(N, sv):
    m = moments_quad(sv)
    # need E[a_in(1-a_out)] = phi0 - E[a_in a_out]
    E_in_1m_out = m['phi0'] - m['E_in_out']
    pcoal = 2.0/(N-1)*E_in_1m_out + (N-2)/(N-1)**2 * m['E_in2']
    return pcoal * N


if __name__ == "__main__":
    print("="*78)
    print("ADVERSARIAL CLAIM 1 -- independent coalescence verification")
    print("="*78)
    for sv in [0.25, 0.5, 1.0]:
        q = moments_quad(sv)
        mc = moments_mc(sv)
        print(f"\nsigma_v={sv}")
        print(f"  phi0:      quad={q['phi0']:.5f}  MC={mc['phi0']:.5f}")
        print(f"  E[a_in^2]: quad={q['E_in2']:.5f}  MC={mc['E_in2']:.5f}")
        print(f"  E[in*out]: quad={q['E_in_out']:.5f}  MC={mc['E_in_out']:.5f}")
        print(f"  corr:      quad={q['corr']:+.5f}  MC={mc['corr']:+.5f}"
              f"   ({q['corr']/(2*q['phi0']):+.1%} of naive)")
        print(f"  CLOSED lambda*N (leading) = {q['lamN']:.5f}")

    print("\n" + "="*78)
    print("DIRECT parent-map coalescence counting  (p_coal*N)  vs closed form")
    print("="*78)
    for sv in [0.5, 1.0]:
        q = moments_quad(sv)
        print(f"\nsigma_v={sv}: closed leading lambda*N = {q['lamN']:.5f}")
        for N in [50, 100, 200, 400]:
            pcN = measure_pcoal(N, sv, chains=30000)
            exact = exact_finiteN_lamN(N, sv)
            print(f"  N={N:>4}: p_coal*N(direct)={pcN:.5f}   "
                  f"exact-finiteN formula={exact:.5f}   "
                  f"leading={q['lamN']:.5f}   "
                  f"direct-vs-exact={abs(pcN-exact)/exact:+.2%}  "
                  f"direct-vs-leading={abs(pcN-q['lamN'])/q['lamN']:+.2%}")
    print("\nDONE.")
