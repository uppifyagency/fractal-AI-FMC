#!/usr/bin/env python3
"""
W31 -- Stationary-distribution check for the FMC cloning kernel.

Purpose: falsify/repair Theorem 2 of docs/MATH_CANON.md (lines 253-301), which
claims the FMC cloning kernel is a Metropolis-Hastings chain with a finite-T
Gibbs stationary distribution  pi*(x) propto R(x)^alpha * rho(x)^-beta.

Exact FMC acceptance (verified in code):
  repos/FractalAI_old/fractalai/swarm.py:527-528
      value = (vr_compas - vir_rew) / vir_rew           #  = r - 1,  r = VR_k/VR_i
      clone = (value >= np.random.random())             #  accept iff U <= r-1
  repos/fragile/src/fragile/fractalai.py:168-173         # identical rule
  =>  a_FMC(r) = P[U <= r-1] = clip(r-1, 0, 1) = min(max(r-1,0), 1)

True Metropolis-Hastings acceptance for target propto VR (symmetric uniform
proposal over partners):
  a_MH(r) = min(r, 1)

The two functions differ on (0, 2). MATH_CANON:186 asserts they are equal.

We test three things numerically:
  A. SELECTION.  Fixation probability of the fitter type from a single copy.
     - a_MH reproduces the classic Moran-with-selection formula exactly
       (async single-update = Moran birth-death).
     - a_FMC is uphill-only (down-move rate = 0) => the fitter type fixes with
       probability 1: an ABSORBING dynamics, NOT a finite-T Gibbs balance.
  B. NEUTRAL DRIFT.  With no systematic selection but fluctuating VR (the honest
     model of "Common Sense" FMC), a_FMC reproduces Wright-Fisher/Moran drift:
     heterozygosity decays with rate lambda(N) ~ 1/N  (deep dive 07: q ~= -1).
     Control: with EXACTLY constant VR, a_FMC(1)=0 => frozen, no drift.
  C. FIXATION TIME scaling with N under neutral drift (O(N) generations).

All numbers printed are produced by this script (seeded, reproducible).
"""

import numpy as np

RNG = np.random.default_rng(20260709)


# ----------------------------------------------------------------------------
# Acceptance functions
# ----------------------------------------------------------------------------
def a_fmc(r):
    """Exact FMC cloning acceptance: clip(r-1, 0, 1)."""
    return np.clip(r - 1.0, 0.0, 1.0)


def a_mh(r):
    """True Metropolis-Hastings acceptance for target propto VR: min(r, 1)."""
    return np.minimum(r, 1.0)


def a_barker(r):
    """Barker acceptance (for contrast): r/(1+r)."""
    return r / (1.0 + r)


def moran_fix_prob(s, N, i=1):
    """Classic Moran fixation prob of a type with relative fitness (1+s),
    starting from i copies out of N.  gamma = 1/(1+s)."""
    gamma = 1.0 / (1.0 + s)
    return (1.0 - gamma ** i) / (1.0 - gamma ** N)


# ----------------------------------------------------------------------------
# PART 0 -- acceptance table: show the three rules are genuinely different
# ----------------------------------------------------------------------------
def part0_acceptance_table():
    print("=" * 74)
    print("PART 0 -- acceptance a(r) for the three rules  [DIMOSTRATO, algebra]")
    print("=" * 74)
    print(f"{'r=VR_k/VR_i':>12} | {'a_FMC=clip(r-1)':>16} | {'a_MH=min(r,1)':>14} | {'a_Barker':>9}")
    print("-" * 74)
    for r in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 3.0]:
        print(f"{r:>12.2f} | {a_fmc(r):>16.4f} | {a_mh(r):>14.4f} | {a_barker(r):>9.4f}")
    print()
    print("MATH_CANON:186 asserts clip(r-1) == min(r,1).  FALSE on (0,2):")
    for r in [0.8, 1.5]:
        print(f"    r={r}:  a_FMC={a_fmc(r):.3f}  vs  a_MH={a_mh(r):.3f}   "
              f"(differ by {abs(a_fmc(r)-a_mh(r)):.3f})")
    print()


# ----------------------------------------------------------------------------
# PART A -- selection: fixation probability of the fitter type
#           async single-update  ==  exact Moran birth-death on j = #(fit type)
# ----------------------------------------------------------------------------
def fixation_prob_async(accept, s, N, runs, max_steps_factor=400):
    """Vectorised across `runs`. Start with j=1 copy of the fit type (fitness 1+s),
    N-1 copies of the wild type (fitness 1). Async single-update:
        pick i uniform (candidate to change), pick k uniform != i (partner),
        clone i->k with prob accept(VR_k/VR_i).
    For 2 types this reduces exactly to a birth-death chain on j:
        T+_j = (N-j)/N * j/(N-1) * accept(1+s)          # a B copies an A
        T-_j = j/N * (N-j)/(N-1) * accept(1/(1+s))       # an A copies a B
    Absorbing at j=0 (fit type lost) and j=N (fit type fixed).
    Returns (fixation_prob_of_fit_type, mean_gens_to_absorption)."""
    a_up = accept(np.array(1.0 + s))          # scalar
    a_dn = accept(np.array(1.0 / (1.0 + s)))
    j = np.ones(runs, dtype=np.int64)         # 1 copy of fit type
    steps = np.zeros(runs, dtype=np.int64)
    active = (j > 0) & (j < N)
    max_steps = max_steps_factor * N * N
    it = 0
    while active.any() and it < max_steps:
        it += 1
        ja = j[active].astype(np.float64)
        tp = (N - ja) / N * ja / (N - 1) * a_up
        tm = ja / N * (N - ja) / (N - 1) * a_dn
        u = RNG.random(active.sum())
        move = np.zeros(active.sum(), dtype=np.int64)
        move[u < tp] = 1
        mid = (u >= tp) & (u < tp + tm)
        move[mid] = -1
        j_new = j[active] + move
        j[active] = j_new
        steps[active] += 1
        active = (j > 0) & (j < N)
    # absorption count: convert absorption "async steps" to generations (/N)
    fixed = (j == N)
    return fixed.mean(), (steps / N).mean()


def part_a_selection():
    print("=" * 74)
    print("PART A -- fixation probability of the FITTER type (start = 1 copy)")
    print("          async single-update == exact Moran birth-death")
    print("=" * 74)
    print(f"{'s':>5} {'N':>5} | {'FMC (exact)':>12} | {'MH':>10} | "
          f"{'Moran theory':>12} | {'neutral 1/N':>11}")
    print("-" * 74)
    runs = 20000
    for s in [0.1, 0.5, 1.0]:
        for N in [32, 64, 128]:
            p_fmc, _ = fixation_prob_async(a_fmc, s, N, runs)
            p_mh, _ = fixation_prob_async(a_mh, s, N, runs)
            p_moran = moran_fix_prob(s, N, i=1)
            print(f"{s:>5.2f} {N:>5} | {p_fmc:>12.4f} | {p_mh:>10.4f} | "
                  f"{p_moran:>12.4f} | {1.0/N:>11.4f}")
    print()
    print("READ-OUT:")
    print(" * a_MH  matches the Moran-with-selection formula  -> reversible,")
    print("   detailed balance holds, finite-T selection balance (Gibbs-like).")
    print(" * a_FMC gives fixation prob ~= 1.0 for every s>0: down-move rate is")
    print("   exactly 0 (a_FMC(1/(1+s))=0) -> ABSORBING, uphill-only. There is")
    print("   NO finite-T Gibbs balance; the invariant law is a point mass.")
    print()


# ----------------------------------------------------------------------------
# PART B -- neutral drift: Wright-Fisher signature under the EXACT FMC kernel
# ----------------------------------------------------------------------------
def heterozygosity_decay(N, ticks, runs, sigma_v=0.5, constant_vr=False):
    """Synchronous FMC (tick = WF generation) with 2 types at p=0.5, NO systematic
    selection. Each tick every walker draws VR iid (type-independent):
        constant_vr=False:  VR ~ LogNormal(0, sigma_v)  (fluctuating fitness)
        constant_vr=True :  VR = 1 exactly              (control)
    Each walker picks a partner != self and clones with prob a_FMC(VR_k/VR_i).
    Returns H(t)/H(0) averaged over runs, where H = 2 p (1-p)."""
    typ = np.zeros((runs, N), dtype=np.int8)
    typ[:, : N // 2] = 1                      # exactly half type-1
    H = np.zeros(ticks + 1)
    p = typ.mean(axis=1)
    H[0] = np.mean(2 * p * (1 - p))
    for t in range(1, ticks + 1):
        if constant_vr:
            vr = np.ones((runs, N))
        else:
            vr = np.exp(RNG.normal(0.0, sigma_v, size=(runs, N)))
        # partner index != self, per (run, walker)
        offset = RNG.integers(1, N, size=(runs, N))
        idx = (np.arange(N)[None, :] + offset) % N
        vr_partner = np.take_along_axis(vr, idx, axis=1)
        r = vr_partner / vr
        accept = RNG.random((runs, N)) < a_fmc(r)
        typ_partner = np.take_along_axis(typ, idx, axis=1)
        typ = np.where(accept, typ_partner, typ)
        p = typ.mean(axis=1)
        H[t] = np.mean(2 * p * (1 - p))
    return H


def part_b_neutral():
    print("=" * 74)
    print("PART B -- neutral drift: heterozygosity decay rate lambda(N) ~ 1/N")
    print("          exact FMC kernel, tick = WF generation")
    print("=" * 74)
    Ns = [32, 64, 128, 256]
    ticks = 40
    runs = 800
    lambdas = []
    for N in Ns:
        H = heterozygosity_decay(N, ticks, runs, sigma_v=0.5)
        # fit H(t)/H(0) = (1-lambda)^t  => slope of log(H/H0) vs t = log(1-lambda)
        y = np.log(H / H[0] + 1e-300)
        t = np.arange(ticks + 1)
        use = H > H[0] * 0.02                      # before it hits the floor
        slope = np.polyfit(t[use], y[use], 1)[0]
        lam = 1.0 - np.exp(slope)
        lambdas.append(lam)
        print(f"  N={N:>4}:  lambda = {lam:.5f}    lambda*N = {lam*N:.3f}")
    Ns = np.array(Ns, float)
    lambdas = np.array(lambdas)
    q = np.polyfit(np.log(Ns), np.log(lambdas), 1)[0]
    print(f"\n  power-law fit  lambda ~ N^q :  q = {q:.3f}   "
          f"(Wright-Fisher predicts q = -1)")
    # control: constant VR -> frozen
    Hc = heterozygosity_decay(64, ticks, runs, constant_vr=True)
    print(f"\n  CONTROL (VR constant, N=64): H(end)/H(0) = {Hc[-1]/Hc[0]:.5f}"
          f"  -> a_FMC(1)=0, frozen, NO drift.")
    print("\n  READ-OUT: fluctuating-VR neutral FMC drifts like WF (q ~= -1),")
    print("  reproducing deep dive 07 (q_measured=-0.948). Exactly-constant VR")
    print("  does NOTHING under a_FMC -- the 'neutral drift' needs VR variance.")
    print()
    return q


# ----------------------------------------------------------------------------
# PART C -- fixation time scaling with N under neutral drift
# ----------------------------------------------------------------------------
def fixation_time_neutral(N, runs, sigma_v=0.5, max_ticks=4000):
    typ = np.zeros((runs, N), dtype=np.int8)
    typ[:, : N // 2] = 1
    tfix = np.full(runs, -1, dtype=np.int64)
    active = np.ones(runs, dtype=bool)
    for t in range(1, max_ticks + 1):
        na = int(active.sum())
        if na == 0:
            break
        vr = np.exp(RNG.normal(0.0, sigma_v, size=(na, N)))
        offset = RNG.integers(1, N, size=(na, N))
        idx = (np.arange(N)[None, :] + offset) % N
        r = np.take_along_axis(vr, idx, axis=1) / vr
        accept = RNG.random((na, N)) < a_fmc(r)
        tp = np.take_along_axis(typ[active], idx, axis=1)
        typ[active] = np.where(accept, tp, typ[active])
        p = typ[active].mean(axis=1)
        done = (p == 0.0) | (p == 1.0)
        act_idx = np.where(active)[0]
        newly = act_idx[done]
        tfix[newly] = t
        active[newly] = False
    return tfix[tfix > 0].mean(), (tfix > 0).mean()


def part_c_fixtime():
    print("=" * 74)
    print("PART C -- neutral fixation time vs N (expect O(N) generations)")
    print("=" * 74)
    Ns = [32, 64, 128, 256]
    runs = 600
    means = []
    for N in Ns:
        mt, frac = fixation_time_neutral(N, runs)
        means.append(mt)
        print(f"  N={N:>4}:  mean fixation time = {mt:8.1f} ticks   "
              f"(fixed fraction {frac:.3f})")
    q = np.polyfit(np.log(Ns), np.log(means), 1)[0]
    print(f"\n  power-law fit  T_fix ~ N^p :  p = {q:.3f}   "
          f"(WF/Moran predicts p = +1)")
    print()
    return q


if __name__ == "__main__":
    part0_acceptance_table()
    part_a_selection()
    qB = part_b_neutral()
    qC = part_c_fixtime()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(" A: a_FMC fixes the fitter type w.p. ~1 (absorbing); a_MH follows the")
    print("    Moran/Gibbs selection formula. The kernels are DIFFERENT => the")
    print("    'Metropolis-Hastings / finite-T Gibbs' label of Thm 2 is wrong.")
    print(f" B: neutral FMC drift exponent q = {qB:.3f}  (WF: -1).")
    print(f" C: neutral fixation time exponent p = {qC:.3f}  (WF/Moran: +1).")
    print(" => Correct invariant law of the cloning-only kernel = point mass")
    print("    (fixation); b_eff -> 1. Non-degenerate law needs mutation")
    print("    (perturbation S) => mutation-selection-drift balance, NOT Gibbs.")
