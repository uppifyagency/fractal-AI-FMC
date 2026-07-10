"""W3.2 - Monte Carlo verification of the FMC relativize "effective inverse
temperature" alpha_eff(alpha, sigma_R).

Claim under test (derived symbolically in the companion .md):

    log VR = alpha * log(relativize(R)) + const          (reward channel, beta held fixed)
    z      = (R - mu_R) / sigma_R
    alpha_eff(R) := d(log VR)/dR   [units 1/reward] = the physical Boltzmann inverse temperature

    Pointwise closed form:
        alpha_eff(z) = (alpha/sigma_R) * g(z),
        g(z) = 1                                   for z <= 0
        g(z) = 1/((1+z)(1+log(1+z)))               for z >  0

    Population scalar (Gaussian z, via Stein's lemma the regression slope equals the
    mean pointwise elasticity):
        alpha_eff_bar(alpha, sigma_R) = C * alpha / sigma_R,
        C = E_{z~N(0,1)}[g(z)]  (a pure dimensionless number)

We verify:
  (A) the pointwise formula against a finite-difference derivative of log VR;
  (B) the population regression slope of log VR on R against C*alpha/sigma_R;
  (C) Stein identity: E[z log Rhat] == E[g(z)] == C;
  (D) the 1/sigma_R scaling law and alpha linearity across a grid;
  (E) robustness to a non-Gaussian (uniform) reward population.

Reference implementation of relativize copied bit-identically from
fmc-core/src/fmc/core.py:33 (== repos/FractalAI_old/fractalai/swarm.py:16).
"""
import numpy as np
from scipy import integrate

rng = np.random.default_rng(20260709)


# ---- relativize: bit-identical to fmc-core/src/fmc/core.py:33 --------------
def relativize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    std = vector.std()
    if std == 0:
        return np.ones(len(vector), dtype=np.float64)
    z = (vector - vector.mean()) / std
    out = np.empty_like(z)
    pos = z > 0
    out[pos] = np.log(1.0 + z[pos]) + 1.0
    out[~pos] = np.exp(z[~pos])
    return out


def g(z):
    """Shape factor g(z) = d log(relativize)/dz."""
    z = np.asarray(z, dtype=np.float64)
    out = np.ones_like(z)
    pos = z > 0
    zp = z[pos]
    out[pos] = 1.0 / ((1.0 + zp) * (1.0 + np.log(1.0 + zp)))
    return out


# ---- the dimensionless constant C = E_{z~N(0,1)}[g(z)] --------------------
def gaussian_C():
    phi = lambda z: np.exp(-z * z / 2.0) / np.sqrt(2.0 * np.pi)
    # negative half: g==1 -> integral of phi over (-inf,0] = 0.5
    neg = 0.5
    pos, _ = integrate.quad(lambda z: g(np.array([z]))[0] * phi(z), 0, np.inf)
    return neg + pos


C = gaussian_C()
print(f"[const] C = E_N(0,1)[g(z)] = {C:.6f}  (Gaussian-quadrature)")

# Stein cross-check of C by high-N sampling: E[z * log relativize(z-population)]
zbig = rng.standard_normal(20_000_000)
# For a standardized Gaussian sample, relativize(zbig) uses its own (near 0,1) moments;
# compute log Rhat directly from the theoretical branch to isolate the constant.
def log_relativize(z):
    z = np.asarray(z, dtype=np.float64)
    out = z.copy()                       # z<=0 branch: log(exp(z)) = z
    pos = z > 0
    out[pos] = np.log1p(np.log1p(z[pos]))  # z>0 branch: log(1+log(1+z))
    return out
logRhat = log_relativize(zbig)
C_stein = np.mean(zbig * logRhat)          # E[z log Rhat]
C_meander = np.mean(g(zbig))               # E[g(z)]
print(f"[stein] E[z*logRhat]       = {C_stein:.6f}")
print(f"[stein] E[g(z)]            = {C_meander:.6f}")
print(f"[stein] rel.err E[z logRhat] vs E[g] = {abs(C_stein - C_meander)/C_meander:.3%}")
print(f"[stein] rel.err E[g] vs quad C       = {abs(C_meander - C)/C:.3%}")


# ---- (A) pointwise formula vs finite difference ---------------------------
def logVR_of_R(R, mu, sigma, alpha):
    """log VR for a single walker whose raw reward is R, given population (mu,sigma)."""
    z = (R - mu) / sigma
    rhat = np.exp(z) if z <= 0 else (1.0 + np.log(1.0 + z))
    return alpha * np.log(rhat)


def pointwise_check(alpha, sigma, mu=3.0):
    zs = np.array([-2.0, -1.0, -0.2, 0.3, 1.0, 3.0, 8.0])
    Rs = mu + sigma * zs
    h = 1e-6
    fd = np.array([(logVR_of_R(R + h, mu, sigma, alpha)
                    - logVR_of_R(R - h, mu, sigma, alpha)) / (2 * h) for R in Rs])
    ana = (alpha / sigma) * g(zs)
    relerr = np.abs(fd - ana) / np.abs(ana)
    return zs, fd, ana, relerr


print("\n=== (A) pointwise alpha_eff: finite-difference vs closed form ===")
for alpha, sigma in [(1.0, 1.0), (2.0, 5.0), (0.5, 0.3)]:
    zs, fd, ana, relerr = pointwise_check(alpha, sigma)
    print(f"  alpha={alpha}, sigma={sigma}: max rel.err over z-grid = {relerr.max():.2e}")


# ---- (B,D) population regression slope vs C*alpha/sigma -------------------
def empirical_slope(R, alpha):
    """Slope of OLS regression of log VR on R across a walker population."""
    rhat = relativize(R)
    logVR = alpha * np.log(rhat)           # beta channel omitted (isolated)
    b = np.polyfit(R, logVR, 1)[0]
    return b


print("\n=== (B,D) population regression slope vs analytic C*alpha/sigma_R ===")
print("  (Gaussian reward populations, N=200000)")
N = 200_000
rows = []
header = f"  {'alpha':>6} {'sigma_R':>8} {'emp_slope':>12} {'analytic':>12} {'rel_err':>9}"
print(header)
for alpha in [0.5, 1.0, 2.0]:
    for sigma in [0.2, 0.5, 1.0, 3.0, 10.0]:
        R = rng.normal(5.0, sigma, N)
        emp = empirical_slope(R, alpha)
        ana = C * alpha / sigma
        rel = abs(emp - ana) / abs(ana)
        rows.append(rel)
        print(f"  {alpha:>6} {sigma:>8} {emp:>12.5f} {ana:>12.5f} {rel:>8.2%}")
print(f"  --> max rel.err (Gaussian) = {max(rows):.2%}, mean = {np.mean(rows):.2%}")


# ---- (E) robustness: non-Gaussian (uniform) population --------------------
# For non-Gaussian z, Stein no longer forces slope == C; the slope constant becomes
# C_dist = E_dist[z_std * logRhat] with z_std the standardized variable. We recompute
# the distribution-specific constant and check the 1/sigma law still holds exactly.
print("\n=== (E) robustness to non-Gaussian (Uniform) reward population ===")
# constant for the uniform distribution, computed once on a huge standardized sample
u = rng.uniform(0, 1, 20_000_000)
u_std = (u - u.mean()) / u.std()
logRhat_u = log_relativize(u_std)
C_unif = np.mean(u_std * logRhat_u)
print(f"  C_uniform = E_U[z*logRhat] = {C_unif:.6f}  (differs from Gaussian C={C:.6f})")
rows_u = []
print(header)
for alpha in [1.0, 2.0]:
    for sigma in [0.5, 2.0, 8.0]:
        width = sigma * np.sqrt(12.0)               # uniform width for target sigma
        R = rng.uniform(5.0 - width / 2, 5.0 + width / 2, N)
        emp = empirical_slope(R, alpha)
        ana = C_unif * alpha / sigma
        rel = abs(emp - ana) / abs(ana)
        rows_u.append(rel)
        print(f"  {alpha:>6} {sigma:>8} {emp:>12.5f} {ana:>12.5f} {rel:>8.2%}")
print(f"  --> max rel.err (Uniform, dist-specific C) = {max(rows_u):.2%}")


# ---- affine-invariance sanity: additive & multiplicative shaping are invisible ---
print("\n=== affine invariance: additive bonus & global rescale leave VR unchanged ===")
R0 = rng.normal(5.0, 2.0, 10_000)
vr0 = relativize(R0) ** 1.7
vr_add = relativize(R0 + 100.0) ** 1.7          # additive constant bonus
vr_mul = relativize(3.0 * R0) ** 1.7            # global multiplicative rescale
print(f"  max|VR(R+100)-VR(R)|      = {np.abs(vr_add - vr0).max():.2e}")
print(f"  max|VR(3R)-VR(R)|         = {np.abs(vr_mul - vr0).max():.2e}")
# but a per-element (structured) multiplicative shaping DOES change selection:
mask = R0 > R0.mean()                            # "tier" subset
R_struct = R0.copy(); R_struct[mask] *= 1.5      # non-uniform multiplicative shaping
vr_struct = relativize(R_struct) ** 1.7
print(f"  max|VR(structured mul)-VR| = {np.abs(vr_struct - vr0).max():.3f}  (NON-zero: structure bites)")
