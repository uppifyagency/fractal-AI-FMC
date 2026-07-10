"""Symbolic derivation of alpha_eff for FMC relativize.

log VR = alpha * log(relativize(R)) + const  (isolating the reward channel, beta term held fixed)
z = (R - mu)/sigma
alpha_eff(R) := d(log VR)/dR   (units 1/reward, = Boltzmann inverse temperature)
"""
import sympy as sp

R, mu, sigma, alpha = sp.symbols('R mu sigma alpha', real=True, positive=True)
z = sp.symbols('z', real=True)

# relativize branches as functions of z
Rhat_neg = sp.exp(z)                       # z <= 0
Rhat_pos = 1 + sp.log(1 + z)               # z > 0

print("=== d/dz log(relativize) ===")
for name, Rhat in [("neg (z<=0)", Rhat_neg), ("pos (z>0)", Rhat_pos)]:
    dlog = sp.simplify(sp.diff(sp.log(Rhat), z))
    print(f"  {name}: d log Rhat / dz = {dlog}")

# chain rule: alpha_eff = alpha * d log Rhat/dz * dz/dR, dz/dR = 1/sigma
zexpr = (R - mu)/sigma
dz_dR = sp.diff(zexpr, R)
print(f"\ndz/dR = {dz_dR}")

print("\n=== alpha_eff(R) closed form (chain rule) ===")
for name, Rhat in [("neg (z<=0)", Rhat_neg), ("pos (z>0)", Rhat_pos)]:
    dlog_dz = sp.diff(sp.log(Rhat), z)
    alpha_eff = alpha * dlog_dz * dz_dR
    print(f"  {name}: alpha_eff = {sp.simplify(alpha_eff)}")

# value at the mean (z -> 0)
print("\n=== limits at z=0 (both branches) ===")
for name, Rhat in [("neg", Rhat_neg), ("pos", Rhat_pos)]:
    dlog_dz = sp.diff(sp.log(Rhat), z)
    lim = sp.limit(dlog_dz, z, 0, '+' if name == 'pos' else '-')
    print(f"  {name}: lim_{{z->0}} d log Rhat/dz = {lim}")

# Stein check: for z~N(0,1), E[z * f(z)] = E[f'(z)]. We use f = log Rhat.
# => regression-slope constant K == mean-derivative constant C. Confirm symbolically that
# g(z) = d log Rhat / dz is exactly what Stein returns.
print("\n=== g(z) = d log Rhat/dz (the shape factor) ===")
g_pos = sp.simplify(sp.diff(sp.log(Rhat_pos), z))
print(f"  g_pos(z) = {g_pos}")
print(f"  g_neg(z) = 1")
print(f"  g_pos(0) = {g_pos.subs(z,0)}  (matches g_neg -> C1 continuity of pressure)")
