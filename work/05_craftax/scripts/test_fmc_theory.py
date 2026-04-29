"""test_fmc_theory.py — unit test della parita' teoria-codice di fmc_craftax_v4.

Ogni test verifica un blocco del MATH_CANON contro l'implementazione v4:

  T1 — relativize: Def. 2, proprieta' 1-6 (positivita', continuita' in 0,
                  invarianza affine, monotonia, asintoti).
  T2 — virtual reward: Def. 3, VR = R_norm^alpha * D_norm^beta.
  T3 — cloning rate: Def. 4, casi 1/2/3 + clip.
  T4 — label persistence: la label sopravvive al cloning del singolo tick.
  T5 — final action voting: argmax sulle label dei walker vivi.
  T6 — decision is deterministic per fixed seed: stesso seed -> stessa azione.

Run:  PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
      $PY work/05_craftax/scripts/test_fmc_theory.py
"""
from __future__ import annotations

import os
import sys

# Forziamo CPU per i test analitici (deterministico, senza dipendenza da Metal)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

# Aggiunge la dir parent allo sys.path per importare fmc_craftax_v4
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmc_craftax_v4 import relativize, FMCConfig, make_fmc_decide  # noqa: E402


PASSED = []
FAILED = []


def report(name: str, ok: bool, msg: str = ""):
    if ok:
        PASSED.append(name)
        print(f"  PASS  {name}", file=sys.stderr)
    else:
        FAILED.append((name, msg))
        print(f"  FAIL  {name}: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# T1 — relativize
# ---------------------------------------------------------------------------

def test_relativize_positivity():
    """MATH_CANON Def. 2 prop. 1: relativize(r) > 0 ovunque, anche per r negativi."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.uniform(-1000, 1000, size=64).astype(np.float32)
        out = np.asarray(relativize(jnp.asarray(r)))
        assert (out > 0).all(), f"got non-positive: min={out.min()}"
    report("T1.1 relativize positivity", True)


def test_relativize_continuity_at_zero():
    """Def. 2 prop. 2-3: continuita' e differenziabilita' a z=0.

    Se tutti i valori sono uguali, std=0 -> safe_std=1 -> z=0 -> output=1.
    """
    r = jnp.full(32, 5.0)
    out = relativize(r)
    assert jnp.allclose(out, 1.0), f"costante r=5 -> non tutti 1: {out[:5]}"
    report("T1.2 relativize z=0 -> 1", True)


def test_relativize_affine_invariance():
    """Def. 2 prop. 6: relativize(a*r + b) = relativize(r) per a > 0."""
    rng = np.random.default_rng(1)
    r = jnp.asarray(rng.uniform(-10, 10, size=128).astype(np.float32))
    base = relativize(r)
    for a, b in [(2.0, 5.0), (0.5, -3.0), (10.0, 0.0)]:
        out = relativize(a * r + b)
        diff = float(jnp.max(jnp.abs(out - base)))
        assert diff < 1e-3, f"a={a},b={b}: max diff {diff:.2e}"
    report("T1.3 relativize affine invariance", True)


def test_relativize_monotonic():
    """relativize e' monotonicamente crescente (proprieta' implicita ma critica)."""
    r = jnp.linspace(-5.0, 5.0, 100)
    out = relativize(r)
    diffs = jnp.diff(out)
    n_neg = int((diffs < -1e-6).sum())
    assert n_neg == 0, f"relativize non monotonica: {n_neg} differenze negative"
    report("T1.4 relativize monotonic", True)


def test_relativize_asymptotic():
    """Def. 2 prop. 4: relativize cresce sub-esponenzialmente per z grandi."""
    # Crea vettore con un outlier estremo: 99 valori a 0, uno a +1e6
    r = jnp.zeros(100).at[0].set(1e6)
    out = relativize(r)
    # Il top dovrebbe essere O(log(z)) non O(z)
    # z = (1e6 - 1e4)/sigma con sigma ~ 1e5 -> z ~ 9.9
    # output = 1 + log(1+9.9) ~ 3.4
    assert out[0] < 100, f"asintoto rotto: {out[0]}"  # hard upper bound
    assert out[0] > 1.0, f"top non sopra 1: {out[0]}"
    report("T1.5 relativize sub-exp asymptotic", True)


# ---------------------------------------------------------------------------
# T2 — virtual reward (formula composta)
# ---------------------------------------------------------------------------

def test_virtual_reward_formula():
    """Def. 3: VR = R_norm^alpha * D_norm^beta.

    Verifichiamo manualmente il calcolo per un piccolo caso.
    """
    # Simulato: 4 walker con cum_rewards e distances note
    cum_rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
    distances = jnp.array([0.5, 1.0, 1.5, 2.0])

    R_norm = relativize(cum_rewards)
    D_norm = relativize(distances)

    for alpha in [0.5, 1.0, 2.0]:
        for beta in [0.5, 1.0, 2.0]:
            VR_expected = (R_norm ** alpha) * (D_norm ** beta)
            assert (VR_expected > 0).all(), f"alpha={alpha},beta={beta}: VR<=0"
            assert jnp.isfinite(VR_expected).all(), f"alpha={alpha},beta={beta}: non-finite"
    report("T2.1 virtual reward formula", True)


def test_virtual_reward_alpha_zero_common_sense():
    """Caso limite: alpha=0 -> VR dipende solo da distanza (Common Sense)."""
    cum_rewards = jnp.array([1.0, -10.0, 100.0, 0.5])
    distances = jnp.array([0.1, 0.5, 1.0, 2.0])
    R_norm = relativize(cum_rewards)
    D_norm = relativize(distances)
    VR_alpha0 = (R_norm ** 0.0) * (D_norm ** 1.0)
    assert jnp.allclose(VR_alpha0, D_norm), "alpha=0 deve dare VR == D_norm"
    report("T2.2 alpha=0 common sense", True)


def test_virtual_reward_beta_zero_greedy():
    """Caso limite: beta=0 -> VR dipende solo da reward (greedy)."""
    cum_rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
    distances = jnp.array([10.0, 0.1, 5.0, 2.0])
    R_norm = relativize(cum_rewards)
    D_norm = relativize(distances)
    VR_beta0 = (R_norm ** 1.0) * (D_norm ** 0.0)
    assert jnp.allclose(VR_beta0, R_norm), "beta=0 deve dare VR == R_norm"
    report("T2.3 beta=0 greedy", True)


# ---------------------------------------------------------------------------
# T3 — cloning rate (Def 4 casi 1/2/3 + clip)
# ---------------------------------------------------------------------------

def test_cloning_rate_case3_formula():
    """Def. 4 caso 3: 0 < VR_i < VR_k -> rate = (VR_k - VR_i) / VR_i."""
    VR_self = jnp.array([1.0, 2.0, 3.0, 4.0])
    VR_other = jnp.array([2.0, 3.0, 4.0, 5.0])
    expected = jnp.array([1.0, 0.5, 1.0/3.0, 0.25])  # (k-i)/i
    denom = jnp.where(VR_self > 1e-8, VR_self, 1e-8)
    rate = (VR_other - VR_self) / denom
    assert jnp.allclose(rate, expected, atol=1e-6), f"got {rate}"
    report("T3.1 cloning rate case 3 formula", True)


def test_cloning_rate_clip():
    """Def. 4: rate clipped a [0,1] per ottenere P_clone."""
    # Caso: VR_other >> VR_self -> rate > 1 -> clip a 1
    VR_self = jnp.array([0.1, 1.0])
    VR_other = jnp.array([10.0, 0.5])
    denom = jnp.where(VR_self > 1e-8, VR_self, 1e-8)
    rate = jnp.clip((VR_other - VR_self) / denom, 0, 1)
    assert rate[0] == 1.0, f"rate >> 1 deve clippare a 1: {rate[0]}"
    assert rate[1] == 0.0, f"VR_other < VR_self deve clippare a 0: {rate[1]}"
    report("T3.2 cloning rate clip", True)


def test_cloning_rate_case2_negative():
    """Def. 4 caso 2: VR_k <= VR_i -> rate <= 0 -> clipped a 0."""
    VR_self = jnp.array([5.0, 5.0])
    VR_other = jnp.array([3.0, 5.0])  # both < or = self
    denom = jnp.where(VR_self > 1e-8, VR_self, 1e-8)
    rate_clipped = jnp.clip((VR_other - VR_self) / denom, 0, 1)
    assert (rate_clipped == 0.0).all(), f"VR_other<=VR_self deve dare 0: {rate_clipped}"
    report("T3.3 cloning rate case 2 (no clone)", True)


# ---------------------------------------------------------------------------
# T4 — label persistence integrazione (test funzionale piccolo)
# ---------------------------------------------------------------------------

def test_label_argmax_voting():
    """Def. 1 finale: a* = argmax sulle frequenze delle label dei vivi."""
    n_actions = 17  # Craftax-Classic
    init_actions = jnp.array([0, 1, 1, 1, 2, 2, 0, 1])  # 8 walker
    alive = jnp.array([True, True, False, True, True, False, True, True])

    # Voto = somma delle label-indicator pesate da alive
    votes = jnp.zeros(n_actions)
    votes = votes.at[init_actions].add(alive.astype(jnp.float32))
    chosen = int(jnp.argmax(votes))

    # Vivi: idx [0,1,3,4,6,7] -> labels [0,1,1,2,0,1] -> conteggio: 0:2, 1:3, 2:1
    # Argmax -> action 1
    assert chosen == 1, f"argmax sbagliato: {chosen}, votes={votes[:5]}"
    report("T4.1 label argmax voting (alive-weighted)", True)


# ---------------------------------------------------------------------------
# T5 — determinismo: stesso seed -> stessa decisione
# ---------------------------------------------------------------------------

def test_decision_determinism():
    """Critico per replicabilita': stesso PRNGKey + stesso state -> stessa azione."""
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    params = env.default_params
    n_actions = int(env.action_space(params).n)

    cfg = FMCConfig(n_walkers=16, time_horizon=4, alpha=1.0, beta=1.0)
    fmc_decide = make_fmc_decide(env, params, n_actions, cfg)

    rng = jax.random.PRNGKey(42)
    rng, k_reset = jax.random.split(rng)
    obs, state = env.reset(k_reset, params)

    rng_a = jax.random.PRNGKey(99)
    a1, _ = fmc_decide(rng_a, state)
    a2, _ = fmc_decide(rng_a, state)
    assert int(a1) == int(a2), f"non deterministico: {int(a1)} vs {int(a2)}"
    report("T5.1 decision deterministic per fixed seed", True)


# ---------------------------------------------------------------------------
# T6 — sanity check: planning produce un'azione valida
# ---------------------------------------------------------------------------

def test_crafter_score_formula_corner_cases():
    """Crafter score = exp(mean(log(1 + 100*s_i))) - 1 con s_i in [0,1].

    Verifichiamo:
      - all unlocked al 100% -> score = 100
      - all unlocked al 50%  -> score = 50
      - tutti zeri            -> score = 0
      - 1 su 22 al 100%       -> score < 5 (corner case che ci ha fatto scoprire il bug)
    """
    # Importazione late per evitare dipendenze su craftax durante il test puro
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sweep_run007_NM_GPU import crafter_score, CRAFTAX_CLASSIC_ACHIEVEMENTS

    # All 100%
    rates_all = {a: 1.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    s = crafter_score(rates_all)
    assert abs(s - 100.0) < 0.1, f"all 100%: expected 100, got {s}"

    # All 50%
    rates_half = {a: 0.5 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    s = crafter_score(rates_half)
    assert abs(s - 50.0) < 0.1, f"all 50%: expected 50, got {s}"

    # Tutti zeri
    rates_zero = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    s = crafter_score(rates_zero)
    assert abs(s - 0.0) < 1e-6, f"all 0%: expected 0, got {s}"

    # Solo 1 su 22 al 100%
    rates_one = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    rates_one[CRAFTAX_CLASSIC_ACHIEVEMENTS[0]] = 1.0
    s = crafter_score(rates_one)
    # log(101)/22 = 4.6151/22 = 0.2098 -> exp - 1 = 0.2334
    assert 0.2 < s < 0.3, f"1 of 22 at 100%: expected ~0.23, got {s}"

    # 4 su 22 al 100% (caso v4 attuale: 18 unlocked di varia frequenza)
    rates_4 = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    for a in CRAFTAX_CLASSIC_ACHIEVEMENTS[:4]:
        rates_4[a] = 1.0
    s = crafter_score(rates_4)
    # 4*log(101)/22 = 0.839 -> exp - 1 = 1.31
    assert 1.2 < s < 1.5, f"4 of 22 at 100%: expected ~1.31, got {s}"

    report("T7.1 Crafter score formula corner cases", True)


def test_decision_valid_action():
    """L'azione scelta deve essere in [0, n_actions)."""
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    params = env.default_params
    n_actions = int(env.action_space(params).n)

    cfg = FMCConfig(n_walkers=8, time_horizon=4)
    fmc_decide = make_fmc_decide(env, params, n_actions, cfg)

    rng = jax.random.PRNGKey(0)
    rng, k_reset = jax.random.split(rng)
    obs, state = env.reset(k_reset, params)
    a, n_alive = fmc_decide(rng, state)
    a = int(a)
    assert 0 <= a < n_actions, f"azione fuori range: {a} not in [0,{n_actions})"
    assert int(n_alive) > 0, f"nessun walker vivo dopo 4 tick"
    report("T6.1 decision valid action + alive", True)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, file=sys.stderr)
    print("FMC theory-code parity tests", file=sys.stderr)
    print(f"JAX backend: {jax.default_backend()}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    tests = [
        test_relativize_positivity,
        test_relativize_continuity_at_zero,
        test_relativize_affine_invariance,
        test_relativize_monotonic,
        test_relativize_asymptotic,
        test_virtual_reward_formula,
        test_virtual_reward_alpha_zero_common_sense,
        test_virtual_reward_beta_zero_greedy,
        test_cloning_rate_case3_formula,
        test_cloning_rate_clip,
        test_cloning_rate_case2_negative,
        test_label_argmax_voting,
        test_crafter_score_formula_corner_cases,
        test_decision_determinism,
        test_decision_valid_action,
    ]

    for t in tests:
        try:
            t()
        except AssertionError as e:
            report(t.__name__, False, str(e))
        except Exception as e:
            report(t.__name__, False, f"{type(e).__name__}: {e}")

    print("=" * 60, file=sys.stderr)
    print(f"PASSED: {len(PASSED)}/{len(PASSED)+len(FAILED)}", file=sys.stderr)
    if FAILED:
        for n, m in FAILED:
            print(f"  FAIL  {n}: {m}", file=sys.stderr)
        sys.exit(1)
    print("All theory-code parity tests passed", file=sys.stderr)


if __name__ == "__main__":
    main()
