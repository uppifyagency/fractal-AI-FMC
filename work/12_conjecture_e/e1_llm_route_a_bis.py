"""E1-LLM Route A-bis — persistence enforced by the framework. Executes E1_LLM_ROUTE_A_BIS_DESIGN.md.

Route A falsified hRA-3: the online LLM world-model kept death at 35%, the cause
pinned to absorbing-persistence collapsing to 0.53. But persistence is not world
knowledge — a terminal state is absorbing by FMC's framework definition. Route
A-bis fixes the design: the harness enforces "done -> stay" structurally, and the
LLM is queried ONLY for live (done=False) states. f_abs and movement stay the
LLM's online predictions; persistence is 1.0 by construction.

ONE delta vs Route A. Everything else identical (6 layouts, Llama 3.3 70B, same
prompt, alpha in {0,0.1}, N=64, M=20, n=30, episodes on the true env, death
rate). The LLM-backed transition reuses route_a_cache.json (done=0 entries) ->
the run is cache-warm, near-zero new API calls.

Run:  python -u work/12_conjecture_e/e1_llm_route_a_bis.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from e1_llm_common import (  # noqa: E402
    LAYOUTS6, WorldModelEnv, fabs_probe, greedy_policy, make_wm_fmc_policy,
    parse_layout, random_policy, run_episode, summarize, true_transition,
    two_prop_z, wilson_ci,
)
from e1_llm_route_a import (  # noqa: E402  (import-safe: __main__-guarded)
    ALPHAS, DISP, LLMWorldModel, MODEL, N_EPISODES, local_obs,
)

# Pooled f_abs-sweep curve at alpha=0 (E1_LLM_RESULT.md section 1) — the
# pre-registered hRAb-4 reference: death rate vs absorbing fidelity.
SWEEP_FABS = [1.00, 0.98, 0.97, 0.88, 0.76, 0.50]
SWEEP_DEATH = [0.000, 0.017, 0.156, 0.339, 0.550, 0.650]


def wm_transition(wm: LLMWorldModel):
    """LLM-backed transition with framework-enforced persistence: a done walker
    stays put and stays done (NO LLM query); a live walker's step is the LLM's
    online prediction. This is the whole Route A-bis delta."""
    def fn(r, c, done, action, grid, H, W):
        if done:                                  # framework enforces absorption
            return (r, c, True)
        disp, ndone = wm.query(local_obs(grid, H, W, r, c), False, int(action))
        dr, dc = DISP[disp]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            nr, nc = r, c
        return (nr, nc, bool(ndone))
    return fn


def probe_layout(fn, env):
    """f_abs (balanced, sweep-comparable) + move-fidelity, on done=False states."""
    fa = fabs_probe(fn, env, seed=0)               # balanced [0.5,1] — sweep scale
    ok = tot = 0
    for r in range(env.H):
        for c in range(env.W):
            for a in env.actions():
                tot += 1
                tnr, tnc, _ = true_transition(r, c, False, a, env.grid,
                                              env.H, env.W)
                nr, nc, _ = fn(r, c, False, a, env.grid, env.H, env.W)
                if (nr, nc) == (tnr, tnc):
                    ok += 1
    return {"f_abs": fa["f_abs"], "terminal_recall": fa["terminal_recall"],
            "move_fidelity": ok / tot}


def main():
    t0 = time.time()
    cache_path = _HERE / "results" / "route_a_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"loaded cache: {len(cache)} entries (Route A — done=0 reused)", flush=True)
    wm = LLMWorldModel(MODEL, cache, cache_path)
    fn = wm_transition(wm)

    # --- Route A's death rates, for the hRAb-2 comparison --------------------
    route_a = json.loads((_HERE / "results" / "e1_llm_route_a.json").read_text())
    ra_death = {L: route_a["results"][L]["fmc_a0.0"]["death_rate"]
                for L in route_a["results"]}

    envs = {ln: parse_layout(lt) for ln, lt in LAYOUTS6.items()}

    print("\n[probe] f_abs + move-fidelity (done=False; persistence is 1.0 by "
          "construction)", flush=True)
    fidelity, results = {}, {}
    for li, (lname, (env, start)) in enumerate(envs.items()):
        fidelity[lname] = probe_layout(fn, env)
        m = fidelity[lname]
        print(f"  [{time.time()-t0:6.1f}s] {lname:12s} "
              f"f_abs={m['f_abs']:.2f} move={m['move_fidelity']:.2f} "
              f"persist=1.00(enforced)  (cache {len(wm.cache)}, api {wm.api_calls})",
              flush=True)

        # --- FMC test on the persistence-enforced online world-model --------
        cell = {}
        for pi, (pname, pol) in enumerate((("random", random_policy),
                                           ("greedy", greedy_policy))):
            oc = [run_episode(env, start, pol, 50_000_000 + 100_000 * li
                              + 1000 * pi + e) for e in range(N_EPISODES)]
            cell[pname] = summarize(oc)
        wm_env = WorldModelEnv(env, fn)
        for a in ALPHAS:
            p = make_wm_fmc_policy(a, wm_env)
            oc = [run_episode(env, start, p, 51_000_000 + 100_000 * li
                              + 10_000 * int(a * 10) + e)
                  for e in range(N_EPISODES)]
            cell[f"fmc_a{a}"] = summarize(oc)
        results[lname] = cell

    verdict = _report(results, fidelity, ra_death, wm, t0)
    out = _HERE / "results" / "e1_llm_route_a_bis.json"
    out.write_text(json.dumps({
        "params": {"model": MODEL, "alphas": ALPHAS, "n_episodes": N_EPISODES,
                   "persistence": "framework-enforced (done->stay)"},
        "verdict": verdict, "fidelity": fidelity,
        "api_calls_new": wm.api_calls, "results": results,
        "route_a_fmc_a0_death": ra_death,
    }, indent=2))
    print(f"\nwrote {out}   ({time.time()-t0:.1f}s)", flush=True)


def _report(results, fidelity, ra_death, wm, t0):
    layouts = list(results)
    print("\n" + "=" * 84)
    print("E1-LLM Route A-bis RESULTS — persistence enforced by the framework")
    print("=" * 84)
    print(f"\nNew API calls (beyond Route A's cache): {wm.api_calls}")

    fa = float(np.mean([fidelity[L]["f_abs"] for L in layouts]))
    mv = float(np.mean([fidelity[L]["move_fidelity"] for L in layouts]))
    print(f"fidelity means: f_abs={fa:.3f} (balanced), move={mv:.3f}, "
          f"persistence=1.000 (enforced)")

    print(f"\nhRAb-2/3 — death rate (fmc alpha=0) vs Route A and baselines:")
    print(f"{'layout':12s} {'A-bis':>7s} {'Route A':>8s} {'random':>7s} "
          f"{'greedy':>7s}  vs random")
    n_leq = n_sig = 0
    for L in layouts:
        c = results[L]
        a0, rd, gd = c["fmc_a0.0"], c["random"]["deaths"], c["greedy"]["deaths"]
        z, p = two_prop_z(a0["deaths"], a0["n"], rd, a0["n"])
        leq = a0["deaths"] <= rd and a0["deaths"] <= gd
        sig = (a0["deaths"] < rd) and (p / 2 < 0.05)
        n_leq += leq
        n_sig += sig
        print(f"{L:12s} {a0['death_rate']*100:6.0f}% {ra_death[L]*100:7.0f}% "
              f"{c['random']['death_rate']*100:6.0f}% "
              f"{c['greedy']['death_rate']*100:6.0f}%   z={z:+.2f} p={p:.3f}"
              f"{'  PASS' if (leq and sig) else ''}")
    d = sum(results[L]["fmc_a0.0"]["deaths"] for L in layouts)
    rd = sum(results[L]["random"]["deaths"] for L in layouts)
    n = sum(results[L]["fmc_a0.0"]["n"] for L in layouts)
    ra_pooled = float(np.mean(list(ra_death.values())))
    zp, pp = two_prop_z(d, n, rd, n)
    print(f"  pooled A-bis {d}/{n} ({d/n*100:.1f}%) | Route A {ra_pooled*100:.1f}% "
          f"| random {rd/n*100:.1f}%   z={zp:+.2f} p={pp:.2e}")

    # hRAb-4: does the f_abs-sweep curve predict the A-bis death?
    pred = float(np.interp(fa, SWEEP_FABS[::-1], SWEEP_DEATH[::-1]))
    print(f"\nhRAb-4 — f_abs-sweep prediction at f_abs={fa:.2f}: "
          f"death ~ {pred*100:.0f}%  (A-bis pooled: {d/n*100:.1f}%)")

    print("\n" + "-" * 84)
    print("VERDICT (pre-registered, E1_LLM_ROUTE_A_BIS_DESIGN sections 4-5)")
    print("-" * 84)
    hrab2 = d / n < ra_pooled - 1e-9
    hrab3 = n_leq >= 3 and n_sig >= 3
    print(f"  hRAb-2 directional recovery : {hrab2}  "
          f"(A-bis {d/n*100:.1f}% vs Route A {ra_pooled*100:.1f}%)")
    print(f"  hRAb-3 full recovery        : {hrab3}  "
          f"({n_leq}/6 <=baselines, {n_sig}/6 sig)")
    if hrab3:
        verdict = "ROUTE A-bis — FULL RECOVERY"
        print("  => persistence WAS the block: with it framework-enforced, the")
        print("     online merge recovers self-preservation. Strong positive.")
    elif hrab2:
        verdict = "ROUTE A-bis — PARTIAL RECOVERY"
        print("  => enforcing persistence helps substantially but does not fully")
        print("     recover survival: the residual is bounded by entry-detection")
        print(f"     fidelity (f_abs={fa:.2f} < the sweep's near-perfect threshold).")
    else:
        verdict = "ROUTE A-bis — NO RECOVERY"
        print("  => persistence was not the dominant factor — diagnosis to revisit.")
    return verdict


if __name__ == "__main__":
    main()
