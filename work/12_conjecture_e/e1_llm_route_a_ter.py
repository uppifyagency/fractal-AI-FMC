"""E1-LLM Route A-ter — prior-alignment test. Executes E1_LLM_ROUTE_A_TER_DESIGN.md.

Route A-bis corrected the Route A diagnosis: the online merge fails at
entry-detection (balanced f_abs ~0.54, chance floor) — the LLM, queried per-step
from local observations without the rules, models the death tile "lava" with the
wrong prior ("obstacle to avoid", not "lethal terminal tile you step onto").

Route A-ter tests that diagnosis with ONE delta vs Route A: the death tile is
named "pit" instead of "lava" — a name whose LLM prior ("you fall into a pit, the
run ends") is meant to MATCH the world's rule. Everything else identical (6
layouts, Llama 3.3 70B, FMC params, the online per-step query, the hardened
harness). The balanced f_abs probe is the corrected canon metric.

  hRAt-1: balanced f_abs recovers >= 0.80 (vs Route A's 0.54)?  -> semantic mismatch
  hRAt-2: death rate recovers (<= baselines, sig, >=3 layouts)? -> online merge works
  hRAt-3: if f_abs does NOT recover -> the LLM refuses to predict entering ANY
          hazard regardless of name (a wisdom-vs-prediction confound).

Fresh cache (route_a_ter_cache.json) — new tile name = new queries. Kernel
fmc-core unchanged.

Run:  python -u work/12_conjecture_e/e1_llm_route_a_ter.py
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from e1_llm_common import (  # noqa: E402
    LAYOUTS6, WorldModelEnv, absorbing_cells, fabs_probe, greedy_policy,
    make_wm_fmc_policy, parse_layout, random_policy, run_episode, summarize,
    true_transition, two_prop_z, wilson_ci,
)
from e1_llm_route_a import (  # noqa: E402  (import-safe: __main__-guarded)
    ACTION_NAME, ALPHAS, DISP, MAX_TRIES, MODEL, N_EPISODES, TEMPERATURE,
    _backoff, _paced_chat, parse_response,
)
from gridworld_terminal import GOAL, LAVA, State  # noqa: E402

# === the Route A-ter delta: the death tile is "pit", not "lava" ==============
TYPE_NAME = {0: "ground", 1: "pit", 2: "goal"}

_SYSTEM = ("You are the world-model for a grid navigation task: a walker steps "
           "between tiles, and from a local view you predict the result of one "
           "step. Reply with EXACTLY the two requested lines and nothing else.")


def local_obs(grid, H, W, r, c):
    def t(rr, cc):
        if not (0 <= rr < H and 0 <= cc < W):
            return "edge"
        return TYPE_NAME[int(grid[rr][cc])]
    return (t(r, c), t(r - 1, c), t(r + 1, c), t(r, c - 1), t(r, c + 1))


def build_prompt(obs, done, action):
    """Identical to Route A's prompt except the death tile is 'pit' not 'lava'.
    States tile kinds and movement mechanics; NOT the terminal/absorbing
    dynamics — those remain the inference under test."""
    own, up, dn, lf, rt = obs
    run = "OVER" if done else "ONGOING"
    user = (
        f"The grid has four tile kinds: 'ground' (ordinary footing), 'pit', "
        f"'goal', and 'edge' (the grid boundary, outside the playable area).\n"
        f"Movement: the walker steps one tile in the chosen direction; if the "
        f"chosen direction is 'edge' it cannot move there and stays put.\n\n"
        f"A walker is on a {own} tile.\n"
        f"Its four neighbours — up: {up}, down: {dn}, left: {lf}, right: {rt}.\n"
        f"The walker's run so far is {run}.\n"
        f"Chosen step: {ACTION_NAME[action]}.\n\n"
        f"Reply with exactly two lines:\n"
        f"DISPLACEMENT: up | down | left | right | stay   (where the walker ends up)\n"
        f"RUN_OVER: yes | no   (whether the run is over after this step)")
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user}]


class LLMWorldModel:
    """Online world-model, LLM-backed, globally cached on the local observation.
    Identical machinery to Route A's; uses this module's 'pit' build_prompt."""

    def __init__(self, model, cache, cache_path=None):
        self.model = model
        self.cache = cache
        self.cache_path = cache_path
        self.api_calls = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(obs, done, action):
        return "|".join(obs) + f"|done={int(done)}|a={action}"

    def query(self, obs, done, action):
        k = self.key(obs, done, action)
        cached = self.cache.get(k)
        if cached is not None:
            self.hits += 1
            return cached[0], bool(cached[1])
        self.misses += 1
        disp, ndone = self._ask(obs, done, action)
        self.cache[k] = [disp, bool(ndone)]
        if self.cache_path is not None:
            self.cache_path.write_text(json.dumps(self.cache))
        return disp, bool(ndone)

    def _ask(self, obs, done, action):
        msgs = build_prompt(obs, done, action)
        last = None
        for attempt in range(MAX_TRIES):
            self.api_calls += 1
            try:
                parsed = parse_response(_paced_chat(self.model, msgs))
                if parsed is not None:
                    return parsed
                last = "unparseable response"
            except Exception as e:                     # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            if attempt < MAX_TRIES - 1:
                _backoff(attempt)
        raise RuntimeError(f"world-model query failed after {MAX_TRIES} tries "
                           f"(obs={obs}, done={done}, action={action}): {last}")


def wm_transition(wm):
    """LLM-backed transition, Route A style — the LLM is queried for EVERY state
    (including done=True; persistence is NOT framework-enforced here)."""
    def fn(r, c, done, action, grid, H, W):
        disp, ndone = wm.query(local_obs(grid, H, W, r, c), bool(done), int(action))
        dr, dc = DISP[disp]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            nr, nc = r, c
        return (nr, nc, bool(ndone))
    return fn


def probe_layout(fn, wm, env):
    """3-axis fidelity: balanced f_abs (sweep-comparable), move-fidelity (done=
    False), done-persistence (done=True on absorbing cells)."""
    fa = fabs_probe(fn, env, seed=0)
    mv_ok = mv_n = dp_ok = dp_n = 0
    for r in range(env.H):
        for c in range(env.W):
            for a in env.actions():
                tnr, tnc, _ = true_transition(r, c, False, a, env.grid,
                                              env.H, env.W)
                nr, nc, _ = fn(r, c, False, a, env.grid, env.H, env.W)
                mv_n += 1
                if (nr, nc) == (tnr, tnc):
                    mv_ok += 1
    for (r, c) in absorbing_cells(env):
        for a in env.actions():
            dp_n += 1
            obs = local_obs(env.grid, env.H, env.W, r, c)
            disp, ndone = wm.query(obs, True, a)
            if disp == "stay" and bool(ndone) is True:
                dp_ok += 1
    return {"f_abs": fa["f_abs"], "terminal_recall": fa["terminal_recall"],
            "move_fidelity": mv_ok / mv_n,
            "done_persistence": dp_ok / dp_n if dp_n else 1.0}


def main():
    t0 = time.time()
    cache_path = _HERE / "results" / "route_a_ter_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache:
        print(f"loaded cache: {len(cache)} entries (resume)", flush=True)
    wm = LLMWorldModel(MODEL, cache, cache_path)
    fn = wm_transition(wm)
    envs = {ln: parse_layout(lt) for ln, lt in LAYOUTS6.items()}

    print("\n[probe] 3-axis fidelity — death tile named 'pit' (warms cache)",
          flush=True)
    fidelity, results = {}, {}
    for li, (lname, (env, start)) in enumerate(envs.items()):
        fidelity[lname] = probe_layout(fn, wm, env)
        m = fidelity[lname]
        print(f"  [{time.time()-t0:6.1f}s] {lname:12s} f_abs={m['f_abs']:.2f} "
              f"move={m['move_fidelity']:.2f} persist={m['done_persistence']:.2f}  "
              f"(cache {len(wm.cache)}, api {wm.api_calls})", flush=True)
        cache_path.write_text(json.dumps(wm.cache))

        cell = {}
        for pi, (pname, pol) in enumerate((("random", random_policy),
                                           ("greedy", greedy_policy))):
            oc = [run_episode(env, start, pol, 60_000_000 + 100_000 * li
                              + 1000 * pi + e) for e in range(N_EPISODES)]
            cell[pname] = summarize(oc)
        wm_env = WorldModelEnv(env, fn)
        for a in ALPHAS:
            p = make_wm_fmc_policy(a, wm_env)
            oc = [run_episode(env, start, p, 61_000_000 + 100_000 * li
                              + 10_000 * int(a * 10) + e)
                  for e in range(N_EPISODES)]
            cell[f"fmc_a{a}"] = summarize(oc)
        results[lname] = cell
        a0 = cell["fmc_a0.0"]
        print(f"  [{time.time()-t0:6.1f}s] {lname:12s} fmc_a0 death="
              f"{a0['death_rate']*100:.0f}%  random={cell['random']['death_rate']*100:.0f}%",
              flush=True)

    verdict = _report(results, fidelity, wm, t0)
    out = _HERE / "results" / "e1_llm_route_a_ter.json"
    out.write_text(json.dumps({
        "params": {"model": MODEL, "death_tile": "pit", "alphas": ALPHAS,
                   "n_episodes": N_EPISODES},
        "verdict": verdict, "fidelity": fidelity,
        "api_calls": wm.api_calls, "results": results,
    }, indent=2))
    print(f"\nwrote {out}   ({time.time()-t0:.1f}s)", flush=True)


def _report(results, fidelity, wm, t0):
    layouts = list(results)
    print("\n" + "=" * 84)
    print("E1-LLM Route A-ter RESULTS — death tile = 'pit' (prior-alignment test)")
    print("=" * 84)
    fa = float(np.mean([fidelity[L]["f_abs"] for L in layouts]))
    mv = float(np.mean([fidelity[L]["move_fidelity"] for L in layouts]))
    dp = float(np.mean([fidelity[L]["done_persistence"] for L in layouts]))
    tr = float(np.mean([fidelity[L]["terminal_recall"] for L in layouts]))
    print(f"\n3-axis fidelity (balanced probe):")
    print(f"{'layout':12s} {'f_abs':>7s} {'recall':>7s} {'move':>7s} {'persist':>8s}")
    for L in layouts:
        m = fidelity[L]
        print(f"{L:12s} {m['f_abs']:7.2f} {m['terminal_recall']:7.2f} "
              f"{m['move_fidelity']:7.2f} {m['done_persistence']:8.2f}")
    print(f"{'mean':12s} {fa:7.2f} {tr:7.2f} {mv:7.2f} {dp:8.2f}")
    print(f"  (Route A baseline, 'lava': balanced f_abs = 0.54)")

    print(f"\ndeath rate (fmc alpha=0) vs baselines:")
    n_leq = n_sig = 0
    for L in layouts:
        c = results[L]
        a0, rd, gd = c["fmc_a0.0"], c["random"]["deaths"], c["greedy"]["deaths"]
        z, p = two_prop_z(a0["deaths"], a0["n"], rd, a0["n"])
        leq = a0["deaths"] <= rd and a0["deaths"] <= gd
        sig = (a0["deaths"] < rd) and (p / 2 < 0.05)
        n_leq += leq
        n_sig += sig
        print(f"  {L:12s} fmc_a0={a0['death_rate']*100:5.0f}%  "
              f"random={c['random']['death_rate']*100:4.0f}%  "
              f"greedy={c['greedy']['death_rate']*100:4.0f}%  "
              f"z={z:+.2f} p={p:.3f}{'  PASS' if (leq and sig) else ''}")
    d = sum(results[L]["fmc_a0.0"]["deaths"] for L in layouts)
    rd = sum(results[L]["random"]["deaths"] for L in layouts)
    n = sum(results[L]["fmc_a0.0"]["n"] for L in layouts)
    zp, pp = two_prop_z(d, n, rd, n)
    print(f"  pooled fmc_a0 {d}/{n} ({d/n*100:.1f}%) vs random {rd/n*100:.1f}%  "
          f"z={zp:+.2f} p={pp:.2e}")

    print("\n" + "-" * 84)
    print("VERDICT (pre-registered, E1_LLM_ROUTE_A_TER_DESIGN sections 3-4)")
    print("-" * 84)
    hrat1 = fa >= 0.80
    hrat2 = n_leq >= 3 and n_sig >= 3
    print(f"  hRAt-1 entry-detection recovers (f_abs>=0.80) : {hrat1}  "
          f"(f_abs={fa:.2f} vs Route A 0.54)")
    print(f"  hRAt-2 self-preservation recovers             : {hrat2}  "
          f"({n_leq}/6 <=baselines, {n_sig}/6 sig)")
    if hrat1 and hrat2:
        verdict = "ROUTE A-ter — ONLINE MERGE RECOVERS"
        print("  => the Route A failure was SEMANTIC (the 'lava' prior). With a")
        print("     prior-aligned tile name the online FMC+LLM merge WORKS.")
    elif hrat1 and not hrat2:
        verdict = "ROUTE A-ter — fidelity recovers, survival does not"
        print("  => entry-detection recovers but death does not drop enough — the")
        print("     bottleneck is elsewhere (movement, or f_abs below the sweep cliff).")
    elif not hrat1:
        verdict = "ROUTE A-ter — NO RECOVERY (deeper confound, hRAt-3)"
        print("  => renaming did NOT recover entry-detection: the LLM refuses to")
        print("     predict entering a hazard regardless of name — a wisdom-vs-")
        print("     prediction confound in the per-query online world-model.")
    else:
        verdict = "ROUTE A-ter — MIXED"
    return verdict


if __name__ == "__main__":
    try:
        main()
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        raise
