"""E1-LLM Route A — online LLM world-model, open domain. Executes E1_LLM_ROUTE_A_DESIGN.md.

Route B (E1-LLM, E1-LLM-curve): the LLM read the RULES and wrote the transition
as CODE once, offline. Route A: the LLM IS the world-model, queried ONLINE during
FMC planning, from LOCAL OBSERVATIONS only — no global layout, no rules.

  - Online   : the LLM answers per (state, action) while FMC explores the future.
  - Local    : the LLM sees the current tile type + its 4 neighbours' types
               (ground/lava/goal/edge) + the done flag — and must infer the
               dynamics (is lava terminal? does a done walker stay?) from tile
               semantics and world knowledge.
  - Cache    : the query depends only on the local observation, not coordinates;
               a global cache makes the distinct-query count bounded (R1 measured,
               not avoided). Geometry (displacement -> cell) is the harness's job.

Phases: a 4-axis fidelity probe (entry-detection f_abs, move-fidelity,
done-persistence, consistency) that also warms the cache; then the full FMC test
(6 layouts, alpha in {0,0.1}, n=30) on the cached online world-model. Episodes
run on the TRUE simulator; metric = death rate. Kernel fmc-core unchanged.

Run:  python -u work/12_conjecture_e/e1_llm_route_a.py
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from e1_llm_common import (  # noqa: E402
    BETA, HORIZON_M, LAYOUTS6, N_WALKERS, greedy_policy, make_wm_fmc_policy,
    parse_layout, random_policy, run_episode, summarize, true_transition,
    two_prop_z, wilson_ci,
)
from e1_llm_client import chat  # noqa: E402
from gridworld_terminal import GOAL, LAVA, State  # noqa: E402

# --- pre-registered parameters (E1_LLM_ROUTE_A_DESIGN sections 5-6) -----------
MODEL = "meta/llama-3.3-70b-instruct"
TEMPERATURE = 0.0
QUERY_CAP = 4000                       # hRA-1 hard cap on distinct LLM queries
ALPHAS = [0.0, 0.1]
N_EPISODES = 30
CONSISTENCY_FRACTION = 0.20            # share of cache keys re-queried for hRA-2

TYPE_NAME = {0: "ground", 1: "lava", 2: "goal"}
ACTION_NAME = {0: "move up", 1: "move down", 2: "move left",
               3: "move right", 4: "stay still"}
DISP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1),
        "right": (0, 1), "stay": (0, 0)}

_SYSTEM = ("You are the world-model for a grid navigation task: a walker steps "
           "between tiles, and from a local view you predict the result of one "
           "step. Reply with EXACTLY the two requested lines and nothing else.")

# --- API pacing + backoff ----------------------------------------------------
# The first Route A run was invalidated by silent free-tier rate-limiting: 97%
# of queries hit a fabricated ("stay", done) fallback. The fix: pace calls under
# the rate limit, back off on errors, and FAIL LOUD — never fabricate a
# world-model answer.
MIN_INTERVAL = 3.0          # seconds between LLM calls (~20/min, conservative)
MAX_TRIES = 10              # attempts before a query is a HARD failure
_last_call = [0.0]


def _backoff(attempt):
    """Exponential backoff, 90 s cap — generous, to ride out transient
    rate-limits and connection blips (the first re-run aborted on one)."""
    time.sleep(min(90.0, 5.0 * 1.7 ** attempt))


def _paced_chat(model, msgs):
    """chat() with a minimum inter-call gap, to stay under free-tier rate limits."""
    wait = MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    return chat(model, msgs, temperature=TEMPERATURE, max_tokens=200)


# === local observation + prompt ==============================================

def local_obs(grid, H, W, r, c):
    """The walker's local view: (own, up, down, left, right) tile types; a
    neighbour outside the grid is 'edge'. Coordinates never enter — this is what
    makes the query cacheable and the domain 'open' (no global layout)."""
    def t(rr, cc):
        if not (0 <= rr < H and 0 <= cc < W):
            return "edge"
        return TYPE_NAME[int(grid[rr][cc])]
    return (t(r, c), t(r - 1, c), t(r + 1, c), t(r, c - 1), t(r, c + 1))


def build_prompt(obs, done, action):
    """The world-model query. States the tile vocabulary and the movement
    mechanics (mechanical, harness-domain) — but NOT the terminal/absorbing
    dynamics of lava/goal nor that an ended run is frozen: those are the
    inference under test. Wording finalised after a 5-query fairness smoke
    test (the first wording under-specified the tile vocabulary)."""
    own, up, dn, lf, rt = obs
    run = "OVER" if done else "ONGOING"
    user = (
        f"The grid has four tile kinds: 'ground' (ordinary footing), 'lava', "
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


def parse_response(text):
    disp = done = None
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("displacement:"):
            v = low.split(":", 1)[1]
            for d in ("up", "down", "left", "right", "stay"):
                if d in v:
                    disp = d
                    break
        elif low.startswith("run_over:"):
            v = low.split(":", 1)[1]
            if "yes" in v:
                done = True
            elif "no" in v:
                done = False
    if disp is None or done is None:
        return None
    return (disp, done)


# === the online LLM world-model (cached) =====================================

class LLMWorldModel:
    """Online world-model: query(obs, done, action) -> (displacement, ndone),
    LLM-backed, globally cached on the local observation."""

    def __init__(self, model, cache, cache_path=None):
        self.model = model
        self.cache = cache                       # dict: key -> [disp, ndone]
        self.cache_path = cache_path             # checkpoint target (per miss)
        self.api_calls = 0
        self.hits = 0
        self.misses = 0
        self.parse_fail = 0

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
        if self.cache_path is not None:          # checkpoint every miss —
            self.cache_path.write_text(json.dumps(self.cache))   # abort loses ~0
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
            except Exception as e:               # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            if attempt < MAX_TRIES - 1:
                _backoff(attempt)
        # NO fabricated fallback — a query that cannot be answered is a hard
        # failure that must abort the run, not silently corrupt the world-model.
        raise RuntimeError(f"world-model query failed after {MAX_TRIES} tries "
                           f"(obs={obs}, done={done}, action={action}): {last}")


class LLMWorldModelOnlineEnv:
    """Wraps a true env; step() routes through the online LLM world-model.
    Same interface as e1_llm_common.WorldModelEnv -> reuses make_wm_fmc_policy.
    Geometry (displacement -> cell, grid clamp) is the harness's job."""

    def __init__(self, true_env, wm: LLMWorldModel):
        self.true = true_env
        self.wm = wm
        self.grid = true_env.grid
        self.H, self.W = true_env.H, true_env.W
        self.goal = true_env.goal

    def actions(self):                return self.true.actions()
    def clone_state(self, s):         return State(s.r, s.c, s.done)
    def observe(self, s):             return self.true.observe(s)
    def reward(self, s):              return self.true.reward(s)
    def sample_action(self, s, rng):  return self.true.sample_action(s, rng)

    def step(self, s, action):
        obs = local_obs(self.grid, self.H, self.W, s.r, s.c)
        disp, ndone = self.wm.query(obs, bool(s.done), int(action))
        dr, dc = DISP[disp]
        nr, nc = s.r + dr, s.c + dc
        if not (0 <= nr < self.H and 0 <= nc < self.W):
            nr, nc = s.r, s.c                    # clamp: geometry is the harness
        return State(nr, nc, bool(ndone))


# === 4-axis fidelity probe (also warms the cache) ============================

def fidelity_probe(wm, envs, t0, cache_path):
    """Enumerate every (cell, action, done in {F,T}) of all 6 layouts, query the
    online world-model, score the four fidelity axes. Warms the cache for the
    FMC test (checkpointed per layout — a hard failure is resumable). Returns
    per-layout metrics."""
    print(f"\n[probe] 4-axis fidelity — warms the cache", flush=True)
    per_layout = {}
    for li, (lname, (env, _)) in enumerate(envs.items()):
        H, W = env.H, env.W
        fa_ok = fa_n = mv_ok = mv_n = dp_ok = dp_n = 0
        for r in range(H):
            for c in range(W):
                obs = local_obs(env.grid, H, W, r, c)
                for a in range(5):
                    # done=False : entry-detection (f_abs) + move-fidelity
                    disp, ndone = wm.query(obs, False, a)
                    tnr, tnc, tdone = true_transition(r, c, False, a,
                                                      env.grid, H, W)
                    dr, dc = DISP[disp]
                    enr, enc = r + dr, c + dc
                    if not (0 <= enr < H and 0 <= enc < W):
                        enr, enc = r, c
                    fa_n += 1
                    if bool(ndone) == bool(tdone):
                        fa_ok += 1
                    mv_n += 1
                    if (enr, enc) == (tnr, tnc):
                        mv_ok += 1
                    # done=True : done-persistence (warms cache for all cells)
                    dp_disp, dp_ndone = wm.query(obs, True, a)
                    if env.grid[r][c] in (LAVA, GOAL):
                        dp_n += 1
                        if dp_disp == "stay" and bool(dp_ndone) is True:
                            dp_ok += 1
        per_layout[lname] = {
            "f_abs": fa_ok / fa_n,
            "move_fidelity": mv_ok / mv_n,
            "done_persistence": dp_ok / dp_n if dp_n else 1.0,
        }
        m = per_layout[lname]
        print(f"  [{time.time()-t0:6.1f}s] {lname:12s} "
              f"f_abs={m['f_abs']:.2f} move={m['move_fidelity']:.2f} "
              f"persist={m['done_persistence']:.2f}  "
              f"(cache {len(wm.cache)}, api {wm.api_calls})", flush=True)
        cache_path.write_text(json.dumps(wm.cache, indent=2))   # checkpoint
    return per_layout


def consistency_probe(wm, t0):
    """Re-query a random sample of cached keys; at temperature 0 a faithful
    world-model answers identically. Disagreement = API non-determinism (hRA-2)."""
    keys = list(wm.cache.keys())
    rng = np.random.default_rng(404)
    sample = [keys[i] for i in rng.permutation(len(keys))
              [:max(1, int(CONSISTENCY_FRACTION * len(keys)))]]
    same = answered = 0
    for k in sample:
        obs_part, done_part, a_part = k.rsplit("|", 2)
        obs = tuple(obs_part.split("|"))
        done = bool(int(done_part.split("=")[1]))
        a = int(a_part.split("=")[1])
        before = list(wm.cache[k])
        msgs = build_prompt(obs, done, a)
        again = None
        for attempt in range(MAX_TRIES):          # paced + backoff, like _ask
            wm.api_calls += 1
            try:
                again = parse_response(_paced_chat(wm.model, msgs))
                if again is not None:
                    break
            except Exception:                     # noqa: BLE001
                pass
            if attempt < MAX_TRIES - 1:
                _backoff(attempt)
        if again is None:
            continue                              # API failure ≠ inconsistency
        answered += 1
        if [again[0], bool(again[1])] == before:
            same += 1
    rate = same / answered if answered else 1.0
    print(f"  [{time.time()-t0:6.1f}s] consistency: {same}/{answered} = "
          f"{rate:.3f}  ({len(sample)-answered} unanswerable, excluded)", flush=True)
    return {"n_sampled": len(sample), "n_answered": answered,
            "agree": same, "rate": rate}


# === main ====================================================================

def main():
    t0 = time.time()
    envs = {ln: parse_layout(lt) for ln, lt in LAYOUTS6.items()}
    cache_path = _HERE / "results" / "route_a_cache.json"
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"loaded cache: {len(cache)} entries", flush=True)
    wm = LLMWorldModel(MODEL, cache, cache_path)

    # --- probe (warms the cache) --------------------------------------------
    fidelity = fidelity_probe(wm, envs, t0, cache_path)
    if len(wm.cache) > QUERY_CAP:
        print(f"\n!! distinct queries {len(wm.cache)} > cap {QUERY_CAP} — "
              f"hRA-1 falsified; stopping. Sparse scheme S1 needed.", flush=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        return
    cache_path.write_text(json.dumps(cache, indent=2))      # checkpoint
    consistency = consistency_probe(wm, t0)
    cache_path.write_text(json.dumps(cache, indent=2))
    api_after_probe = wm.api_calls
    distinct = len(wm.cache)

    # --- full FMC test (cache is warm -> ~0 new API calls) ------------------
    print(f"\n[test] FMC on the online LLM world-model "
          f"(cache warm, {distinct} distinct queries)", flush=True)
    results = {}
    for li, (lname, (env, start)) in enumerate(envs.items()):
        cell = {}
        for pi, (pname, pol) in enumerate((("random", random_policy),
                                           ("greedy", greedy_policy))):
            oc = [run_episode(env, start, pol, 40_000_000 + 100_000 * li
                              + 1000 * pi + e) for e in range(N_EPISODES)]
            cell[pname] = summarize(oc)
        env_llm = LLMWorldModelOnlineEnv(env, wm)
        for a in ALPHAS:
            pol = make_wm_fmc_policy(a, env_llm)
            oc = [run_episode(env, start, pol, 41_000_000 + 100_000 * li
                              + 10_000 * int(a * 10) + e)
                  for e in range(N_EPISODES)]
            cell[f"fmc_a{a}"] = summarize(oc)
        results[lname] = cell
        a0 = cell["fmc_a0.0"]
        print(f"  [{time.time()-t0:6.1f}s] {lname:12s} "
              f"fmc_a0 death={a0['death_rate']*100:.0f}%  "
              f"random={cell['random']['death_rate']*100:.0f}%  "
              f"greedy={cell['greedy']['death_rate']*100:.0f}%", flush=True)
    cache_path.write_text(json.dumps(cache, indent=2))

    fmc_api_calls = wm.api_calls - api_after_probe
    verdict = _report(results, fidelity, consistency, wm, distinct,
                      fmc_api_calls)
    out = _HERE / "results" / "e1_llm_route_a.json"
    out.write_text(json.dumps({
        "params": {"model": MODEL, "temperature": TEMPERATURE,
                   "alphas": ALPHAS, "n_episodes": N_EPISODES,
                   "query_cap": QUERY_CAP},
        "verdict": verdict,
        "cost": {"distinct_queries": distinct, "api_calls_total": wm.api_calls,
                 "api_calls_in_fmc_test": fmc_api_calls,
                 "cache_hits": wm.hits, "parse_failures": wm.parse_fail},
        "fidelity": fidelity, "consistency": consistency, "results": results,
    }, indent=2))
    print(f"\nwrote {out}   ({time.time()-t0:.1f}s)", flush=True)


def _report(results, fidelity, consistency, wm, distinct, fmc_api_calls):
    layouts = list(results)
    print("\n" + "=" * 82)
    print("E1-LLM Route A RESULTS — online LLM world-model, open domain")
    print("=" * 82)

    # --- hRA-1: cost ---------------------------------------------------------
    hit_rate = wm.hits / (wm.hits + wm.misses) if (wm.hits + wm.misses) else 0
    print(f"\nhRA-1 (cost R1):")
    print(f"  distinct LLM queries : {distinct}  (cap {QUERY_CAP})")
    print(f"  total API calls      : {wm.api_calls}  (incl. retries + consistency)")
    print(f"  API calls in FMC test: {fmc_api_calls}  (post-probe; ~0 expected)")
    print(f"  cache hit rate       : {hit_rate*100:.1f}%   parse failures: {wm.parse_fail}")
    hra1 = distinct <= QUERY_CAP

    # --- hRA-2: consistency --------------------------------------------------
    print(f"\nhRA-2 (consistency @ temp 0): {consistency['rate']:.3f} "
          f"({consistency['agree']}/{consistency['n_answered']} answered)")
    hra2 = consistency["rate"] >= 0.98

    # --- fidelity (4-axis gate) ----------------------------------------------
    print(f"\n4-axis fidelity of the online LLM world-model:")
    print(f"{'layout':12s} {'f_abs':>7s} {'move':>7s} {'persist':>8s}")
    for L in layouts:
        m = fidelity[L]
        print(f"{L:12s} {m['f_abs']:7.2f} {m['move_fidelity']:7.2f} "
              f"{m['done_persistence']:8.2f}")
    fa = float(np.mean([fidelity[L]["f_abs"] for L in layouts]))
    mv = float(np.mean([fidelity[L]["move_fidelity"] for L in layouts]))
    dp = float(np.mean([fidelity[L]["done_persistence"] for L in layouts]))
    print(f"{'mean':12s} {fa:7.2f} {mv:7.2f} {dp:8.2f}")

    # --- hRA-3: self-preservation -------------------------------------------
    print(f"\nhRA-3 (self-preservation online) — death rate:")
    print(f"{'layout':12s} {'fmc_a0':>8s} {'fmc_a0.1':>9s} {'random':>8s} "
          f"{'greedy':>8s}  R2 vs random")
    n_leq = n_sig = 0
    for L in layouts:
        c = results[L]
        a0, rd, gd = c["fmc_a0.0"], c["random"]["deaths"], c["greedy"]["deaths"]
        z, p = two_prop_z(a0["deaths"], a0["n"], rd, a0["n"])
        leq = a0["deaths"] <= rd and a0["deaths"] <= gd
        sig = (a0["deaths"] < rd) and (p / 2 < 0.05)
        n_leq += leq
        n_sig += sig
        print(f"{L:12s} {a0['death_rate']*100:7.0f}% "
              f"{c['fmc_a0.1']['death_rate']*100:8.0f}% "
              f"{c['random']['death_rate']*100:7.0f}% "
              f"{c['greedy']['death_rate']*100:7.0f}%   "
              f"z={z:+.2f} p={p:.3f} {'PASS' if (leq and sig) else ''}")
    fa0_d = sum(results[L]["fmc_a0.0"]["deaths"] for L in layouts)
    rnd_d = sum(results[L]["random"]["deaths"] for L in layouts)
    n_tot = sum(results[L]["fmc_a0.0"]["n"] for L in layouts)
    zp, pp = two_prop_z(fa0_d, n_tot, rnd_d, n_tot)
    print(f"  pooled fmc_a0 {fa0_d}/{n_tot} ({fa0_d/n_tot*100:.1f}%) vs "
          f"random {rnd_d}/{n_tot} ({rnd_d/n_tot*100:.1f}%)  z={zp:+.2f} p={pp:.2e}")
    hra3 = n_leq >= 3 and n_sig >= 3

    print("\n" + "-" * 82)
    print("VERDICT (pre-registered, E1_LLM_ROUTE_A_DESIGN sections 6-7)")
    print("-" * 82)
    print(f"  hRA-1 cost tractable   : {hra1}  ({distinct} distinct queries, "
          f"FMC test added {fmc_api_calls})")
    print(f"  hRA-2 consistency      : {hra2}  (rate {consistency['rate']:.3f})")
    print(f"  hRA-3 survives online  : {hra3}  ({n_leq}/6 <=baselines, {n_sig}/6 sig)")
    if hra1 and hra3:
        verdict = "ROUTE A VERIFIED"
        print("  => ROUTE A VERIFIED — the FMC+LLM merge holds online, from local")
        print("     observation: self-preservation survives, cost is tractable.")
    elif hra1 and not hra3:
        verdict = "ROUTE A FALSIFIED (fidelity)"
        print("  => ROUTE A FALSIFIED — online/local degrades world-model fidelity")
        print("     below the survival threshold; the merge is bounded to offline.")
    else:
        verdict = "ROUTE A — cost wall"
        print("  => cost R1 not tractable with caching alone — sparse scheme S1 next.")
    print(f"  (fidelity means: f_abs={fa:.2f}, move={mv:.2f}, persistence={dp:.2f})")
    return verdict


if __name__ == "__main__":
    try:
        main()
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        raise
