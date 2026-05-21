"""Conjecture E / E1-LLM-curve — an LLM *inside* the tolerance curve.

Pre-registered design: E1_LLM_CURVE_DESIGN.md. E1-LLM verified at f_abs=1 (easy:
the LLM world-model was functionally exact). This experiment puts LLM-generated
world-models at f_abs < 1 and asks whether their death rate lands on the
random-ablation tolerance curve (f_abs sufficient) or deviates (errors are
structured).

Two phases, one script:
  Phase A  the reference BAND — K random absorbing-ablation draws per layout,
           a cloud of (f_abs, death) points. No LLM.
  Phase B  the LLM sweep — 4 Llama models x 3 prompt-fidelity levels x 3 reps;
           each writes a world-model (Code World Model form); probe f_abs +
           error profile; FMC at alpha=0 -> death rate.

Episode runs on the TRUE simulator throughout; the swapped model is only the
internal world-model of the FMC planner. Metric: death rate. Kernel fmc-core
unchanged. alpha=0 only (the Common Sense / self-preservation regime).

Run:  python work/12_conjecture_e/e1_llm_curve.py
"""

from __future__ import annotations

import json
import textwrap
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from e1_llm_common import (  # noqa: E402
    BETA, HORIZON_M, LAYOUTS6, N_WALKERS, WorldModelEnv, absorbing_cells,
    assert_worldmodel_identity, fabs_probe, greedy_policy, make_ablated_transition,
    make_wm_fmc_policy, parse_layout, random_policy, run_episode, summarize,
    true_transition,
)
from e1_llm_client import (  # noqa: E402
    WORLD_DESCRIPTION, _SYSTEM, chat, compile_transition, extract_code,
)
from gridworld_terminal import _DELTA  # noqa: E402

# --- pre-registered parameters (E1_LLM_CURVE_DESIGN sections 2-4) -------------
ALPHA = 0.0                 # Common Sense / self-preservation regime
N_EPISODES = 30
K_DRAWS = 80                # random-ablation draws per layout (the band)
MODELS = ["meta/llama-3.2-1b-instruct", "meta/llama-3.2-3b-instruct",
          "meta/llama-3.1-8b-instruct", "meta/llama-3.3-70b-instruct"]
REPS = 3
TEMPERATURE = 0.7

# --- prompt-fidelity axis (E1_LLM_CURVE_DESIGN section 2, axis B) -------------
# P0 = the verified E1-LLM prompt. P1/P2 differ ONLY in the two paragraphs that
# describe the absorbing semantics of lava/goal and the `done` flag; the grid
# description, the action rules, the signature and the requirements are byte-
# identical across all three, so the only manipulated variable is fidelity.
P0 = WORLD_DESCRIPTION

# Locate the absorbing-semantics block by its surrounding anchors, so P1/P2
# differ from P0 in exactly that block and nowhere else.
_HEAD_END = "- A single walker occupies one cell, at row r, column c.\n"
_TAIL_START = "\n\nA step applies one action:"
assert _HEAD_END in P0 and _TAIL_START in P0, \
    "WORLD_DESCRIPTION anchors drifted — cannot build prompt variants"
_i = P0.index(_HEAD_END) + len(_HEAD_END)
_j = P0.index(_TAIL_START)

_ABS_P1 = (
    "- Lava cells are dangerous terrain the walker should avoid. The goal cell\n"
    "  is where the walker is trying to get to.\n"
    "- The walker carries a boolean `done` that records whether the episode\n"
    "  is over.")
_ABS_P2 = (
    "- Some cells contain lava and one cell is the goal. The walker moves\n"
    "  around the grid looking for the goal.\n"
    "- The walker carries a boolean `done`: set it to True once the walker\n"
    "  has reached the goal cell.")
P1 = P0[:_i] + _ABS_P1 + P0[_j:]
P2 = P0[:_i] + _ABS_P2 + P0[_j:]
assert P1 != P0 and P2 != P0 and P1 != P2, "prompt variants failed to differ"
PROMPTS = {"P0": P0, "P1": P1, "P2": P2}


# === LLM world-model generation (parametrised — e1_llm_client logic reused) ==

def generate(model, description, temperature, max_attempts=3):
    """Ask `model` to write the transition from `description`; retry on a failed
    safety gate, feeding the error back. Returns (fn|None, source|None,
    transcript). None fn = no valid world-model in max_attempts (a data point)."""
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": description}]
    transcript = []
    for attempt in range(1, max_attempts + 1):
        raw = chat(model, messages, temperature=temperature)
        src = extract_code(raw)
        rec = {"attempt": attempt, "source": src}
        try:
            fn = compile_transition(src)
            rec["status"] = "OK"
            transcript.append(rec)
            return fn, src, transcript
        except Exception as e:                                # noqa: BLE001
            rec["status"] = f"REJECTED: {e}"
            transcript.append(rec)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             f"That was rejected: {e}. Fix it and resend ONLY "
                             f"the function in one ```python block."})
    return None, None, transcript


# === error profile: false-negative vs false-positive absorbing errors ========

def error_profile(transition_fn, env):
    """Full enumeration of (cell, action): classify every absorbing-flag error.
    FN = real absorbing landing predicted non-terminal (the ablation's error
    kind). FP = free landing predicted terminal (only an LLM can make this —
    outside the band's support). Returns counts + unique wrong-cell lists."""
    fn_cells, fp_cells = set(), set()
    for r in range(env.H):
        for c in range(env.W):
            for a in env.actions():
                dr, dc = _DELTA[a]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < env.H and 0 <= nc < env.W):
                    nr, nc = r, c
                _, _, true_done = true_transition(r, c, False, a,
                                                   env.grid, env.H, env.W)
                try:
                    _, _, cand_done = transition_fn(r, c, False, a,
                                                    env.grid, env.H, env.W)
                except Exception:                             # noqa: BLE001
                    cand_done = None
                if bool(true_done) and not bool(cand_done):
                    fn_cells.add((nr, nc))
                elif not bool(true_done) and bool(cand_done):
                    fp_cells.add((nr, nc))
    return {"n_false_neg": len(fn_cells), "n_false_pos": len(fp_cells),
            "fn_cells": sorted(fn_cells), "fp_cells": sorted(fp_cells)}


# === FMC death-rate test on a swapped world-model ============================

def fmc_death(env, start, wm_env, base_seed):
    """FMC at alpha=0 planning on wm_env; n=N_EPISODES on the true env."""
    pol = make_wm_fmc_policy(ALPHA, wm_env)
    oc = [run_episode(env, start, pol, base_seed + e) for e in range(N_EPISODES)]
    return summarize(oc)


# === Phase A: the random-ablation reference band =============================

def phase_a():
    print("\n" + "=" * 78, flush=True)
    print(f"PHASE A — reference band ({K_DRAWS} random-ablation draws/layout, "
          f"alpha={ALPHA})", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    layouts = {}
    for li, (lname, ltext) in enumerate(LAYOUTS6.items()):
        env, start = parse_layout(ltext)
        abs_cells = absorbing_cells(env)
        n_abs = len(abs_cells)
        base = {}
        for pi, (pname, pol) in enumerate((("random", random_policy),
                                           ("greedy", greedy_policy))):
            oc = [run_episode(env, start, pol, 9_500_000 + li * 10_000
                              + pi * 1000 + e) for e in range(N_EPISODES)]
            base[pname] = summarize(oc)
        draws = []
        for k in range(K_DRAWS):
            rng = np.random.default_rng(5_000 + li * 100 + k)
            n_break = int(rng.integers(0, n_abs + 1))
            broken = {abs_cells[i] for i in rng.permutation(n_abs)[:n_break]}
            tfn = make_ablated_transition(broken)
            wm = WorldModelEnv(env, tfn)
            fabs = fabs_probe(tfn, env, seed=li)["f_abs"]
            st = fmc_death(env, start, wm,
                           9_000_000 + li * 1_000_000 + k * 1000)
            draws.append({"n_broken": n_break,
                          "broken": sorted(list(broken)),
                          "f_abs": fabs, **st})
        layouts[lname] = {"n_absorbing": n_abs, "baselines": base,
                          "draws": draws}
        dr = [d["death_rate"] for d in draws]
        print(f"  [{time.time()-t0:6.1f}s] {lname:12s} "
              f"f_abs∈[{min(d['f_abs'] for d in draws):.2f},"
              f"{max(d['f_abs'] for d in draws):.2f}]  "
              f"death∈[{min(dr)*100:.0f}%,{max(dr)*100:.0f}%]  "
              f"random={base['random']['death_rate']*100:.0f}%", flush=True)
    return {"params": {"K_draws": K_DRAWS, "alpha": ALPHA,
                       "n_episodes": N_EPISODES, "beta": BETA,
                       "N": N_WALKERS, "M": HORIZON_M},
            "layouts": layouts}


# === Phase B: the LLM model x prompt sweep ===================================

def phase_b(checkpoint):
    print("\n" + "=" * 78, flush=True)
    print(f"PHASE B — LLM sweep ({len(MODELS)} models x {len(PROMPTS)} prompts "
          f"x {REPS} reps, temp={TEMPERATURE})", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    envs = {ln: parse_layout(lt) for ln, lt in LAYOUTS6.items()}
    generations = []
    ci = 0
    for mi, model in enumerate(MODELS):
        for pname, descr in PROMPTS.items():
            for rep in range(REPS):
                tag = f"{model.split('/')[-1]}/{pname}/r{rep}"
                rec = {"model": model, "prompt": pname, "rep": rep}
                try:
                    fn, src, transcript = generate(model, descr, TEMPERATURE)
                except Exception as e:                        # noqa: BLE001
                    rec.update(valid=False, error=f"{type(e).__name__}: {e}",
                               n_attempts=0, statuses=[], source=None,
                               layouts={})
                    generations.append(rec)
                    print(f"  [{time.time()-t0:6.1f}s] {tag:38s} "
                          f"API-ERROR {e}", flush=True)
                    ci += 6
                    continue
                rec["n_attempts"] = len(transcript)
                rec["statuses"] = [t["status"].split(":")[0] for t in transcript]
                rec["source"] = src
                if fn is None:
                    rec.update(valid=False, layouts={})
                    generations.append(rec)
                    print(f"  [{time.time()-t0:6.1f}s] {tag:38s} "
                          f"NO-VALID-MODEL ({rec['statuses']})", flush=True)
                    ci += 6
                    continue
                lay = {}
                for li, (lname, (env, start)) in enumerate(envs.items()):
                    fabs = fabs_probe(fn, env, seed=li)
                    err = error_profile(fn, env)
                    wm = WorldModelEnv(env, fn)
                    st = fmc_death(env, start, wm, 6_000_000 + ci * 1000)
                    ci += 1
                    lay[lname] = {"f_abs": fabs["f_abs"],
                                  "terminal_recall": fabs["terminal_recall"],
                                  **err, **st}
                rec.update(valid=True, layouts=lay)
                generations.append(rec)
                fa = np.mean([lay[L]["f_abs"] for L in lay])
                dr = np.mean([lay[L]["death_rate"] for L in lay])
                print(f"  [{time.time()-t0:6.1f}s] {tag:38s} "
                      f"valid  mean f_abs={fa:.3f}  mean death={dr*100:.1f}%",
                      flush=True)
        # checkpoint after each model
        checkpoint["llm"] = {"params": {"models": MODELS,
                             "prompts": list(PROMPTS), "reps": REPS,
                             "temperature": TEMPERATURE, "alpha": ALPHA,
                             "n_episodes": N_EPISODES},
                             "generations": generations}
        _write(checkpoint)
        print(f"  -- checkpoint after model {mi+1}/{len(MODELS)} --", flush=True)
    return checkpoint["llm"]


def _write(obj):
    out = _HERE / "results" / "e1_llm_curve.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(obj, indent=2))


def main():
    t0 = time.time()
    assert assert_worldmodel_identity(), "WorldModelEnv != true kernel"
    print("sanity: WorldModelEnv(true_transition) == fmc.core.plan() : OK",
          flush=True)
    # the true model probes at f_abs=1 and has zero error — control check
    env0, _ = parse_layout(LAYOUTS6["archipelago"])
    assert abs(fabs_probe(true_transition, env0)["f_abs"] - 1.0) < 1e-9
    ep = error_profile(true_transition, env0)
    assert ep["n_false_neg"] == 0 and ep["n_false_pos"] == 0
    print("sanity: true_transition -> f_abs=1.0, 0 FN, 0 FP : OK", flush=True)

    checkpoint = {"experiment": "E1-LLM-curve", "design": "E1_LLM_CURVE_DESIGN.md"}
    checkpoint["band"] = phase_a()
    _write(checkpoint)
    checkpoint["llm"] = phase_b(checkpoint)
    _write(checkpoint)
    print(f"\nwrote {_HERE/'results'/'e1_llm_curve.json'}   "
          f"({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        raise
