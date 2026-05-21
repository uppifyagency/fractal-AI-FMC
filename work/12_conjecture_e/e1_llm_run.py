"""Conjecture E / E1-LLM — the full test (Route B, schema S2). NEEDS the LLM.

Pre-registered design: E1_LLM_DESIGN.md sections 3, 4.1, 6, 7.

  step 2a  an LLM is given the gridworld RULES in natural language and writes
           the transition code (e1_llm_client.generate_world_model);
  step 2b  the absorbing-fidelity probe -> f_abs (design section 3);
  step 2c  FMC plans on the LLM world-model; episodes run on the TRUE simulator;
           6 layouts, alpha in {0,0.1,1.0}, beta=1, N=64, M=20, n=30; baselines
           random, greedy. Metric: death rate (design section 2 — NOT
           decision-agreement).

Verdict (design sections 6-7):
  hE1L-1  at f_abs ~ 1, FMC at low alpha keeps death <= random AND <= greedy,
          significant, on >=3 layouts  -> self-preservation survives the swap.
  hE1L-3  if at f_abs ~ 1 death ~ random -> E1-LLM FALSIFIED.

Kernel fmc-core unchanged. The LLM is only the internal world-model organ.

Run:  python work/12_conjecture_e/e1_llm_run.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from e1_llm_common import (  # noqa: E402
    ALPHAS, LAYOUTS6, N_EPISODES, WorldModelEnv, assert_worldmodel_identity,
    fabs_probe, greedy_policy, make_wm_fmc_policy, parse_layout, random_policy,
    run_episode, summarize, two_prop_z, wilson_ci,
)
from e1_llm_client import generate_world_model  # noqa: E402

MODEL = "meta/llama-3.3-70b-instruct"
FABS_INFORMATIVE = 0.90       # design section 3: test informative iff f_abs high


def main():
    t0 = time.time()
    assert assert_worldmodel_identity(), "WorldModelEnv != true kernel"
    print("sanity: WorldModelEnv(true_transition) == fmc.core.plan() : OK")

    # --- step 2a: LLM writes the world-model ---------------------------------
    print(f"\n[2a] generating world-model — {MODEL}")
    fn, src, transcript = generate_world_model(MODEL)
    rdir = _HERE / "results"
    rdir.mkdir(exist_ok=True)
    (rdir / "e1_llm_worldmodel.py").write_text(
        f"# LLM-generated gridworld world-model — {MODEL}\n"
        f"# E1-LLM Route B (code form). {len(transcript)} attempt(s).\n\n{src}\n")
    (rdir / "e1_llm_transcript.json").write_text(json.dumps(
        {"model": MODEL, "attempts": transcript}, indent=2))
    print(f"     valid world-model in {len(transcript)} attempt(s) "
          f"(statuses: {[t['status'].split(':')[0] for t in transcript]})")

    # --- step 2b: absorbing-fidelity probe -----------------------------------
    print("\n[2b] absorbing-fidelity probe (f_abs)")
    fabs = {}
    for lname, ltext in LAYOUTS6.items():
        env, _ = parse_layout(ltext)
        pr = fabs_probe(fn, env, seed=hash(lname) & 0xFFFF)
        fabs[lname] = pr
        print(f"     {lname:12s} f_abs={pr['f_abs']:.3f}  "
              f"terminal_recall={pr['terminal_recall']:.3f}  "
              f"({pr['n_probes']} probes)")
    fabs_mean = float(np.mean([fabs[L]["f_abs"] for L in fabs]))
    print(f"     mean f_abs = {fabs_mean:.3f}  "
          f"(informative gate: >= {FABS_INFORMATIVE})")

    # --- step 2c: full FMC test ----------------------------------------------
    print("\n[2c] full FMC test on the LLM world-model")
    results = {}
    for li, (lname, ltext) in enumerate(LAYOUTS6.items()):
        env, start = parse_layout(ltext)
        wm = WorldModelEnv(env, fn)
        cell = {}
        for pi, (pname, pol) in enumerate((("random", random_policy),
                                           ("greedy", greedy_policy))):
            oc = [run_episode(env, start, pol, 8_000_000 + 100_000 * li + 1000 * pi + e)
                  for e in range(N_EPISODES)]
            cell[pname] = summarize(oc)
        for a in ALPHAS:
            pol = make_wm_fmc_policy(a, wm)
            oc = [run_episode(env, start, pol,
                              8_200_000 + 100_000 * li + 10_000 * int(a * 10) + e)
                  for e in range(N_EPISODES)]
            cell[f"fmc_a{a}"] = summarize(oc)
        results[lname] = cell
        a0 = cell["fmc_a0.0"]
        print(f"  [{time.time()-t0:6.1f}s] {lname:12s} "
              f"fmc_a0 death={a0['death_rate']*100:.0f}%  "
              f"random={cell['random']['death_rate']*100:.0f}%  "
              f"greedy={cell['greedy']['death_rate']*100:.0f}%")

    verdict = _report(results, fabs, fabs_mean)
    (rdir / "e1_llm.json").write_text(json.dumps({
        "params": {"model": MODEL, "alphas": ALPHAS, "n_episodes": N_EPISODES,
                   "fabs_informative_gate": FABS_INFORMATIVE},
        "f_abs": fabs, "f_abs_mean": fabs_mean,
        "verdict": verdict, "results": results,
    }, indent=2))
    print(f"\nwrote {rdir/'e1_llm.json'}   ({time.time()-t0:.1f}s)")


def _report(results, fabs, fabs_mean):
    layouts = list(results)
    print("\n" + "=" * 80)
    print("E1-LLM RESULTS — does emergent self-preservation survive the LLM "
          "world-model?")
    print("=" * 80)
    print(f"{'layout':12s} {'policy':10s} {'death%':>8s} {'goal%':>7s} "
          f"{'vs random':>15s} {'vs greedy':>15s}")
    layout_pass = {}
    for lname in layouts:
        lr = results[lname]
        rd, gd = lr["random"]["deaths"], lr["greedy"]["deaths"]
        for pname in ("random", "greedy", "fmc_a0.0", "fmc_a0.1", "fmc_a1.0"):
            st = lr[pname]
            line = (f"{lname:12s} {pname:10s} {st['death_rate']*100:7.1f}% "
                    f"{st['goal_rate']*100:6.1f}%")
            if pname.startswith("fmc"):
                zr, pr = two_prop_z(st["deaths"], st["n"], rd, st["n"])
                zg, pg = two_prop_z(st["deaths"], st["n"], gd, st["n"])
                line += f"  z={zr:+.2f} p={pr:.3f}  z={zg:+.2f} p={pg:.3f}"
            print(line)
        a0 = lr["fmc_a0.0"]
        lo, hi = wilson_ci(a0["deaths"], a0["n"])
        zr, pr = two_prop_z(a0["deaths"], a0["n"], rd, a0["n"])
        # hE1L-1 (per layout): fmc_a0 death <= both baselines, sig below random.
        sig_below_random = (a0["deaths"] < rd) and (pr / 2.0 < 0.05)
        leq_both = (a0["deaths"] <= rd) and (a0["deaths"] <= gd)
        layout_pass[lname] = {
            "fmc_a0_death_rate": a0["death_rate"],
            "wilson95": [lo, hi], "leq_both_baselines": leq_both,
            "sig_below_random": sig_below_random}
        print(f"  -> fmc_a0.0 death={a0['death_rate']*100:.1f}% "
              f"(Wilson95 [{lo*100:.1f}%,{hi*100:.1f}%]) | <=both={leq_both} "
              f"| sig<random={sig_below_random}\n")

    n_leq = sum(v["leq_both_baselines"] for v in layout_pass.values())
    n_sig = sum(v["sig_below_random"] for v in layout_pass.values())
    # pooled fmc_a0 vs random — falsification check (hE1L-3)
    fa0_d = sum(results[L]["fmc_a0.0"]["deaths"] for L in layouts)
    rnd_d = sum(results[L]["random"]["deaths"] for L in layouts)
    n_tot = sum(results[L]["fmc_a0.0"]["n"] for L in layouts)
    z_pool, p_pool = two_prop_z(fa0_d, n_tot, rnd_d, n_tot)

    print("=" * 80)
    print("E1-LLM VERDICT (pre-registered, E1_LLM_DESIGN sections 6-7)")
    print("=" * 80)
    print(f"  LLM world-model mean f_abs = {fabs_mean:.3f} "
          f"({'informative' if fabs_mean >= FABS_INFORMATIVE else 'BELOW gate'})")
    print(f"  fmc_a0 death <= both baselines : {n_leq}/{len(layouts)} layouts")
    print(f"  fmc_a0 death sig < random      : {n_sig}/{len(layouts)} layouts")
    print(f"  pooled fmc_a0 death {fa0_d}/{n_tot} ({fa0_d/n_tot*100:.1f}%) "
          f"vs random {rnd_d}/{n_tot} ({rnd_d/n_tot*100:.1f}%)  "
          f"z={z_pool:+.2f} p={p_pool:.4f}")

    if fabs_mean < FABS_INFORMATIVE:
        verdict = "INCONCLUSIVE-FABS"
        print("  => INCONCLUSIVE — the LLM world-model is below the f_abs gate; "
              "E1-LLM cannot be read as a test of Conjecture E (it tests the "
              "LLM's modelling instead). See the f_abs sweep for the curve.")
    elif n_leq >= 3 and n_sig >= 3:
        verdict = "VERIFIED"
        print("  => E1-LLM VERIFIED (hE1L-1) — emergent self-preservation "
              "SURVIVES the world-model organ swap to an LLM.")
    elif fa0_d >= rnd_d or p_pool >= 0.05:
        verdict = "FALSIFIED"
        print("  => E1-LLM FALSIFIED (hE1L-3) — at f_abs~1 the death rate is "
              "not below random; self-preservation does not survive the swap.")
    else:
        verdict = "MIXED"
        print("  => E1-LLM MIXED — layout-dependent; report honestly.")
    return {"verdict": verdict, "n_leq_both": n_leq, "n_sig_below_random": n_sig,
            "pooled": {"fmc_a0_deaths": fa0_d, "random_deaths": rnd_d,
                       "n": n_tot, "z": z_pool, "p": p_pool},
            "layouts": layout_pass}


if __name__ == "__main__":
    main()
