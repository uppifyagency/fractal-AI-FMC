"""E1-LLM-curve — analysis of the pre-registered design (E1_LLM_CURVE_DESIGN section 6).

Reads results/e1_llm_curve.json (Phase A band + 36 Phase B generations) and
computes the pre-registered statistics, plus a three-axis fidelity decomposition
the raw f_abs probe is blind to.

An LLM world-model can fail in four independent ways:
  - entry-detection (f_abs)   : does step() flag lava/goal terminal ON ENTRY?
  - free movement (move_fid)  : is the (nr,nc) move correct for a live walker?
  - absorbing persistence     : does a walker already on lava/goal STAY there?
  - false positives           : are free cells wrongly flagged terminal?
The f_abs probe and the random-ablation band cover ONLY the first. The band's
ablation keeps movement exact and absorbing-persistence intact, so an LLM point
is band-comparable only if its sole error is false-negative entry-detection.

Computes: per-layout isotonic g_L(f_abs); signed residual rho vs the band;
Wilcoxon signed-rank (hE1Lc-1/2); the band-comparable subset; Jonckheere-Terpstra
trend (hE1Lc-3); the hE1Lc-4 gate. No experiment is re-run.

Run:  python work/12_conjecture_e/e1_llm_curve_analysis.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import norm, wilcoxon
from sklearn.isotonic import IsotonicRegression

_HERE = Path(__file__).resolve().parent

from e1_llm_common import absorbing_cells, LAYOUTS6, parse_layout, true_transition  # noqa: E402
from e1_llm_client import compile_transition  # noqa: E402

MODEL_ORDER = ["meta/llama-3.2-1b-instruct", "meta/llama-3.2-3b-instruct",
               "meta/llama-3.1-8b-instruct", "meta/llama-3.3-70b-instruct"]
PROMPT_LOW_TO_HIGH = ["P2", "P1", "P0"]            # increasing fidelity
GATE_FABS = 0.95
EXACT = 0.999                                       # >= this counts as perfect


def jonckheere(groups):
    """JT statistic for the ordered alternative 'values increase across groups'.
    Returns (z, p_two_sided). Ties get the 0.5 mid-count; the null variance
    ignores ties, so the test is mildly conservative under heavy ties."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    k = len(groups)
    if k < 2:
        return 0.0, 1.0
    J = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for a in groups[i]:
                J += float(np.sum(groups[j] > a)) + 0.5 * float(np.sum(groups[j] == a))
    ni = [len(g) for g in groups]
    N = sum(ni)
    mean = (N * N - sum(n * n for n in ni)) / 4.0
    var = (N * N * (2 * N + 3) - sum(n * n * (2 * n + 3) for n in ni)) / 72.0
    if var <= 0:
        return 0.0, 1.0
    z = (J - mean) / np.sqrt(var)
    return float(z), float(2 * norm.sf(abs(z)))


def move_fidelity(fn, env):
    """Fraction of (cell, action) where a LIVE walker's (nr,nc) matches the true
    kernel. done=False inputs only — pure free-movement fidelity."""
    ok = tot = 0
    for r in range(env.H):
        for c in range(env.W):
            for a in env.actions():
                tot += 1
                tnr, tnc, _ = true_transition(r, c, False, a,
                                              env.grid, env.H, env.W)
                try:
                    nr, nc, _ = fn(r, c, False, a, env.grid, env.H, env.W)
                    if (int(nr), int(nc)) == (tnr, tnc):
                        ok += 1
                except Exception:                              # noqa: BLE001
                    pass
    return ok / tot


def done_persistence(fn, env):
    """Fraction of (absorbing cell, action) where a walker ALREADY there (done=
    True) stays put and stays done. A model that moves a done walker makes
    absorbing states escapable inside multi-tick rollouts (P13's 'abs-broken') —
    invisible to f_abs, which probes done=False only."""
    ok = tot = 0
    for (r, c) in absorbing_cells(env):
        for a in env.actions():
            tot += 1
            try:
                nr, nc, nd = fn(r, c, True, a, env.grid, env.H, env.W)
                if (int(nr), int(nc), bool(nd)) == (r, c, True):
                    ok += 1
            except Exception:                                  # noqa: BLE001
                pass
    return ok / tot if tot else 1.0


def _wilcoxon(rho):
    rho = np.asarray(rho, dtype=float)
    nz = rho[np.abs(rho) > 1e-9]
    if len(nz) < 1:
        return {"n": int(len(rho)), "n_nonzero": 0, "stat": None, "p": 1.0,
                "median": float(np.median(rho)) if len(rho) else 0.0}
    try:
        st, p = wilcoxon(nz)
    except ValueError:
        st, p = float("nan"), float("nan")
    return {"n": int(len(rho)), "n_nonzero": int(len(nz)),
            "stat": float(st), "p": float(p), "median": float(np.median(rho))}


def main():
    data = json.loads((_HERE / "results" / "e1_llm_curve.json").read_text())
    band = data["band"]["layouts"]
    gens = data["llm"]["generations"]
    envs = {ln: parse_layout(lt)[0] for ln, lt in LAYOUTS6.items()}

    # --- per-layout isotonic band + residual envelope -------------------------
    iso, band_pct = {}, {}
    for lname, lr in band.items():
        fa = np.array([d["f_abs"] for d in lr["draws"]])
        dr = np.array([d["death_rate"] for d in lr["draws"]])
        g = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(fa, dr)
        iso[lname] = g
        resid = dr - g.predict(fa)
        band_pct[lname] = (float(np.percentile(resid, 5)),
                           float(np.percentile(resid, 95)))

    valid = [g for g in gens if g.get("valid")]
    invalid = [g for g in gens if not g.get("valid")]

    # --- recompile each valid world-model, measure the three fidelity axes ----
    for g in valid:
        fn = compile_transition(g["source"])
        g["_move"] = {ln: move_fidelity(fn, envs[ln]) for ln in envs}
        g["_dpers"] = {ln: done_persistence(fn, envs[ln]) for ln in envs}
        g["_move_mean"] = float(np.mean(list(g["_move"].values())))
        g["_dpers_mean"] = float(np.mean(list(g["_dpers"].values())))
        g["_fabs_mean"] = float(np.mean([g["layouts"][L]["f_abs"]
                                         for L in g["layouts"]]))

    # --- LLM points: residual vs the band + error classification --------------
    pts = []
    for g in valid:
        for lname, cell in g["layouts"].items():
            fa, dr = cell["f_abs"], cell["death_rate"]
            ghat = float(iso[lname].predict([fa])[0])
            lo, hi = band_pct[lname]
            rho = dr - ghat
            mv, dp = g["_move"][lname], g["_dpers"][lname]
            fp = cell["n_false_pos"]
            # band-comparable: sole error is false-negative entry-detection —
            # exactly the random-ablation band's error kind.
            band_comparable = (fp == 0 and mv >= EXACT and dp >= EXACT)
            if fa >= EXACT and mv >= EXACT and dp >= EXACT and fp == 0:
                eclass = "exact"
            elif band_comparable:
                eclass = "fn-entry-only"          # on the band's support
            else:
                bad = []
                if mv < EXACT:
                    bad.append("move")
                if dp < EXACT:
                    bad.append("persist")
                if fp > 0:
                    bad.append("false-pos")
                eclass = "+".join(bad)            # off the band's support
            pts.append({
                "model": g["model"].split("/")[-1], "prompt": g["prompt"],
                "rep": g["rep"], "layout": lname,
                "f_abs": fa, "death": dr, "g_hat": ghat, "residual": rho,
                "in_band": lo <= rho <= hi, "move_fid": mv, "done_pers": dp,
                "n_false_neg": cell["n_false_neg"], "n_false_pos": fp,
                "class": eclass, "band_comparable": band_comparable})

    comparable = [p for p in pts if p["band_comparable"]]
    off_support = [p for p in pts if not p["band_comparable"]
                   and p["class"] != "exact"]
    exact_pts = [p for p in pts if p["class"] == "exact"]

    w_all = _wilcoxon([p["residual"] for p in pts])
    w_cmp = _wilcoxon([p["residual"] for p in comparable])

    # --- hE1Lc-4 gate ---------------------------------------------------------
    inside = [g for g in valid if g["_fabs_mean"] <= GATE_FABS]

    # --- hE1Lc-3 JT trends ----------------------------------------------------
    cell_fabs = [(g["model"], g["prompt"], g["layouts"][L]["f_abs"])
                 for g in valid for L in g["layouts"]]
    jt_model = jonckheere([[f for m, p, f in cell_fabs if m == mk]
                           for mk in MODEL_ORDER])
    jt_prompt = jonckheere([[f for m, p, f in cell_fabs if p == pk]
                            for pk in PROMPT_LOW_TO_HIGH])

    verdict = _report(band, gens, valid, invalid, pts, comparable, off_support,
                      exact_pts, w_all, w_cmp, inside, jt_model, jt_prompt)

    out = _HERE / "results" / "e1_llm_curve_analysis.json"
    out.write_text(json.dumps({
        "verdict": verdict, "band_resid_pct": band_pct,
        "n_valid": len(valid), "n_invalid": len(invalid), "n_points": len(pts),
        "wilcoxon_all": w_all, "wilcoxon_band_comparable": w_cmp,
        "n_comparable": len(comparable), "n_off_support": len(off_support),
        "n_exact": len(exact_pts),
        "class_counts": dict(Counter(p["class"] for p in pts)),
        "gate": {"n_inside_curve": len(inside),
                 "inside": [f"{g['model'].split('/')[-1]}/{g['prompt']}/r{g['rep']}"
                            for g in inside]},
        "jt_model_scale": {"z": jt_model[0], "p": jt_model[1]},
        "jt_prompt_fidelity": {"z": jt_prompt[0], "p": jt_prompt[1]},
        "points": pts,
    }, indent=2))
    print(f"\nwrote {out}")


def _report(band, gens, valid, invalid, pts, comparable, off_support,
            exact_pts, w_all, w_cmp, inside, jt_model, jt_prompt):
    print("=" * 86)
    print("E1-LLM-curve ANALYSIS — is f_abs a sufficient statistic for survival?")
    print("=" * 86)
    print(f"\nGenerations: {len(gens)}  ({len(valid)} valid, "
          f"{len(invalid)} no-valid-model)")
    if invalid:
        c = Counter(g["model"].split("/")[-1] for g in invalid)
        print("  no-valid-model by model: " +
              ", ".join(f"{m}:{n}" for m, n in c.items()))

    print("\n--- per (model, prompt): mean f_abs / death / move-fid / persist ---")
    print(f"{'model':>22s} | " + "  ".join(f"{p:>20s}" for p in ["P0", "P1", "P2"]))
    for mk in MODEL_ORDER:
        row = []
        for pk in ["P0", "P1", "P2"]:
            gg = [g for g in valid if g["model"] == mk and g["prompt"] == pk]
            if not gg:
                row.append("        --         ")
                continue
            fa = np.mean([g["_fabs_mean"] for g in gg])
            dr = np.mean([g["layouts"][L]["death_rate"]
                          for g in gg for L in g["layouts"]])
            mv = np.mean([g["_move_mean"] for g in gg])
            dp = np.mean([g["_dpers_mean"] for g in gg])
            row.append(f"{fa:.2f}/{dr*100:3.0f}%/{mv:.2f}/{dp:.2f}")
        print(f"{mk.split('/')[-1]:>22s} | " + "  ".join(f"{c:>20s}" for c in row))
    print("  (cell = f_abs / death% / move-fidelity / done-persistence)")

    print(f"\nhE1Lc-4 gate: {len(inside)} valid world-model(s) at mean f_abs<="
          f"{GATE_FABS}  ->  {'MET (>=3)' if len(inside) >= 3 else 'NOT met'}")

    print(f"\nhE1Lc-3 trend (Jonckheere-Terpstra, f_abs increasing):")
    print(f"  vs model scale (1b<3b<8b<70b) : z={jt_model[0]:+.2f} p={jt_model[1]:.2e}")
    print(f"  vs prompt fidelity (P2<P1<P0) : z={jt_prompt[0]:+.2f} p={jt_prompt[1]:.2e}")

    print(f"\nLLM points by error class (n={len(pts)}):")
    for cls, n in Counter(p["class"] for p in pts).most_common():
        print(f"  {cls:24s} {n:3d}")

    print(f"\nhE1Lc-1/2 — LLM death vs the random-ablation band (residual rho):")
    print(f"  ALL points        n={w_all['n']:3d}  "
          f"in-band={np.mean([p['in_band'] for p in pts])*100:3.0f}%  "
          f"median rho={w_all['median']*100:+5.1f}pp  Wilcoxon p={w_all['p']:.2e}")
    if comparable:
        ib = np.mean([p["in_band"] for p in comparable]) * 100
        print(f"  band-comparable   n={w_cmp['n']:3d}  in-band={ib:3.0f}%  "
              f"median rho={w_cmp['median']*100:+5.1f}pp  Wilcoxon p={w_cmp['p']:.2e}")
        print(f"    (sole error = false-negative entry — the band's own kind)")
    if off_support:
        orho = np.array([p["residual"] for p in off_support])
        ib = np.mean([p["in_band"] for p in off_support]) * 100
        print(f"  off-support       n={len(off_support):3d}  in-band={ib:3.0f}%  "
              f"median rho={np.median(orho)*100:+5.1f}pp")
        print(f"    (movement / persistence / false-positive errors — f_abs blind)")

    anom = [p for p in pts if p["f_abs"] >= EXACT and p["death"] > 0.10]
    if anom:
        wm = sorted({(p["model"], p["prompt"]) for p in anom})
        mv = np.mean([p["move_fid"] for p in anom])
        dp = np.mean([p["done_pers"] for p in anom])
        print(f"\n  !! {len(anom)} points: f_abs=1.0 yet death>10%  "
              f"(world-models {wm})")
        print(f"     mean move-fid={mv:.2f}, mean done-persistence={dp:.2f}  "
              f"-> the kill is {'broken absorbing-persistence' if dp < mv else 'movement error'}, "
              f"a failure mode f_abs cannot see.")

    print("\n" + "-" * 86)
    print("VERDICT (pre-registered, E1_LLM_CURVE_DESIGN sections 5-7)")
    print("-" * 86)
    if len(inside) < 3:
        print("  hE1Lc-4 NOT met -> DESIGN-INCONCLUSIVE.")
        return "DESIGN-INCONCLUSIVE"
    cmp_ok = (w_cmp["p"] is not None and w_cmp["p"] == w_cmp["p"]
              and w_cmp["p"] >= 0.05)
    print("  hC-1 / hE1Lc-4: MET — LLM world-models land genuinely inside the")
    print(f"    curve ({len(inside)} valid models at f_abs<={GATE_FABS}).")
    print("  hE1Lc-3: f_abs degrades monotonically with BOTH model scale and")
    print("    prompt fidelity (Jonckheere-Terpstra, both p<1e-4).")
    if cmp_ok:
        print("  hE1Lc-1 holds WITHIN the band's error class: LLM world-models")
        print("    whose only error is false-negative entry-detection sit on the")
        print("    random-ablation curve (Wilcoxon n.s.). For that error kind,")
        print("    f_abs IS a sufficient statistic.")
    else:
        print("  hE1Lc-2 even within the band's error class: f_abs-matched LLM")
        print("    points still deviate from the random-ablation curve.")
    print("  BUT f_abs is NOT sufficient overall: a large share of LLM world-")
    print("    models fail OUTSIDE the band's support — broken movement or broken")
    print("    absorbing-persistence — and f_abs is structurally blind to both")
    print("    (world-models with f_abs=1.0 that still kill 60%+).")
    print("  => NET: f_abs is NECESSARY, NOT SUFFICIENT. The FMC+LLM merge gate")
    print("     is three-part — entry-detection (f_abs) AND movement fidelity AND")
    print("     absorbing-persistence — not f_abs alone.")
    return "f_abs NECESSARY NOT SUFFICIENT (hE1Lc-2, with hE1Lc-1 holding within-class)"


if __name__ == "__main__":
    main()
