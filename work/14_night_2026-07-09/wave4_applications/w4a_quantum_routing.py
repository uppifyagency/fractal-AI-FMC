"""W4A — FMC applied to qubit routing / SWAP insertion (vs SABRE).

First applicative spike of Fractal Monte Carlo (FMC) on a concrete 2026 HW
compilation problem: routing a logical circuit onto a device coupling map by
inserting SWAP gates, minimizing the SWAP count.

WHY FMC MIGHT FIT
-----------------
Routing is NP-hard, the action space is discrete (which SWAP to insert), the
simulator is a trivial reversible permutation update (<1 us/step), planning is
per-instance, and there is a strong-but-beatable baseline: SABRE.

DESIGN (fmc-core Environment protocol — immutable/atomic state)
--------------------------------------------------------------
State = (pos: logical->physical mapping, idx: next unprocessed 2q gate, swaps).
Gates are processed strictly in program order (the spec's "index of next gate").
The spec's "apply the gate if adjacent" action is folded into GREEDY EXECUTION:
after every SWAP (and at reset) we apply every consecutive gate whose two logical
qubits are currently on adjacent physical qubits, for free. This mirrors SABRE's
front-layer execution and makes the only real decision "which SWAP to insert".
Action space = the device edges (K = |E| candidate SWAPs), fixed => FMC-compatible.

REWARD (per-state, pre-relativize)
  r(s) = W_GATE * gates_done - W_SWAP * swaps - W_DIST * dist_next
where dist_next = coupling-graph shortest-path distance between the two physical
qubits of the *next* gate (0 if done). The distance term is the SAME information
SABRE's heuristic uses, so giving it to FMC is fair, not cheating. Minimizing
swaps-to-finish is equivalent to finishing in fewer steps (1 swap == 1 step),
so "complete more gates per step" == "fewer swaps".

Run:  python3 w4a_quantum_routing.py            # full run
      python3 w4a_quantum_routing.py --pilot    # quick pilot (timing/smoke)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- fmc-core (installed) + the wave3 E2 gate ------------------------------- #
_WAVE3 = Path(__file__).resolve().parent.parent / "wave3_validation"
sys.path.insert(0, str(_WAVE3))
from fmc.core import plan                                    # noqa: E402
from w34_e2_smoke import e2_divergence, _fmt as _e2_fmt      # noqa: E402

# --- qiskit SABRE ---------------------------------------------------------- #
from qiskit import QuantumCircuit                            # noqa: E402
from qiskit.transpiler import CouplingMap, PassManager       # noqa: E402
from qiskit.transpiler.passes import SabreSwap               # noqa: E402

from scipy.stats import wilcoxon                             # noqa: E402


# =========================================================================== #
# Coupling maps                                                                #
# =========================================================================== #

def linear_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def grid_edges(rows, cols):
    e = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                e.append((q, q + 1))
            if r + 1 < rows:
                e.append((q, q + cols))
    return e


def all_pairs_dist(n, edges):
    """Undirected BFS all-pairs shortest path length on the coupling graph."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    dist = np.full((n, n), n + 1, dtype=np.int64)
    for src in range(n):
        dist[src, src] = 0
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[src, v] > dist[src, u] + 1:
                    dist[src, v] = dist[src, u] + 1
                    dq.append(v)
    return dist


# =========================================================================== #
# Routing environment (fmc-core Environment protocol)                          #
# =========================================================================== #

@dataclass
class RouteState:
    pos: np.ndarray   # pos[l]  = physical location of logical qubit l
    occ: np.ndarray   # occ[p]  = logical qubit currently at physical qubit p
    idx: int          # index of next 2q gate to process
    swaps: int        # swaps inserted so far
    done: bool


class RoutingEnv:
    """SWAP-insertion router for one (coupling map, circuit) pair."""

    W_GATE = 10.0
    W_SWAP = 0.1
    W_DIST = 1.0

    def __init__(self, n, edges, gates):
        self.n = n
        self.edges = list(edges)
        self.edgeset = set(self.edges) | {(b, a) for a, b in self.edges}
        self.gates = [tuple(g) for g in gates]
        self.dist = all_pairs_dist(n, edges)
        self._actions = tuple(range(len(self.edges)))

    # ---- protocol ---- #
    def actions(self):
        return self._actions

    def reset(self):
        pos = np.arange(self.n)
        occ = np.arange(self.n)
        s = RouteState(pos=pos, occ=occ, idx=0, swaps=0, done=False)
        self._greedy_apply(s)
        return s

    def clone_state(self, s):
        return RouteState(pos=s.pos.copy(), occ=s.occ.copy(),
                          idx=s.idx, swaps=s.swaps, done=s.done)

    def step(self, s, a):
        ns = self.clone_state(s)
        if ns.done:
            return ns
        pa, pb = self.edges[a]
        la, lb = int(ns.occ[pa]), int(ns.occ[pb])
        ns.occ[pa], ns.occ[pb] = lb, la
        ns.pos[la], ns.pos[lb] = pb, pa
        ns.swaps += 1
        self._greedy_apply(ns)
        return ns

    def observe(self, s):
        return np.concatenate([s.pos.astype(np.float64), [float(s.idx)]])

    def reward(self, s):
        if s.done:
            return self.W_GATE * len(self.gates) - self.W_SWAP * s.swaps
        l1, l2 = self.gates[s.idx]
        d = float(self.dist[s.pos[l1], s.pos[l2]])
        return self.W_GATE * s.idx - self.W_SWAP * s.swaps - self.W_DIST * d

    def sample_action(self, s, rng):
        return int(rng.integers(0, len(self.edges)))

    # ---- internals ---- #
    def _greedy_apply(self, s):
        while s.idx < len(self.gates):
            l1, l2 = self.gates[s.idx]
            p1, p2 = int(s.pos[l1]), int(s.pos[l2])
            if (p1, p2) in self.edgeset:
                s.idx += 1
            else:
                break
        if s.idx >= len(self.gates):
            s.done = True


# =========================================================================== #
# Circuit generators (logical 2q gate lists)                                   #
# =========================================================================== #

def gen_random(n, depth, rng):
    gates = []
    for _ in range(depth):
        a, b = rng.choice(n, size=2, replace=False)
        gates.append((int(a), int(b)))
    return gates


def gen_ghz(n):
    return [(0, j) for j in range(1, n)]


def gen_qft(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def build_suite(n, n_random, depth, rng, include_structured=True):
    suite = []
    for k in range(n_random):
        suite.append((f"rand{k}", gen_random(n, depth, rng)))
    if include_structured:
        suite.append(("ghz", gen_ghz(n)))
        suite.append(("qft", gen_qft(n)))
        # a couple of shuffled-QFT variants (order permuted -> different routing)
        for k in range(2):
            g = gen_qft(n)
            rng.shuffle(g)
            suite.append((f"qft_shuf{k}", list(g)))
    return suite


# =========================================================================== #
# SABRE routing (identity / trivial initial layout)                            #
# =========================================================================== #

def sabre_route(n, edges, gates, seed):
    cm = CouplingMap(couplinglist=[list(e) for e in edges])
    cm.make_symmetric()
    qc = QuantumCircuit(n)
    for a, b in gates:
        qc.cx(a, b)
    t0 = time.perf_counter()
    pm = PassManager([SabreSwap(cm, heuristic="decay", seed=seed)])
    routed = pm.run(qc)
    dt = time.perf_counter() - t0
    swaps = routed.count_ops().get("swap", 0)
    return {"swaps": int(swaps), "depth": int(routed.depth()), "time": dt}


# =========================================================================== #
# FMC routing (closed-loop per-step planner) + independent validity check      #
# =========================================================================== #

def fmc_route(env, N, M, alpha, beta, seed):
    s = env.reset()
    swap_seq = []
    steps = 0
    max_steps = 4 * len(env.gates) + 40      # safety cap against oscillation
    t0 = time.perf_counter()
    while not s.done and steps < max_steps:
        a = plan(env, s, N=N, M=M, alpha=alpha, beta=beta, seed=seed * 100003 + steps)
        s = env.step(s, a)
        swap_seq.append(env.edges[a])
        steps += 1
    dt = time.perf_counter() - t0
    return s, swap_seq, dt


def build_physical_circuit(env, swap_seq):
    """Independently replay the SWAP sequence, build the physical circuit,
    and verify every 2q op lands on an adjacent physical pair."""
    n = env.n
    pos = list(range(n))
    occ = list(range(n))
    qc = QuantumCircuit(n)
    idx = 0

    def try_apply():
        nonlocal idx
        while idx < len(env.gates):
            l1, l2 = env.gates[idx]
            p1, p2 = pos[l1], pos[l2]
            if (p1, p2) in env.edgeset:
                qc.cx(p1, p2)
                idx += 1
            else:
                break

    try_apply()
    for (a, b) in swap_seq:
        la, lb = occ[a], occ[b]
        occ[a], occ[b] = lb, la
        pos[la], pos[lb] = b, a
        qc.swap(a, b)
        try_apply()

    completed = (idx == len(env.gates))
    adj_ok = True
    for inst in qc.data:
        if inst.operation.num_qubits == 2:
            qs = [qc.find_bit(q).index for q in inst.qubits]
            if (qs[0], qs[1]) not in env.edgeset:
                adj_ok = False
                break
    return qc, completed, adj_ok, idx


# =========================================================================== #
# Stats helpers                                                                #
# =========================================================================== #

def bootstrap_ci(diffs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# =========================================================================== #
# Experiment drivers                                                           #
# =========================================================================== #

def run_e2_gate(maps, N=64, M=30):
    print("\n" + "=" * 78)
    print("STEP 2 — E2 DIVERGENCE GATE (does the free swarm diverge on routing?)")
    print("=" * 78)
    rng = np.random.default_rng(7)
    rows = []
    for mname, n, edges in maps:
        # use a moderately hard random circuit for the gate
        gates = gen_random(n, depth=max(12, n * 2), rng=rng)
        env = RoutingEnv(n, edges, gates)
        s0 = env.reset()
        m = e2_divergence(env, s0, N=N, M=M, alpha=1.0, beta=1.0,
                          seeds=(0, 1, 2, 3, 4))
        m["map"] = mname
        rows.append(m)
        print(f"\n[{mname}]  K={m['K']}  n_gates={len(gates)}")
        print("  " + _e2_fmt(m))
        print(f"  verdict: {m['verdict']}")
    return rows


def run_comparison(maps, n_random, depth, fmc_N, fmc_M, fmc_seeds, sabre_seeds,
                   suite_seed=42):
    print("\n" + "=" * 78)
    print("STEP 3 — FMC vs SABRE (paired, identity initial layout for both)")
    print("=" * 78)
    all_results = {}
    for mname, n, edges in maps:
        rng = np.random.default_rng(suite_seed)
        d = depth if depth is not None else 3 * n
        suite = build_suite(n, n_random, d, rng)
        per_circ = []
        print(f"\n### coupling map: {mname}  (n={n}, |E|={len(edges)}, "
              f"circuits={len(suite)})")
        print(f"{'circuit':<12} {'gates':>5} | {'SABRE_sw':>8} {'FMC_sw':>7} "
              f"{'FMCbest':>7} | {'SAB_dep':>7} {'FMC_dep':>7} | "
              f"{'valid':>5} {'FMC_ms':>8}")
        for cname, gates in suite:
            env = RoutingEnv(n, edges, gates)

            # SABRE: best over seeds
            sab = [sabre_route(n, edges, gates, sd) for sd in sabre_seeds]
            sab_best = min(sab, key=lambda r: r["swaps"])
            sab_sw = sab_best["swaps"]
            sab_dep = sab_best["depth"]
            sab_time = float(np.mean([r["time"] for r in sab]))

            # FMC: over seeds
            fmc_runs = []
            for sd in fmc_seeds:
                s, seq, dt = fmc_route(env, fmc_N, fmc_M, 1.0, 1.0, sd)
                qc, completed, adj_ok, ncomp = build_physical_circuit(env, seq)
                fmc_runs.append({
                    "swaps": s.swaps, "depth": qc.depth(),
                    "completed": completed and s.done, "adj_ok": adj_ok,
                    "time": dt, "n_decisions": len(seq),
                })
            valid_runs = [r for r in fmc_runs if r["completed"] and r["adj_ok"]]
            all_valid = len(valid_runs) == len(fmc_runs)
            if valid_runs:
                fmc_best = min(valid_runs, key=lambda r: r["swaps"])
                fmc_mean_sw = float(np.mean([r["swaps"] for r in valid_runs]))
                fmc_best_sw = fmc_best["swaps"]
                fmc_dep = fmc_best["depth"]
            else:
                fmc_best = None
                fmc_mean_sw = float("nan")
                fmc_best_sw = None
                fmc_dep = None
            fmc_time = float(np.mean([r["time"] for r in fmc_runs]))
            fmc_dec = float(np.mean([r["n_decisions"] for r in fmc_runs]))

            per_circ.append({
                "circuit": cname, "gates": len(gates),
                "sab_sw": sab_sw, "sab_dep": sab_dep, "sab_time": sab_time,
                "fmc_mean_sw": fmc_mean_sw, "fmc_best_sw": fmc_best_sw,
                "fmc_dep": fmc_dep, "fmc_time": fmc_time,
                "fmc_dec": fmc_dec, "all_valid": all_valid,
                "n_valid": len(valid_runs), "n_seeds": len(fmc_seeds),
            })
            fb = f"{fmc_best_sw:>7d}" if fmc_best_sw is not None else f"{'FAIL':>7}"
            fm = f"{fmc_mean_sw:>7.1f}" if valid_runs else f"{'--':>7}"
            fd = f"{fmc_dep:>7d}" if fmc_dep is not None else f"{'--':>7}"
            vok = "OK" if all_valid else f"{len(valid_runs)}/{len(fmc_seeds)}"
            print(f"{cname:<12} {len(gates):>5} | {sab_sw:>8d} {fm} {fb} | "
                  f"{sab_dep:>7d} {fd} | {vok:>5} {fmc_time*1000:>8.1f}")

        all_results[mname] = per_circ
        _summarize(mname, per_circ, fmc_seeds)
    return all_results


def _summarize(mname, per_circ, fmc_seeds):
    print(f"\n--- summary [{mname}] ---")
    # paired on circuits where FMC produced a valid routing for >=1 seed
    ok = [c for c in per_circ if c["fmc_best_sw"] is not None]
    n_fail = len(per_circ) - len(ok)
    print(f"circuits: {len(per_circ)}  |  FMC produced a valid routing: {len(ok)}"
          f"  |  FMC failed entirely: {n_fail}")

    if not ok:
        print("  no valid FMC routings -> no paired comparison possible.")
        return

    sab = np.array([c["sab_sw"] for c in ok], dtype=float)
    fmc_best = np.array([c["fmc_best_sw"] for c in ok], dtype=float)
    fmc_mean = np.array([c["fmc_mean_sw"] for c in ok], dtype=float)

    for label, fmc in (("best-of-seeds", fmc_best), ("mean-over-seeds", fmc_mean)):
        diff = sab - fmc  # >0 => FMC uses fewer swaps (FMC better)
        wins = int((fmc < sab).sum())
        ties = int((fmc == sab).sum())
        loss = int((fmc > sab).sum())
        print(f"\n  [FMC {label}]  SABRE mean={sab.mean():.2f}  "
              f"FMC mean={fmc.mean():.2f}  mean(SABRE-FMC)={diff.mean():+.3f}")
        print(f"    win/tie/loss (FMC vs SABRE): {wins}/{ties}/{loss}")
        lo, hi = bootstrap_ci(diff)
        print(f"    bootstrap 95% CI of mean(SABRE-FMC): [{lo:+.3f}, {hi:+.3f}]"
              f"   ({'excludes 0' if (lo>0 or hi<0) else 'includes 0'})")
        if len(diff) >= 6 and np.any(diff != 0):
            try:
                stat, p = wilcoxon(sab, fmc, zero_method="wilcox")
                print(f"    Wilcoxon signed-rank: stat={stat:.1f}  p={p:.4g}")
            except ValueError as e:
                print(f"    Wilcoxon: n/a ({e})")

    # timing / cost
    fmc_time = np.array([c["fmc_time"] for c in ok])
    sab_time = np.array([c["sab_time"] for c in ok])
    fmc_dec = np.array([c["fmc_dec"] for c in ok])
    ms_per_dec = 1000.0 * fmc_time / np.maximum(fmc_dec, 1)
    print(f"\n  cost: SABRE {sab_time.mean()*1000:.2f} ms/circuit  |  "
          f"FMC {fmc_time.mean()*1000:.1f} ms/circuit  "
          f"({ms_per_dec.mean():.1f} ms/decision)  "
          f"=> FMC ~{fmc_time.mean()/max(sab_time.mean(),1e-9):.0f}x slower")

    # depth (only where FMC valid best exists)
    sab_dep = np.array([c["sab_dep"] for c in ok], dtype=float)
    fmc_dep = np.array([c["fmc_dep"] for c in ok], dtype=float)
    print(f"  depth: SABRE mean={sab_dep.mean():.1f}  FMC mean={fmc_dep.mean():.1f}"
          f"  mean(SABRE-FMC)={np.mean(sab_dep-fmc_dep):+.2f}")


# =========================================================================== #
# main                                                                         #
# =========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="quick smoke/timing run")
    args = ap.parse_args()

    MAPS = [
        ("linear5", 5, linear_edges(5)),
        ("grid3x3", 9, grid_edges(3, 3)),
    ]

    print("W4A — FMC vs SABRE on qubit routing / SWAP insertion")
    print(f"pilot={args.pilot}")

    # STEP 2 — E2 gate first
    run_e2_gate(MAPS, N=64, M=30)

    # STEP 3 — comparison
    if args.pilot:
        run_comparison(
            [("linear5", 5, linear_edges(5))],
            n_random=6, depth=12,
            fmc_N=48, fmc_M=15, fmc_seeds=(0, 1, 2), sabre_seeds=(0, 1, 2, 3),
        )
    else:
        run_comparison(
            MAPS,
            n_random=24, depth=None,   # depth set per-map below via wrapper
            fmc_N=64, fmc_M=20, fmc_seeds=(0, 1, 2, 3, 4),
            sabre_seeds=tuple(range(8)),
        )


if __name__ == "__main__":
    main()
