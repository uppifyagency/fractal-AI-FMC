#!/usr/bin/env python3
"""fractal_loop.py — M-tick FMC state machine for goal-directed coding.

Faithful translation of paper §4.3 (Hernández-Cerezo & Duran-Ballester 2020),
adapted to the OCTOPUS structure (the user's vision):

    octopus_loop(goal):
        while not goal_reached(main, goal):
            next_commit = fmc_decide(main_HEAD, goal, N, M)
            cherry_pick(next_commit, main)

    fmc_decide(state, goal, N, M):
        # spawn N worktrees with distinct strategies
        for tick in 1..M:
            for w in walkers:
                if tick == 1:
                    w.commit(prompt=w.init_strategy)
                else:
                    w.commit(prompt="continuation")
            score, distance, virtual_reward
            if ESS(VR) < ess_threshold * N:
                cloning_step()  # git reset src_head + inherit init_strategy + init_commit_sha
        winner_label = argmax bincount(init_strategy among alive)
        return first_commit_sha of any alive walker with winner_label

Key fidelity points vs paper:
- relativize §2.2.3              ← from fractal_reward.py
- VR = R^alpha * D^beta §4.4     ← from fractal_reward.py
- pairwise stochastic clone §4.4 ← here
- decision = argmax bincount §4.6 ← here, on init_strategy_label
- ESS-adaptive cloning           ← Doucet et al. 2001, deep dive 05 §4.1

Critically, what travels with the clone is:
  (current_head)        — the worktree state (orchestrator runs git reset)
  (init_action_label)   — the strategy label for bincount
  (init_commit_sha)     — the SHA of the FIRST commit on that walker's lineage,
                           which is what gets cherry-picked into main when this
                           walker's strategy wins the swarm vote.

State persists in .fractal/sessions/<session_id>/state.json across CLI invocations,
so the orchestrator (slash command) can drive the loop step-by-step.

CLI:
  init  --task "..." --n 3 --m 3 [--ess-threshold 0.7]   → creates session
  record --session <id> --file walkers.json              → records walker outputs for current tick
  step  --session <id> [--seed S]                        → computes VR + ESS, returns clone_plan
  apply-clones --session <id>                            → updates state after orchestrator runs git reset
  decide --session <id>                                  → final argmax bincount → init_commit_sha
  status --session <id>                                  → dumps full state
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fractal_reward as fr  # noqa: E402


SESSIONS_ROOT = Path(".fractal/sessions")


def _session_path(session_id: str) -> Path:
    return SESSIONS_ROOT / session_id / "state.json"


def _load(session_id: str) -> dict:
    p = _session_path(session_id)
    if not p.exists():
        sys.stderr.write(f"error: session {session_id!r} not found at {p}\n")
        sys.exit(2)
    return json.loads(p.read_text())


def _save(session_id: str, state: dict) -> None:
    p = _session_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------- #
# init                                                                        #
# --------------------------------------------------------------------------- #

def cmd_init(args: argparse.Namespace) -> None:
    session_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    state = {
        "session_id": session_id,
        "task": args.task,
        "goal": args.goal or args.task,
        "n": args.n,
        "m": args.m,
        "alpha": args.alpha,
        "beta": args.beta,
        "ess_threshold": args.ess_threshold,
        "tick": 0,
        "walkers": [
            {
                "idx": i,
                "init_action_label": None,
                "init_action_desc": None,
                "init_commit_sha": None,
                "init_commit_message": None,
                "alive": True,
                "current_branch": None,
                "current_path": None,
                "current_head": None,
                "history": [],
            }
            for i in range(args.n)
        ],
        "decisions": [],
    }
    _save(session_id, state)
    print(
        json.dumps(
            {
                "session_id": session_id,
                "n": args.n,
                "m": args.m,
                "ess_threshold": args.ess_threshold,
                "alpha": args.alpha,
                "beta": args.beta,
            }
        )
    )


# --------------------------------------------------------------------------- #
# record                                                                      #
# --------------------------------------------------------------------------- #

def cmd_record(args: argparse.Namespace) -> None:
    state = _load(args.session)
    walker_jsons = json.loads(Path(args.file).read_text())
    if len(walker_jsons) != state["n"]:
        sys.stderr.write(
            f"error: expected {state['n']} walkers, got {len(walker_jsons)}\n"
        )
        sys.exit(2)

    tick = state["tick"]
    for i, wj in enumerate(walker_jsons):
        walker = state["walkers"][i]
        if not walker["alive"]:
            continue

        if tick == 0:
            walker["init_action_label"] = wj.get("approach_label")
            walker["init_action_desc"] = wj.get("approach_description")
            # init_commit_sha is the SHA of the FIRST commit the walker made.
            # At tick 0, walker_head IS the init_commit_sha (only one commit so far).
            walker["init_commit_sha"] = (
                wj.get("init_commit_sha") or wj.get("walker_head")
            )
            walker["init_commit_message"] = wj.get("init_commit_message", "")

        walker["current_branch"] = wj.get("worktree_branch")
        walker["current_path"] = wj.get("worktree_path")
        walker["current_head"] = wj.get("walker_head")

        if not wj.get("compile_ok", True):
            walker["alive"] = False

        breakdown = fr.composite_reward(wj)
        walker["history"].append(
            {
                "tick": tick,
                "walker_json": wj,
                "R": breakdown["R"],
                "breakdown": {k: v for k, v in breakdown.items() if k != "explanation"},
            }
        )

    _save(args.session, state)
    alive = sum(1 for w in state["walkers"] if w["alive"])
    print(json.dumps({"recorded": True, "tick": tick, "alive_count": alive}))


# --------------------------------------------------------------------------- #
# step (compute VR + ESS, decide whether to clone, emit clone_plan)           #
# --------------------------------------------------------------------------- #

def cmd_step(args: argparse.Namespace) -> None:
    state = _load(args.session)
    tick = state["tick"]
    n = state["n"]

    walkers_data = []
    for w in state["walkers"]:
        if not w["history"]:
            sys.stderr.write(f"error: walker {w['idx']} has no recordings\n")
            sys.exit(2)
        last = w["history"][-1]
        walkers_data.append(
            {
                "idx": w["idx"],
                "alive": w["alive"],
                "R": last["R"] if w["alive"] else 0.0,
                "files_changed": last["walker_json"].get("files_changed", []),
                "lines_added": last["walker_json"].get("lines_added", 0),
                "lines_deleted": last["walker_json"].get("lines_deleted", 0),
            }
        )

    rewards = [d["R"] for d in walkers_data]
    walker_state_for_vr = [
        {
            "files_changed": d["files_changed"],
            "lines_added": d["lines_added"],
            "lines_deleted": d["lines_deleted"],
        }
        for d in walkers_data
    ]

    if args.seed is not None:
        random.seed(args.seed + tick)

    vrs = fr.virtual_reward(
        walker_state_for_vr, rewards, alpha=state["alpha"], beta=state["beta"]
    )

    # ESS check (deep dive 05 §4.1, Doucet et al. 2001):
    # ESS = (sum VR)^2 / sum(VR^2). Range [1, N].
    # If sciame already diverse (high ESS), skip cloning to save sub-agent calls.
    sum_vr = sum(vrs)
    sum_vr_sq = sum(v * v for v in vrs)
    if sum_vr_sq > 1e-12:
        ess = (sum_vr * sum_vr) / sum_vr_sq
    else:
        ess = 0.0
    ess_threshold_abs = state["ess_threshold"] * n
    skip_cloning = ess > ess_threshold_abs

    clone_plan: list[dict] = []
    if not skip_cloning:
        for i in range(n):
            if not walkers_data[i]["alive"]:
                # Dead walker: clone from a random alive partner with prob 1
                alive_partners = [
                    j for j in range(n) if j != i and walkers_data[j]["alive"]
                ]
                if not alive_partners:
                    continue
                partner = random.choice(alive_partners)
                clone_plan.append(
                    _make_clone_entry(state, i, partner, "dead_revive", 1.0)
                )
                continue

            partners = [j for j in range(n) if j != i]
            if not partners:
                continue
            partner = random.choice(partners)

            vr_i, vr_p = vrs[i], vrs[partner]
            if vr_i <= 1e-8:
                prob = 1.0
            elif vr_p <= vr_i:
                prob = 0.0
            else:
                prob = min(1.0, (vr_p - vr_i) / vr_i)

            if random.random() < prob:
                clone_plan.append(
                    _make_clone_entry(state, i, partner, "vr_clone", prob)
                )

    state["decisions"].append(
        {
            "tick": tick,
            "vrs": vrs,
            "rewards": rewards,
            "ess": ess,
            "ess_threshold_abs": ess_threshold_abs,
            "cloning_skipped": skip_cloning,
            "clone_plan": clone_plan,
        }
    )
    _save(args.session, state)

    next_tick = tick + 1
    print(
        json.dumps(
            {
                "tick": tick,
                "next_tick": next_tick,
                "done": next_tick >= state["m"],
                "ess": ess,
                "ess_threshold_abs": ess_threshold_abs,
                "cloning_skipped": skip_cloning,
                "walkers": [
                    {
                        "idx": d["idx"],
                        "alive": d["alive"],
                        "R": d["R"],
                        "VR": vrs[d["idx"]],
                        "init_action_label": state["walkers"][d["idx"]][
                            "init_action_label"
                        ],
                    }
                    for d in walkers_data
                ],
                "clone_plan": clone_plan,
            },
            indent=2,
        )
    )


def _make_clone_entry(
    state: dict, dst_idx: int, src_idx: int, reason: str, prob: float
) -> dict:
    src = state["walkers"][src_idx]
    dst = state["walkers"][dst_idx]
    return {
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "reason": reason,
        "clone_prob": prob,
        "src_branch": src["current_branch"],
        "src_path": src["current_path"],
        "src_head": src["current_head"],
        "src_init_commit_sha": src["init_commit_sha"],
        "dst_branch": dst["current_branch"],
        "dst_path": dst["current_path"],
        "init_action_to_inherit": src["init_action_label"],
    }


# --------------------------------------------------------------------------- #
# apply-clones (mirror of orchestrator-side git reset into state)             #
# --------------------------------------------------------------------------- #

def cmd_apply_clones(args: argparse.Namespace) -> None:
    """Apply the clone plan from the last step to the in-memory state.

    The git reset --hard <src_head> is run by the orchestrator (slash command)
    BEFORE this is called. This function only mirrors the cloning into state:
    propagating init_action_label, init_commit_sha, and current_head from
    src to dst. All using OLD values (paper §4.4: snapshot before applying).
    """
    state = _load(args.session)
    if not state["decisions"]:
        sys.stderr.write("error: no step computed yet\n")
        sys.exit(2)

    last = state["decisions"][-1]
    old_init = [w["init_action_label"] for w in state["walkers"]]
    old_init_desc = [w["init_action_desc"] for w in state["walkers"]]
    old_init_sha = [w["init_commit_sha"] for w in state["walkers"]]
    old_init_msg = [w["init_commit_message"] for w in state["walkers"]]
    old_head = [w["current_head"] for w in state["walkers"]]

    for clone in last["clone_plan"]:
        dst = state["walkers"][clone["dst_idx"]]
        src_idx = clone["src_idx"]
        dst["init_action_label"] = old_init[src_idx]
        dst["init_action_desc"] = old_init_desc[src_idx]
        dst["init_commit_sha"] = old_init_sha[src_idx]
        dst["init_commit_message"] = old_init_msg[src_idx]
        dst["current_head"] = old_head[src_idx]
        dst["alive"] = True  # revived after reset

    state["tick"] += 1
    _save(args.session, state)
    print(
        json.dumps(
            {"applied": len(last["clone_plan"]), "new_tick": state["tick"]}
        )
    )


# --------------------------------------------------------------------------- #
# decide (paper §4.6 — argmax bincount over init_actions of alive walkers)    #
# --------------------------------------------------------------------------- #

def cmd_decide(args: argparse.Namespace) -> None:
    """Final decision after M ticks.

    paper §4.6: winner = argmax bincount(init_action) over alive walkers.
    Returns the init_commit_sha of a representative walker holding the
    winning label, so the orchestrator can cherry-pick it onto main.

    Tie handling:
      When multiple labels share the maximum bincount, ties are broken
      principled: among all walkers holding any tied label, pick the
      one with the highest current R. The output flags this with
      `is_tie: true` and lists `tied_labels` for the orchestrator to
      surface to the user (Sev-2 fix: avoid hidden insertion-order
      tiebreak on `Counter.most_common`).
    """
    state = _load(args.session)
    counts: Counter[str] = Counter()
    label_to_walkers: dict[str, list[dict]] = {}

    for w in state["walkers"]:
        if w["alive"] and w["init_action_label"]:
            counts[w["init_action_label"]] += 1
            label_to_walkers.setdefault(w["init_action_label"], []).append(w)

    if not counts:
        sys.stderr.write("error: no alive walkers with init_actions\n")
        sys.exit(2)

    def latest_R(w: dict) -> float:
        return w["history"][-1]["R"] if w["history"] else 0.0

    # Tie detection: find ALL labels at max count, not just the first one
    max_count = counts.most_common(1)[0][1]
    tied_labels = sorted(
        label for label, count in counts.items() if count == max_count
    )
    is_tie = len(tied_labels) > 1

    if is_tie:
        # Principled tie-break: highest R among ALL walkers with any tied label
        candidates = [
            w
            for w in state["walkers"]
            if w["alive"] and w["init_action_label"] in tied_labels
        ]
        representative = max(candidates, key=latest_R)
        winner_label = representative["init_action_label"]
        tie_break_method = "highest_R_among_tied"
    else:
        winner_label = tied_labels[0]
        representative = max(label_to_walkers[winner_label], key=latest_R)
        tie_break_method = None

    total = sum(counts.values())
    confidence = counts[winner_label] / total

    output = {
        "winner_label": winner_label,
        "winner_walker_idx": representative["idx"],
        "winner_branch": representative["current_branch"],
        "winner_path": representative["current_path"],
        "winner_head": representative["current_head"],
        "winner_init_commit_sha": representative["init_commit_sha"],
        "winner_init_commit_message": representative.get(
            "init_commit_message", ""
        ),
        "winner_latest_R": latest_R(representative),
        "vote_distribution": dict(counts),
        "confidence": confidence,
        "is_tie": is_tie,
        "tied_labels": tied_labels,
        "tie_break_method": tie_break_method,
        "alive_count": sum(1 for w in state["walkers"] if w["alive"]),
        "total_walkers": state["n"],
        "ticks_completed": state["tick"],
    }
    print(json.dumps(output, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    print(json.dumps(_load(args.session), indent=2))


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--task", required=True)
    p.add_argument("--goal", default=None,
                   help="Goal G (defaults to task)")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--ess-threshold", type=float, default=0.7,
                   help="Skip cloning if ESS > threshold * N (default 0.7)")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("record")
    p.add_argument("--session", required=True)
    p.add_argument("--file", required=True)
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("step")
    p.add_argument("--session", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_step)

    p = sub.add_parser("apply-clones")
    p.add_argument("--session", required=True)
    p.set_defaults(fn=cmd_apply_clones)

    p = sub.add_parser("decide")
    p.add_argument("--session", required=True)
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("status")
    p.add_argument("--session", required=True)
    p.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
