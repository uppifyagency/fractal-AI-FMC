#!/usr/bin/env python3
"""fractal_memory.py — Fractal Memory bank for the Fractal Coding Loop.

Implements the dataset-as-Fractal-Memory concept from:
  Hernández-Cerezo & Duran-Ballester, "Fractal Memory: Hybrids for Neural Networks" (2020 slides)

Each "memory unit" is a markdown file in `.fractal/memory/<timestamp>_<task>.md`.
Each memory has:
  - frontmatter: task, winner_idx, walkers (compact), reward_breakdown, visits, last_loss
  - body: human-readable summary

Key operations:
  append    — record a new (task, decision) episode
  recall    — Wigner-weighted sample of memories relevant to a query
  show      — list all memories with their stats
  prune     — remove memories with 0 walkers (= zero relevance, like the Slide doc)

The "loss" of a memory is interpreted as: "1 - confidence" of the past decision.
High-confidence past decisions = low loss = already understood, deprioritize.
Medium-confidence = high loss/avg = ideal to revisit.

Wigner reward formula (for batch sampling weights):
  R(x) = (π/2) · x · exp(-π/4 · x²)    where x = loss / avg_loss

Then debiased by visits: R / (1 + log(1 + visits))
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path


MEMORY_ROOT_NAME = ".fractal/memory"
INDEX_FILENAME = "_index.jsonl"  # one line per memory, lightweight


# =============================================================================
# Helpers
# =============================================================================

def ensure_memory_root(project_root: Path) -> Path:
    root = project_root / MEMORY_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def slugify(text: str, max_len: int = 50) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    s = s[:max_len].strip("-")
    return s or "untitled"


def find_project_root(cwd: Path = None) -> Path:
    """Find git root, fallback to cwd."""
    cwd = cwd or Path.cwd()
    p = cwd.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return cwd.resolve()


# =============================================================================
# Memory file format
# =============================================================================

def write_memory_file(
    memory_root: Path,
    task: str,
    winner_idx: int,
    walkers: list[dict],
    reward_breakdown: dict | None = None,
    confidence: float = 0.0,
) -> Path:
    """Write a markdown memory file with YAML frontmatter."""
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    slug = slugify(task)
    fname = f"{timestamp}_{slug}.md"
    path = memory_root / fname

    # Initial loss = 1 - confidence (high-confidence decisions = low loss)
    initial_loss = max(0.0, 1.0 - confidence)

    frontmatter = {
        "task": task,
        "timestamp": timestamp,
        "winner_idx": winner_idx,
        "winner_label": (
            walkers[winner_idx].get("approach_label", f"walker_{winner_idx}")
            if 0 <= winner_idx < len(walkers)
            else "cancelled"
        ),
        "confidence": round(confidence, 4),
        "loss": round(initial_loss, 4),
        "visits": 1,
        "n_walkers": len(walkers),
        "approaches": [
            w.get("approach_label", f"walker_{i}") for i, w in enumerate(walkers)
        ],
    }

    body_lines = [f"# {task}", ""]
    if winner_idx >= 0:
        body_lines.append(f"**Winner**: walker {winner_idx} — `{frontmatter['winner_label']}`")
    else:
        body_lines.append("**Outcome**: cancelled (no walker merged)")
    body_lines.append("")
    body_lines.append(f"**Confidence**: {confidence:.1%}")
    body_lines.append("")
    body_lines.append("## Walkers")
    body_lines.append("")
    for i, w in enumerate(walkers):
        marker = " ← winner" if i == winner_idx else ""
        body_lines.append(f"### {i}: {w.get('approach_label', 'unnamed')}{marker}")
        body_lines.append("")
        body_lines.append(f"- Files changed: {len(w.get('files_changed', []))}")
        body_lines.append(f"- Lines: +{w.get('lines_added', 0)} / -{w.get('lines_deleted', 0)}")
        if w.get("tests_run"):
            body_lines.append(
                f"- Tests: {w.get('tests_passed', 0)}/{w.get('tests_total', 0)}"
            )
        body_lines.append(f"- Lint warnings: {w.get('lint_warnings', 'n/a')}")
        body_lines.append(f"- Compile ok: {w.get('compile_ok', True)}")
        body_lines.append(f"- Summary: {w.get('summary', '')}")
        body_lines.append("")

    if reward_breakdown:
        body_lines.append("## Reward breakdown")
        body_lines.append("")
        body_lines.append("```json")
        body_lines.append(json.dumps(reward_breakdown, indent=2))
        body_lines.append("```")
        body_lines.append("")

    content = (
        "---\n"
        + "".join(f"{k}: {json.dumps(v)}\n" for k, v in frontmatter.items())
        + "---\n\n"
        + "\n".join(body_lines)
    )
    path.write_text(content, encoding="utf-8")
    return path


def parse_frontmatter(path: Path) -> dict:
    """Parse the YAML-ish frontmatter of a memory file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    block = text[4:end]
    out = {}
    for line in block.split("\n"):
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        try:
            out[key] = json.loads(val)
        except json.JSONDecodeError:
            out[key] = val
    return out


def update_frontmatter(path: Path, updates: dict) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return
    end = text.find("\n---\n", 4)
    if end < 0:
        return
    fm = parse_frontmatter(path)
    fm.update(updates)
    body = text[end + 5 :]
    new_fm = "---\n" + "".join(f"{k}: {json.dumps(v)}\n" for k, v in fm.items()) + "---\n"
    path.write_text(new_fm + body, encoding="utf-8")


# =============================================================================
# Wigner reward
# =============================================================================

def wigner_reward(loss: float, avg_loss: float) -> float:
    """R(x) = (π/2) · x · exp(-π/4 · x²)   with x = loss/avg_loss.
    Pattern: peak around x=1 (medium-difficulty memory), decays for too-easy or too-hard.
    """
    if avg_loss <= 0:
        return 0.0
    x = loss / avg_loss
    return (math.pi / 2.0) * x * math.exp(-(math.pi / 4.0) * x * x)


def memory_weight(loss: float, avg_loss: float, visits: int) -> float:
    """Wigner reward debiased by visit count."""
    R = wigner_reward(loss, avg_loss)
    return R / (1.0 + math.log1p(visits))


# =============================================================================
# Operations: append / recall / show / prune
# =============================================================================

def cmd_append(args):
    walkers = json.loads(args.walkers_json)
    project_root = find_project_root()
    memory_root = ensure_memory_root(project_root)
    path = write_memory_file(
        memory_root,
        task=args.task,
        winner_idx=args.winner_idx,
        walkers=walkers,
        confidence=args.confidence,
        reward_breakdown=json.loads(args.reward_json) if args.reward_json else None,
    )
    print(json.dumps({"saved": str(path)}, indent=2))


def cmd_recall(args):
    project_root = find_project_root()
    memory_root = ensure_memory_root(project_root)
    memories = sorted(memory_root.glob("*.md"))
    if not memories:
        print(json.dumps({"memories": [], "note": "memory bank is empty"}))
        return

    # Compute weights
    losses = []
    fms = []
    for p in memories:
        fm = parse_frontmatter(p)
        if not fm:
            continue
        fms.append((p, fm))
        losses.append(float(fm.get("loss", 0.5)))

    if not fms:
        print(json.dumps({"memories": []}))
        return

    avg_loss = sum(losses) / len(losses) or 1e-6
    weighted = []
    for (p, fm), loss in zip(fms, losses):
        visits = int(fm.get("visits", 1))
        w = memory_weight(loss, avg_loss, visits)
        weighted.append((w, p, fm))

    # Sort by weight desc, optionally filter by query keyword
    weighted.sort(key=lambda x: x[0], reverse=True)
    if args.query:
        q = args.query.lower()
        weighted = [
            (w, p, fm) for w, p, fm in weighted
            if q in str(fm.get("task", "")).lower()
        ]

    top = weighted[: args.top_k]
    out = {
        "avg_loss": avg_loss,
        "memories": [
            {
                "path": str(p.name),
                "weight": round(w, 4),
                "task": fm.get("task"),
                "winner_label": fm.get("winner_label"),
                "confidence": fm.get("confidence"),
                "loss": fm.get("loss"),
                "visits": fm.get("visits"),
            }
            for w, p, fm in top
        ],
    }
    print(json.dumps(out, indent=2))

    # Increment visit counter for recalled ones
    if args.mark_visited:
        for w, p, fm in top:
            fm["visits"] = int(fm.get("visits", 1)) + 1
            update_frontmatter(p, {"visits": fm["visits"]})


def cmd_show(args):
    project_root = find_project_root()
    memory_root = ensure_memory_root(project_root)
    memories = sorted(memory_root.glob("*.md"))
    if not memories:
        print(json.dumps({"memories": [], "note": "memory bank is empty"}))
        return
    out = []
    for p in memories:
        fm = parse_frontmatter(p)
        if not fm:
            continue
        out.append({
            "path": p.name,
            "timestamp": fm.get("timestamp"),
            "task": fm.get("task"),
            "winner_label": fm.get("winner_label"),
            "confidence": fm.get("confidence"),
            "loss": fm.get("loss"),
            "visits": fm.get("visits"),
        })
    print(json.dumps({"count": len(out), "memories": out}, indent=2))


def cmd_prune(args):
    project_root = find_project_root()
    memory_root = ensure_memory_root(project_root)
    memories = sorted(memory_root.glob("*.md"))
    deleted = []
    for p in memories:
        fm = parse_frontmatter(p)
        if not fm:
            continue
        # Loss too low AND many visits = "already learned, retire"
        loss = float(fm.get("loss", 0.5))
        visits = int(fm.get("visits", 1))
        if loss < 0.05 and visits > args.min_visits:
            p.unlink()
            deleted.append(p.name)
    print(json.dumps({"deleted": deleted}, indent=2))


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("append", help="Save a new decision episode")
    a.add_argument("--task", required=True)
    a.add_argument("--winner-idx", type=int, required=True)
    a.add_argument("--walkers-json", required=True, help="JSON array of walker dicts")
    a.add_argument("--confidence", type=float, default=0.0)
    a.add_argument("--reward-json", default=None)
    a.set_defaults(func=cmd_append)

    r = sub.add_parser("recall", help="Recall top-K memories Wigner-weighted")
    r.add_argument("--query", default=None)
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--mark-visited", action="store_true")
    r.set_defaults(func=cmd_recall)

    s = sub.add_parser("show", help="Show all memories")
    s.set_defaults(func=cmd_show)

    p = sub.add_parser("prune", help="Remove already-learned memories")
    p.add_argument("--min-visits", type=int, default=10)
    p.set_defaults(func=cmd_prune)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
