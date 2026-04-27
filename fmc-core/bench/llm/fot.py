"""Bet 2 — Fractal-of-Thought vs greedy vs self-consistency on small math benchmark.

Three methods compared on the same problems:

  1. greedy (temp=0.0, single sample)
  2. self-consistency (K independent samples at temp>0, majority vote)
  3. FoT (N walkers, M cloning cycles, then majority vote of survivors)

For (3), the cloning is FMC-style:
  - Walker state = a chain-of-thought (string)
  - Reward R(walker) = a heuristic confidence score (length-normalized log-prob
    proxy via tokenizer; concretely we use 1/(1+|cot|) scaled — proxy for
    "concise = confident")
  - Distance d(w_i, w_j) = embedding L2 distance from sentence-transformers
  - At each cycle, weak walkers clone to strong walkers (probabilistically),
    then regenerate from a slightly perturbed prompt.

Result: accuracy + token cost per method, across all problems.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from fmc.core import relativize, virtual_reward, clone_step, decide
from bench.runner import _hardware_info
import fmc as _fmc


# ---------------------------------------------------------------------------
# Lazy imports of heavy deps — only loaded if main() runs.
# ---------------------------------------------------------------------------

_MODEL = None
_TOK = None
_EMB = None


def _load_model(model_id: str = "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit"):
    global _MODEL, _TOK
    if _MODEL is None:
        from mlx_lm import load
        _MODEL, _TOK = load(model_id)
    return _MODEL, _TOK


def _load_embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB


# ---------------------------------------------------------------------------
# LLM helpers.
# ---------------------------------------------------------------------------

def _format_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + "\n\nThink step by step, then give the final numeric answer in the format 'Answer: <number>'."}],
        add_generation_prompt=True, tokenize=False,
    )


def _gen(model, tokenizer, prompt: str, max_tokens: int = 256, temp: float = 0.7, seed: int = 0) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temp)
    import mlx.core as mx
    mx.random.seed(seed)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)


_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _extract_answer(text: str) -> float | None:
    """Extract numeric answer. Look for 'Answer: X' first, else last number."""
    m = re.search(r"[Aa]nswer\s*[:=]\s*\$?(-?\d+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    nums = _NUMBER_RE.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Method 1: greedy.
# ---------------------------------------------------------------------------

def solve_greedy(question: str) -> Tuple[float | None, int]:
    model, tok = _load_model()
    prompt = _format_prompt(tok, question)
    out = _gen(model, tok, prompt, max_tokens=256, temp=0.0, seed=0)
    n_tokens = len(tok.encode(out))
    return _extract_answer(out), n_tokens


# ---------------------------------------------------------------------------
# Method 2: self-consistency (K independent samples at temp > 0).
# ---------------------------------------------------------------------------

def solve_self_consistency(question: str, K: int = 8, seed: int = 0) -> Tuple[float | None, int]:
    model, tok = _load_model()
    prompt = _format_prompt(tok, question)
    answers: List[float] = []
    total_tokens = 0
    for k in range(K):
        out = _gen(model, tok, prompt, max_tokens=256, temp=0.7, seed=seed * 100 + k)
        total_tokens += len(tok.encode(out))
        a = _extract_answer(out)
        if a is not None:
            answers.append(a)
    if not answers:
        return None, total_tokens
    # majority vote — round to handle small numeric noise.
    rounded = [round(a, 3) for a in answers]
    most = Counter(rounded).most_common(1)[0][0]
    return float(most), total_tokens


# ---------------------------------------------------------------------------
# Method 3: Fractal-of-Thought.
# ---------------------------------------------------------------------------

def solve_fot(
    question: str,
    N: int = 8,
    M: int = 3,
    alpha: float = 0.1,
    beta: float = 1.0,
    seed: int = 0,
) -> Tuple[float | None, int]:
    """FMC over chains-of-thought.

    At each of M cycles:
      1. Sample N CoTs (or regenerate weak ones from strong seeds).
      2. Compute virtual reward (R = -length proxy, D = embedding distance).
      3. Clone weak to strong via clone_step.

    Final answer = majority vote of survivors at end of cycle M.
    """
    model, tok = _load_model()
    embedder = _load_embedder()
    prompt = _format_prompt(tok, question)
    rng = np.random.default_rng(seed)

    cots: List[str] = []
    total_tokens = 0
    # Cycle 0: initial sample of N CoTs at temperature 0.7.
    for i in range(N):
        out = _gen(model, tok, prompt, max_tokens=256, temp=0.7, seed=seed * 1000 + i)
        cots.append(out)
        total_tokens += len(tok.encode(out))

    for cycle in range(M):
        # Reward: shorter answers with valid extracted number get higher score.
        rewards = np.zeros(N)
        for i, cot in enumerate(cots):
            a = _extract_answer(cot)
            valid = a is not None
            length_pen = -len(cot) / 200.0  # prefer concise
            rewards[i] = (1.0 if valid else -1.0) + length_pen

        # Distance: pairwise embedding L2.
        if beta > 0:
            embs = embedder.encode(cots, show_progress_bar=False, convert_to_numpy=True)
        else:
            embs = np.zeros((N, 1))
        # Random partner.
        partners = rng.permutation(N)
        for i in range(N):
            if partners[i] == i:
                partners[i] = (i + 1) % N
        vr = virtual_reward(rewards, embs.astype(np.float64), partners, alpha=alpha, beta=beta)

        # Clone step.
        idx = clone_step(vr, rng)
        new_cots: List[str] = []
        for i, k in enumerate(idx):
            if k == i:
                new_cots.append(cots[i])
            else:
                # Regenerate, seeded by strong walker's final tokens as hint.
                seed_text = cots[k]
                # Take last 50 chars of strong CoT as continuation seed.
                hint = seed_text[-100:] if len(seed_text) > 100 else seed_text
                regen_prompt = prompt + hint[:80]  # bias toward strong walker
                out = _gen(model, tok, regen_prompt, max_tokens=200, temp=0.7,
                           seed=seed * 10000 + cycle * 100 + i)
                new_cots.append(out)
                total_tokens += len(tok.encode(out))
        cots = new_cots

    # Final answer: majority vote of survivors.
    answers: List[float] = []
    for c in cots:
        a = _extract_answer(c)
        if a is not None:
            answers.append(round(a, 3))
    if not answers:
        return None, total_tokens
    most = Counter(answers).most_common(1)[0][0]
    return float(most), total_tokens


# ---------------------------------------------------------------------------
# Benchmark driver.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=8, help="Self-consistency samples")
    parser.add_argument("--fot-N", type=int, default=8, help="FoT walker count")
    parser.add_argument("--fot-M", type=int, default=2, help="FoT cycles (cloning rounds)")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per method per problem")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems (None = all)")
    parser.add_argument(
        "--tier", choices=["easy", "hard", "all"], default="all",
        help="Which problem set to use",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    from bench.llm.problems import EASY, HARD
    if args.tier == "easy":
        problems = EASY
    elif args.tier == "hard":
        problems = HARD
    else:
        problems = EASY + HARD
    if args.limit:
        problems = problems[:args.limit]

    output = args.output or str(
        Path(__file__).parent.parent / "results" / f"fot_llm_{args.tier}.jsonl"
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    methods = ["greedy", "self_consistency", "fot"]
    results = []  # one record per (method, problem, seed)

    print(f"Bet 2 — Fractal-of-Thought benchmark")
    print(f"Model: LFM2.5-1.2B-Instruct-MLX-4bit (Apple MLX)")
    print(f"Problems: {len(problems)}, seeds: {args.seeds}")
    print(f"Self-consistency K={args.K}, FoT N={args.fot_N} M={args.fot_M}\n")

    method_correct = {m: 0 for m in methods}
    method_tokens = {m: 0 for m in methods}
    method_attempted = {m: 0 for m in methods}

    for p_idx, (q, gt) in enumerate(problems):
        print(f"[{p_idx+1}/{len(problems)}] {q}")
        print(f"  Ground truth: {gt}")
        for seed in range(args.seeds):
            for method in methods:
                t0 = time.time()
                if method == "greedy":
                    pred, tokens = solve_greedy(q)
                elif method == "self_consistency":
                    pred, tokens = solve_self_consistency(q, K=args.K, seed=seed)
                elif method == "fot":
                    pred, tokens = solve_fot(q, N=args.fot_N, M=args.fot_M, seed=seed)
                else:
                    raise ValueError(method)
                dt = time.time() - t0
                correct = pred is not None and abs(pred - gt) < 0.01
                method_attempted[method] += 1
                if correct:
                    method_correct[method] += 1
                method_tokens[method] += tokens
                results.append({
                    "method": method,
                    "problem_idx": p_idx,
                    "question": q,
                    "ground_truth": gt,
                    "seed": seed,
                    "predicted": pred,
                    "correct": correct,
                    "tokens": tokens,
                    "duration_s": dt,
                })
                tag = "✓" if correct else "✗"
                print(f"    seed={seed} {method:<18} pred={pred} {tag}  tokens={tokens}  {dt:.1f}s")
        print()

    # Summary.
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for m in methods:
        att = method_attempted[m]
        acc = method_correct[m] / att if att > 0 else 0
        avg_tok = method_tokens[m] / att if att > 0 else 0
        print(f"  {m:<20} accuracy = {acc:.1%} ({method_correct[m]}/{att})   avg tokens/problem = {avg_tok:.0f}")

    summary = {
        "benchmark": "fot_llm",
        "model": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
        "problems": len(problems),
        "seeds_per_method": args.seeds,
        "methods": {
            m: {
                "correct": method_correct[m],
                "attempted": method_attempted[m],
                "accuracy": method_correct[m] / method_attempted[m] if method_attempted[m] else 0,
                "avg_tokens_per_problem": method_tokens[m] / method_attempted[m] if method_attempted[m] else 0,
            }
            for m in methods
        },
        "params": {
            "self_consistency_K": args.K,
            "fot_N": args.fot_N,
            "fot_M": args.fot_M,
        },
        "hardware": _hardware_info(),
        "fmc_core_version": _fmc.__version__,
    }
    summary_path = Path(output).parent / f"fot_llm_{args.tier}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {summary_path}")

    with open(output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Per-trial results: {output}")


if __name__ == "__main__":
    main()
