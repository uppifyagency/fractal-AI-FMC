"""Conjecture E / E1-LLM — NVIDIA NIM client + LLM world-model generation.

E1_LLM_DESIGN.md section 4.1, Route B, "code form" (Tang et al. 2024, Code World
Model): the LLM is given a natural-language description of the gridworld RULES
and writes the code of the transition function. FMC then plans on that code.
The LLM is the world-model organ; nothing else of FMC changes.

Security: the LLM-written code is (1) AST-allowlisted — no imports, no dunders,
no I/O builtins — then (2) exec'd in a namespace whose __builtins__ is a small
safe subset, then (3) validated on a crash battery before use. A generation that
fails any gate is rejected and retried (a non-runnable function is not a
world-model); all attempts are logged.

The API key is read from the macOS Keychain (service `fractalai-nvidia-api`);
it is never written to disk or committed.

Not a standalone run — imported by e1_llm_run.py.
"""

from __future__ import annotations

import ast
import subprocess
import textwrap

# --- world-model contract ----------------------------------------------------
# The transition the LLM must implement. Signature is fixed; the body is the LLM's.
TRANSITION_SIGNATURE = "transition(r, c, done, action, grid, H, W)"

# Natural-language description of the gridworld rules (E1_LLM_DESIGN section 4.1:
# the rules, fairly and completely — no hint beyond what the rules themselves say).
WORLD_DESCRIPTION = textwrap.dedent("""\
    I have a 2D grid world and need its transition function implemented in Python.

    The world:
    - A rectangular grid, H rows by W columns. Each cell holds one code:
      0 = free ground, 1 = lava, 2 = the goal.
    - A single walker occupies one cell, at row r, column c.
    - Lava cells and the goal cell are *absorbing*: the moment the walker enters
      a lava cell or the goal cell it is stuck there permanently — from then on
      every action leaves it exactly where it is.
    - The walker carries a boolean `done`, which is True exactly when it is
      currently on an absorbing cell (lava or goal).

    A step applies one action:
      0 = move up    (row - 1)
      1 = move down  (row + 1)
      2 = move left  (column - 1)
      3 = move right (column + 1)
      4 = stay       (do not move)
    If a move would leave the grid, the walker does not move — it stays in its
    current cell. The 'stay' action never moves the walker.

    Write ONE pure Python function with EXACTLY this signature:

        def transition(r, c, done, action, grid, H, W):
            # returns a tuple (nr, nc, ndone)

    `grid` is an H-by-W array of cell codes (index it as grid[row][col], values
    0/1/2). `r, c` is the current cell; `done` is the current absorbing flag;
    `action` is an integer 0..4. Return the next cell (nr, nc) and the next
    absorbing flag ndone.

    Requirements:
    - Pure function: no imports, no I/O, no global state, no classes, no prints.
    - Deterministic.
    - Output ONLY the function inside a single ```python code block.
    """)

_SYSTEM = ("You are an expert Python programmer. You write correct, minimal, "
           "pure functions and follow the requested signature and output format "
           "exactly.")

# AST allowlist: builtin names the transition code may reference.
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "int": int, "bool": bool,
    "len": len, "range": range, "tuple": tuple, "enumerate": enumerate,
    "True": True, "False": False, "None": None,
}


# === API key + chat ==========================================================

def api_key() -> str:
    """NVIDIA NIM key from the macOS Keychain. Never persisted, never logged."""
    out = subprocess.run(
        ["security", "find-generic-password", "-s", "fractalai-nvidia-api", "-w"],
        capture_output=True, text=True, check=True)
    key = out.stdout.strip()
    if not key.startswith("nvapi-"):
        raise RuntimeError("Keychain returned no usable NVIDIA key")
    return key


def chat(model: str, messages: list[dict], temperature: float = 0.2,
         max_tokens: int = 1600) -> str:
    """One chat completion against NVIDIA NIM (OpenAI-compatible endpoint)."""
    from openai import OpenAI
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                    api_key=api_key())
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content or ""


# === code extraction + safety ================================================

def extract_code(text: str) -> str:
    """Pull the first ```python ... ``` block (or the first ``` block)."""
    for fence in ("```python", "```py", "```"):
        if fence in text:
            after = text.split(fence, 1)[1]
            return after.split("```", 1)[0].strip()
    return text.strip()


def safe_check(src: str) -> None:
    """AST allowlist. Raises ValueError on anything outside a pure function."""
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or not funcs or funcs[0].name != "transition":
        raise ValueError("must be exactly one top-level def transition(...)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal,
                             ast.With, ast.AsyncFunctionDef, ast.Lambda)):
            # lambda/with/class are not needed for a pure transition
            if not isinstance(node, ast.Lambda):
                raise ValueError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name access is not allowed")


# Crash/well-formedness battery: a 4x4 grid with free/lava/goal, every (cell,
# action, done) combination. A valid world-model returns a clean (int,int,bool).
def _validate(fn) -> None:
    import numpy as np
    grid = np.array([[0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 2],
                     [1, 0, 0, 0]], dtype=np.int64)
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            for done in (False, True):
                for a in range(5):
                    out = fn(r, c, done, a, grid, H, W)
                    if (not isinstance(out, tuple) or len(out) != 3):
                        raise ValueError(f"transition({r},{c},{done},{a}) "
                                         f"did not return a 3-tuple")
                    nr, nc, nd = out
                    int(nr), int(nc), bool(nd)            # must be coercible
                    if not (0 <= int(nr) < H and 0 <= int(nc) < W):
                        raise ValueError(f"transition returned off-grid "
                                         f"cell ({nr},{nc})")


def compile_transition(src: str):
    """AST-check, exec in a sandboxed namespace, validate. Returns the callable."""
    safe_check(src)
    ns: dict = {"__builtins__": _SAFE_BUILTINS}
    exec(compile(src, "<llm_world_model>", "exec"), ns)   # noqa: S102 (sandboxed)
    fn = ns.get("transition")
    if not callable(fn):
        raise ValueError("no callable `transition` produced")
    _validate(fn)
    return fn


# === generation loop =========================================================

def generate_world_model(model: str, description: str = WORLD_DESCRIPTION,
                         temperature: float = 0.2, max_attempts: int = 3):
    """Ask `model` to write the gridworld transition from `description`; retry on
    a failed gate, feeding the error back. Returns (transition_fn, source,
    transcript). `description` lets a caller vary prompt fidelity (E1-LLM-curve)."""
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": description}]
    transcript = []
    for attempt in range(1, max_attempts + 1):
        raw = chat(model, messages, temperature=temperature)
        src = extract_code(raw)
        rec = {"attempt": attempt, "raw": raw, "source": src}
        try:
            fn = compile_transition(src)
            rec["status"] = "OK"
            transcript.append(rec)
            return fn, src, transcript
        except Exception as e:                            # noqa: BLE001
            rec["status"] = f"REJECTED: {e}"
            transcript.append(rec)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             f"That was rejected: {e}. Fix it and resend ONLY "
                             f"the function in one ```python block."})
    raise RuntimeError(f"{model} produced no valid world-model in "
                       f"{max_attempts} attempts")
