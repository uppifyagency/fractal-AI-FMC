#!/usr/bin/env bash
# e2e_test.sh — End-to-end test of fractal-coding-loop plugin components.
#
# Tests:
#   1. plugin.json is valid JSON
#   2. all .md files have valid YAML frontmatter
#   3. fractal_reward.py produces valid output for sample walkers
#   4. fractal_memory.py append/recall/show round-trip works
#   5. Wigner reward formula gives expected shape

set -e
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$PLUGIN_DIR/scripts"
TMP_DIR=$(mktemp -d -t fractal-e2e-XXXXXX)
trap "rm -rf $TMP_DIR" EXIT

PASS=0
FAIL=0

assert() {
  local name="$1"; shift
  if "$@" > "$TMP_DIR/last.log" 2>&1; then
    echo "  ✓ $name"
    PASS=$((PASS+1))
  else
    echo "  ✗ $name"
    echo "    --- log ---"
    cat "$TMP_DIR/last.log" | sed 's/^/      /'
    echo "    -----------"
    FAIL=$((FAIL+1))
  fi
}

# ============================================================================
echo "=== TEST 1: plugin manifest ==="
assert "plugin.json is valid JSON" \
  python3 -c "import json; json.load(open('$PLUGIN_DIR/.claude-plugin/plugin.json'))"

# ============================================================================
echo ""
echo "=== TEST 2: markdown frontmatter ==="
for f in "$PLUGIN_DIR"/commands/*.md "$PLUGIN_DIR"/agents/*.md; do
  if [ -f "$f" ]; then
    name="$(basename "$f")"
    assert "$name has frontmatter" bash -c "head -1 '$f' | grep -q '^---$'"
  fi
done

# ============================================================================
echo ""
echo "=== TEST 3: fractal_reward.py ==="

cat > "$TMP_DIR/walkers.json" <<'EOF'
[
  {
    "approach_label": "in-place",
    "files_changed": ["src/auth.py"],
    "lines_added": 30, "lines_deleted": 15,
    "tests_run": true, "tests_passed": 12, "tests_total": 12,
    "lint_warnings": 0, "compile_ok": true,
    "summary": "Replaced session with JWT in-place"
  },
  {
    "approach_label": "extract-module",
    "files_changed": ["src/auth.py", "src/auth/jwt.py", "src/auth/__init__.py"],
    "lines_added": 80, "lines_deleted": 30,
    "tests_run": true, "tests_passed": 12, "tests_total": 12,
    "lint_warnings": 1, "compile_ok": true,
    "summary": "Extracted auth into module"
  },
  {
    "approach_label": "test-first",
    "files_changed": ["src/auth.py", "tests/test_jwt.py"],
    "lines_added": 50, "lines_deleted": 10,
    "tests_run": true, "tests_passed": 13, "tests_total": 13,
    "lint_warnings": 0, "compile_ok": true,
    "summary": "Wrote JWT tests then implemented"
  }
]
EOF

assert "reward script accepts JSON file" \
  python3 "$SCRIPTS/fractal_reward.py" --walker-jsons-file "$TMP_DIR/walkers.json" --seed 42

# Verify output structure
RESULT=$(python3 "$SCRIPTS/fractal_reward.py" --walker-jsons-file "$TMP_DIR/walkers.json" --seed 42)
assert "reward output has winner_idx" bash -c "echo '$RESULT' | python3 -c 'import sys,json; d=json.load(sys.stdin); assert \"winner_idx\" in d'"
assert "reward output has confidence in [0,1]" bash -c "echo '$RESULT' | python3 -c 'import sys,json; d=json.load(sys.stdin); c=d[\"confidence\"]; assert 0 <= c <= 1, c'"
assert "reward output has 3 walkers" bash -c "echo '$RESULT' | python3 -c 'import sys,json; d=json.load(sys.stdin); assert len(d[\"walkers\"])==3'"

# Verify multiplicative formula: dead walker (compile_ok=false) gets R=0
cat > "$TMP_DIR/dead.json" <<'EOF'
[
  {"approach_label":"alive","compile_ok":true,"tests_run":true,"tests_passed":1,"tests_total":1,"lint_warnings":0,"lines_added":10,"lines_deleted":0,"files_changed":["a.py"]},
  {"approach_label":"dead","compile_ok":false,"tests_run":false,"lint_warnings":-1,"lines_added":50,"lines_deleted":50,"files_changed":["b.py"]}
]
EOF
assert "dead walker has R=0 (multiplicative hard constraint)" bash -c "
  R=\$(python3 '$SCRIPTS/fractal_reward.py' --walker-jsons-file '$TMP_DIR/dead.json' --seed 1 | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[\"walkers\"][1][\"R\"])')
  python3 -c \"assert float('\$R') == 0.0, '\$R should be 0'\"
"

# ============================================================================
echo ""
echo "=== TEST 4: fractal_memory.py round-trip ==="

# Use a temp project root
mkdir -p "$TMP_DIR/proj"
cd "$TMP_DIR/proj"
git init -q 2>&1 | head -1 >/dev/null || true
echo "x" > README.md
git add . 2>&1 | head -1 >/dev/null || true
git -c user.email=test@test -c user.name=test commit -m init -q 2>&1 | head -1 >/dev/null || true

WALKERS_JSON='[{"approach_label":"a","files_changed":["x"]},{"approach_label":"b","files_changed":["y"]},{"approach_label":"c","files_changed":["z"]}]'

assert "memory append" \
  python3 "$SCRIPTS/fractal_memory.py" append \
    --task "test refactor" --winner-idx 1 --confidence 0.67 \
    --walkers-json "$WALKERS_JSON"

assert "memory show shows 1 entry" bash -c "
  count=\$(python3 '$SCRIPTS/fractal_memory.py' show | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[\"count\"])')
  python3 -c \"assert int('\$count') == 1, 'expected 1 got \$count'\"
"

# Add another memory
assert "memory append #2" \
  python3 "$SCRIPTS/fractal_memory.py" append \
    --task "another task" --winner-idx 0 --confidence 0.85 \
    --walkers-json "$WALKERS_JSON"

assert "memory recall returns top-K Wigner-weighted" \
  python3 "$SCRIPTS/fractal_memory.py" recall --top-k 5

# ============================================================================
echo ""
echo "=== TEST 5b: fractal_loop.py decide — tie detection (Sev-2 fix) ==="

# Setup an isolated workdir so we don't pollute cwd's .fractal/
TIE_DIR="$TMP_DIR/tie_test"
mkdir -p "$TIE_DIR"

cat > "$TIE_DIR/walkers_tied.json" <<'EOF'
[
  {"approach_label":"alpha","worktree_path":"/x/0","worktree_branch":"w0","walker_head":"sha0","init_commit_sha":"sha0","init_commit_message":"alpha","files_changed":["a.py"],"lines_added":10,"lines_deleted":0,"tests_run":true,"tests_passed":1,"tests_total":1,"lint_warnings":0,"compile_ok":true,"summary":"alpha"},
  {"approach_label":"bravo","worktree_path":"/x/1","worktree_branch":"w1","walker_head":"sha1","init_commit_sha":"sha1","init_commit_message":"bravo","files_changed":["b.py"],"lines_added":10,"lines_deleted":0,"tests_run":true,"tests_passed":1,"tests_total":1,"lint_warnings":0,"compile_ok":true,"summary":"bravo"},
  {"approach_label":"charlie","worktree_path":"/x/2","worktree_branch":"w2","walker_head":"sha2","init_commit_sha":"sha2","init_commit_message":"charlie","files_changed":["c.py"],"lines_added":10,"lines_deleted":0,"tests_run":true,"tests_passed":1,"tests_total":1,"lint_warnings":0,"compile_ok":true,"summary":"charlie"}
]
EOF

cd "$TIE_DIR"
SESS=$(python3 "$SCRIPTS/fractal_loop.py" init --task t --n 3 --m 1 \
    | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['session_id'])")
python3 "$SCRIPTS/fractal_loop.py" record --session "$SESS" --file walkers_tied.json > /dev/null
DECIDE_OUT=$(python3 "$SCRIPTS/fractal_loop.py" decide --session "$SESS")
cd - > /dev/null

assert "decide reports is_tie=true on uniform bincount" bash -c "
echo '$DECIDE_OUT' | python3 -c '
import json, sys
d = json.loads(sys.stdin.read())
assert d[\"is_tie\"] is True, f\"expected is_tie=true, got {d}\"
'
"

assert "decide lists all 3 tied_labels" bash -c "
echo '$DECIDE_OUT' | python3 -c '
import json, sys
d = json.loads(sys.stdin.read())
v = sorted(d[\"tied_labels\"])
assert v == [\"alpha\", \"bravo\", \"charlie\"], \"got \" + repr(v)
'
"

assert "decide reports tie_break_method=highest_R_among_tied" bash -c "
echo '$DECIDE_OUT' | python3 -c '
import json, sys
d = json.loads(sys.stdin.read())
v = d[\"tie_break_method\"]
assert v == \"highest_R_among_tied\", \"got \" + repr(v)
'
"

assert "decide confidence is 1/3 when 3-way tied" bash -c "
echo '$DECIDE_OUT' | python3 -c '
import json, sys
d = json.loads(sys.stdin.read())
c = d[\"confidence\"]
assert abs(c - 1/3) < 1e-6, f\"expected confidence=0.333, got {c}\"
'
"

# Sanity: when NOT tied (one walker has higher R), is_tie=false
cat > "$TIE_DIR/walkers_clear.json" <<'EOF'
[
  {"approach_label":"alpha","worktree_path":"/x/0","worktree_branch":"w0","walker_head":"sha0","init_commit_sha":"sha0","init_commit_message":"alpha","files_changed":["a.py"],"lines_added":10,"lines_deleted":0,"tests_run":true,"tests_passed":1,"tests_total":1,"lint_warnings":0,"compile_ok":true,"summary":"alpha"},
  {"approach_label":"alpha","worktree_path":"/x/1","worktree_branch":"w1","walker_head":"sha1","init_commit_sha":"sha1","init_commit_message":"alpha-2","files_changed":["b.py"],"lines_added":10,"lines_deleted":0,"tests_run":true,"tests_passed":1,"tests_total":1,"lint_warnings":0,"compile_ok":true,"summary":"alpha-2"},
  {"approach_label":"bravo","worktree_path":"/x/2","worktree_branch":"w2","walker_head":"sha2","init_commit_sha":"sha2","init_commit_message":"bravo","files_changed":["c.py"],"lines_added":10,"lines_deleted":0,"tests_run":true,"tests_passed":1,"tests_total":1,"lint_warnings":0,"compile_ok":true,"summary":"bravo"}
]
EOF

cd "$TIE_DIR"
SESS2=$(python3 "$SCRIPTS/fractal_loop.py" init --task t --n 3 --m 1 \
    | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['session_id'])")
python3 "$SCRIPTS/fractal_loop.py" record --session "$SESS2" --file walkers_clear.json > /dev/null
DECIDE_OUT2=$(python3 "$SCRIPTS/fractal_loop.py" decide --session "$SESS2")
cd - > /dev/null

assert "decide reports is_tie=false when one label has clear majority" bash -c "
echo '$DECIDE_OUT2' | python3 -c '
import json, sys
d = json.loads(sys.stdin.read())
is_tie_v = d[\"is_tie\"]
winner_v = d[\"winner_label\"]
tied_v = d[\"tied_labels\"]
assert is_tie_v is False, \"expected is_tie=false, got \" + repr(is_tie_v)
assert winner_v == \"alpha\", \"expected alpha, got \" + repr(winner_v)
assert tied_v == [\"alpha\"], \"got \" + repr(tied_v)
'
"

# ============================================================================
echo ""
echo "=== TEST 5: Wigner reward shape ==="

assert "Wigner peaks near x=1" python3 -c "
import sys
sys.path.insert(0, '$SCRIPTS')
from fractal_memory import wigner_reward
# Sample at x=0.5, x=1.0, x=1.5, x=3.0
values = [wigner_reward(x, 1.0) for x in [0.5, 1.0, 1.5, 3.0]]
# x=1.0 should be max (or near-max)
assert values[1] >= max(values[0], values[2], values[3]) * 0.95, f'Wigner not peaked near x=1: {values}'
print('Wigner peak verification:', values)
"

# ============================================================================
echo ""
echo "=== SUMMARY ==="
echo "PASSED: $PASS"
echo "FAILED: $FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "OK ✗ — $FAIL failure(s)"
  exit 1
fi
echo "OK ✓ — all components functional"
