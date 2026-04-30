# Sanity test plan

After baseline lock confirms ~29% Crafter, we run ONE mutation cycle to
verify the loop discipline works:

1. Edit fmc_mutable.py: change `action_repeat=1` to `action_repeat=2`
   (mild change, may help or hurt — empirical question)
2. Commit: `git commit -am "sanity: action_repeat=2"`
3. Run: `python evaluate.py --description "sanity test action_repeat=2"`
4. Read crafter_pct from log
5. Verify auto-status fires correctly:
   - If crafter > 30.27 (baseline +1pp): status=keep
   - Else: status=discard
6. If discard: `git reset --hard HEAD~1`
7. Verify fmc_mutable.py reverted
8. Verify results.tsv has 2 rows

This confirms:
- Driver runs end-to-end
- Wall budget triggers correctly at 20 min
- Auto-status decision works
- Git reset --hard discipline works
- results.tsv accumulates correctly
