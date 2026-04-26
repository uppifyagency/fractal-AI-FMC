"""sweep_v4v5.py — confronta inv_a20 (best v3) vs varianti v4 delta-proximity e v5 memory.

Test su 5 seed:
  - inv_a20         : v3 best (intrinsic 2.0)
  - v4_delta_p10    : inv_a20 + proximity delta α=1.0
  - v4_delta_p05    : inv_a20 + proximity delta α=0.5
  - v5_mem05        : inv_a20 + Fractal Memory mem_weight=0.5
  - v5_mem05_p05    : inv_a20 + delta-proximity + memory (full stack)
"""
from __future__ import annotations
import sys, json, time, math
from collections import Counter
sys.path.insert(0, '/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/work/05_craftax/scripts')

from fmc_craftax_v4 import run_episode as run_v4, FMCConfig as Cfg4
from fmc_craftax_v5 import run_episode as run_v5, FMCConfig as Cfg5, FractalMemory
from craftax.craftax_env import make_craftax_env_from_name

CLASSIC = ['collect_coal','collect_diamond','collect_drink','collect_iron','collect_sapling','collect_stone','collect_wood','defeat_skeleton','defeat_zombie','eat_cow','eat_plant','make_iron_pickaxe','make_iron_sword','make_stone_pickaxe','make_stone_sword','make_wood_pickaxe','make_wood_sword','place_furnace','place_plant','place_stone','wake_up','place_table']

def cscore(sr, names):
    return math.exp(sum(math.log(1+sr.get(n,0.0)*100) for n in names)/len(names))-1


def aggregate(per_seed: list[dict]) -> dict:
    counter = Counter()
    for r in per_seed:
        for a in r['achievements_list']: counter[a] += 1
    n = len(per_seed)
    sr = {a: c/n for a, c in counter.items()}
    achs = [r['achievements_unlocked'] for r in per_seed]
    mean_a = sum(achs)/n
    std_a = (sum((a-mean_a)**2 for a in achs)/n)**0.5
    se = std_a / (n**0.5)
    return {
        'achievements_per_seed': achs,
        'achievement_success_rates': sr,
        'n_unique_achievements': len(sr),
        'crafter_score_pct': round(cscore(sr, CLASSIC), 4),
        'mean_achievements': round(mean_a, 2),
        'std_achievements': round(std_a, 2),
        'ci95_achievements': round(1.96 * se, 2),
    }


def run_v4_config(label, cfg, seeds, max_steps):
    print(f'\n[{label}] {cfg}', file=sys.stderr)
    t0 = time.time()
    rs = []
    for s in seeds:
        r = run_v4(s, cfg, max_steps, False, 'Craftax-Classic-Symbolic-v1')
        rs.append(r)
        print(f'  seed={s} ach={r["achievements_unlocked"]} reward={r["reward"]:.1f} '
              f'wall={r["wall_time_s"]:.1f}s', file=sys.stderr)
    out = aggregate(rs)
    out['wall_total_s'] = round(time.time()-t0, 1)
    out['version'] = 'v4'
    return out


def run_v5_config(label, cfg, seeds, max_steps):
    print(f'\n[{label}] {cfg}', file=sys.stderr)
    t0 = time.time()
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
    n_actions = env.action_space(env.default_params).n
    memory = FractalMemory(n_actions=n_actions)
    rs = []
    for s in seeds:
        r = run_v5(s, cfg, memory, max_steps, False, 'Craftax-Classic-Symbolic-v1')
        rs.append(r)
        print(f'  seed={s} ach={r["achievements_unlocked"]} reward={r["reward"]:.1f} '
              f'mem_hit={r["mem_hit_rate"]:.2f} wall={r["wall_time_s"]:.1f}s', file=sys.stderr)
    out = aggregate(rs)
    out['wall_total_s'] = round(time.time()-t0, 1)
    out['version'] = 'v5'
    out['memory_stats'] = memory.stats()
    return out


def main():
    # Best base config from run_003 10-seed: inv_a05 → 19.27% Crafter
    seeds = list(range(42, 47))  # 5 seeds for fast iteration
    max_steps = 500
    base4 = dict(n_walkers=64, time_horizon=20, alpha=1.0, beta=1.0, action_repeat=1)
    base5 = dict(n_walkers=64, time_horizon=20, alpha=1.0, beta=1.0, action_repeat=1)
    INV = 0.5  # winner alpha

    results = {}

    # Reference: v3 best (intrinsic only)
    results['inv_a05_ref'] = run_v4_config(
        'inv_a05_ref',
        Cfg4(**base4, intrinsic_inv_alpha=INV, proximity_alpha=0.0),
        seeds, max_steps,
    )

    # v4: inv + delta proximity at different α
    results['v4_p02_delta'] = run_v4_config(
        'v4_p02_delta',
        Cfg4(**base4, intrinsic_inv_alpha=INV, proximity_alpha=0.2, proximity_mode='delta'),
        seeds, max_steps,
    )
    results['v4_p10_delta'] = run_v4_config(
        'v4_p10_delta',
        Cfg4(**base4, intrinsic_inv_alpha=INV, proximity_alpha=1.0, proximity_mode='delta'),
        seeds, max_steps,
    )

    # v5: inv + memory at different mem_weight
    results['v5_mem03'] = run_v5_config(
        'v5_mem03',
        Cfg5(**base5, intrinsic_inv_alpha=INV, proximity_alpha=0.0, mem_weight=0.3),
        seeds, max_steps,
    )
    results['v5_mem07'] = run_v5_config(
        'v5_mem07',
        Cfg5(**base5, intrinsic_inv_alpha=INV, proximity_alpha=0.0, mem_weight=0.7),
        seeds, max_steps,
    )

    # v5: full stack inv + delta-proximity + memory
    results['v5_full'] = run_v5_config(
        'v5_full',
        Cfg5(**base5, intrinsic_inv_alpha=INV, proximity_alpha=0.5, mem_weight=0.5),
        seeds, max_steps,
    )

    ranking = sorted(
        [(l, r['crafter_score_pct'], r['mean_achievements'], r['ci95_achievements'],
          r['n_unique_achievements'], r['wall_total_s'])
         for l, r in results.items()],
        key=lambda x: -x[1],
    )
    print('\n=== RANKING (5 seeds, N=64 M=20, intrinsic α=2.0) ===', file=sys.stderr)
    for l, s, m, ci, u, w in ranking:
        print(f'  {l:18s}  score={s:5.2f}%  mean={m:.2f}±{ci:.2f}  uniq={u:2d}  wall={w:.0f}s',
              file=sys.stderr)
    print(json.dumps({'sweep': results, 'ranking': ranking}, indent=2, default=str))


if __name__ == "__main__":
    main()
