/**
 * fmc.js — Fractal Monte Carlo (FMC) algorithm in vanilla JavaScript
 *
 * Direct port of the algorithm in:
 *   Hernández-Cerezo & Duran-Ballester, "Fractal AI: A Fragile Theory of Intelligence" (2020)
 *   §4.3 Pseudo-code
 *
 * Same algorithm as fmc_minimal.py but in JS for browser visualizations.
 *
 * The user provides an Environment object with this interface:
 *   env.cloneState(state) -> deep copy of state
 *   env.step(state, action) -> { state, reward, terminal }
 *   env.observe(state) -> feature vector (for distance metric)
 *   env.distance(obs1, obs2) -> scalar (defaults to L2)
 *   env.availableActions(state) -> array of actions (discrete) OR
 *   env.sampleAction(state) -> action (continuous)
 *
 * Configuration:
 *   nWalkers (N)        — number of parallel walkers (default 30)
 *   timeHorizon (M)     — depth of the fractal cone in ticks (default 15)
 *   balance (α)         — exploration/exploitation balance (default 1.0)
 *   distanceCoef (β)    — distance term exponent (default 1.0)
 *   rewardCoef          — alias for balance (default = balance)
 */

const FMC = (() => {
  /**
   * Relativize transformation from paper §2.2.3.
   * Maps any real vector to strictly positive values, preserving order,
   * compressing high outliers (log) and expanding low outliers (exp).
   */
  function relativize(arr) {
    const n = arr.length;
    if (n === 0) return arr;
    const mean = arr.reduce((a, b) => a + b, 0) / n;
    const variance = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    if (std === 0 || !isFinite(std)) return new Array(n).fill(1);
    return arr.map(x => {
      const z = (x - mean) / std;
      if (z <= 0) return Math.exp(Math.max(z, -50));
      return 1 + Math.log1p(z);
    });
  }

  /**
   * Default L2 distance between two observation vectors.
   */
  function l2(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s);
  }

  /**
   * Pick a random index in [0, N) different from `excluded`.
   */
  function randomOther(N, excluded) {
    if (N <= 1) return 0;
    let j;
    do { j = Math.floor(Math.random() * N); } while (j === excluded);
    return j;
  }

  /**
   * Run one FMC decision from `rootState`.
   *
   * Returns:
   *   {
   *     action: chosen action,
   *     confidence: fraction of walkers backing the chosen action,
   *     walkers: final walker pool (for visualization),
   *     trajectories: per-walker path (for visualization),
   *     counts: { action -> walker count },
   *     virtualRewards: VR vector at decision time,
   *   }
   */
  function decide(env, rootState, config) {
    const cfg = Object.assign({
      nWalkers: 30,
      timeHorizon: 15,
      balance: 1.0,
      distanceCoef: 1.0,
      isDiscrete: true,
      recordTrajectories: true,
    }, config || {});

    const N = cfg.nWalkers;
    const M = cfg.timeHorizon;

    // Determine available actions (only used in discrete mode)
    let actions = null;
    if (cfg.isDiscrete) {
      actions = env.availableActions(rootState);
      if (!actions || actions.length === 0) {
        return { action: null, confidence: 0, walkers: [], counts: {} };
      }
    }

    // Initialize N walkers, each with a random initial action
    const walkers = [];
    const trajectories = cfg.recordTrajectories ? [] : null;
    for (let i = 0; i < N; i++) {
      let initAction;
      if (cfg.isDiscrete) {
        initAction = actions[Math.floor(Math.random() * actions.length)];
      } else {
        initAction = env.sampleAction(rootState);
      }
      walkers.push({
        state: env.cloneState(rootState),
        initAction,
        cumReward: 0,
        alive: true,
      });
      if (trajectories) {
        trajectories.push([env.observe(rootState).slice()]);
      }
    }

    // Buffers reused across ticks; declared here so they're available after the loop too
    let rewards = new Array(N).fill(0);
    let distances = new Array(N).fill(0);
    let VR = new Array(N).fill(0);

    // Main FMC loop: M ticks
    for (let t = 0; t < M; t++) {
      // Step each living walker
      for (let i = 0; i < N; i++) {
        if (!walkers[i].alive) continue;
        let a;
        if (t === 0) {
          a = walkers[i].initAction;
        } else if (cfg.isDiscrete) {
          a = actions[Math.floor(Math.random() * actions.length)];
        } else {
          a = env.sampleAction(walkers[i].state);
        }
        const result = env.step(walkers[i].state, a);
        walkers[i].state = result.state;
        walkers[i].cumReward += result.reward;
        if (result.terminal) walkers[i].alive = false;
        if (trajectories) trajectories[i].push(env.observe(walkers[i].state).slice());
      }

      // Compute virtual rewards
      const observations = walkers.map(w => env.observe(w.state));
      const distFn = env.distance || l2;
      const partners = new Array(N);
      for (let i = 0; i < N; i++) {
        partners[i] = randomOther(N, i);
        distances[i] = distFn(observations[i], observations[partners[i]]);
      }

      for (let i = 0; i < N; i++) rewards[i] = walkers[i].cumReward;
      const Rn = relativize(rewards);
      const Dn = relativize(distances);
      // Penalize dead walkers
      for (let i = 0; i < N; i++) {
        if (!walkers[i].alive) { Rn[i] = 0; Dn[i] = 0; }
      }
      for (let i = 0; i < N; i++) {
        VR[i] = Math.pow(Rn[i], cfg.balance) * Math.pow(Dn[i], cfg.distanceCoef);
        if (!isFinite(VR[i])) VR[i] = 0;
      }

      // Cloning step
      for (let i = 0; i < N; i++) {
        const k = randomOther(N, i);
        let pClone;
        if (!walkers[i].alive) {
          pClone = 1.0; // dead walkers always try to clone
        } else if (VR[i] <= 1e-8) {
          pClone = 1.0;
        } else if (VR[k] <= VR[i]) {
          pClone = 0.0;
        } else {
          pClone = (VR[k] - VR[i]) / VR[i];
        }
        if (Math.random() < pClone && walkers[k].alive) {
          // Clone state, init action, reward, alive flag
          walkers[i] = {
            state: env.cloneState(walkers[k].state),
            initAction: walkers[k].initAction,
            cumReward: walkers[k].cumReward,
            alive: true,
          };
          if (trajectories) {
            trajectories[i] = trajectories[k].slice();
          }
        }
      }
    }

    // Decision: most popular initial action among living walkers
    const counts = new Map();
    let totalAlive = 0;
    for (const w of walkers) {
      if (!w.alive) continue;
      const key = JSON.stringify(w.initAction);
      counts.set(key, (counts.get(key) || 0) + 1);
      totalAlive++;
    }

    let bestKey = null;
    let bestCount = -1;
    for (const [key, count] of counts.entries()) {
      if (count > bestCount) { bestCount = count; bestKey = key; }
    }

    let chosenAction = null;
    if (bestKey !== null) {
      chosenAction = JSON.parse(bestKey);
    } else if (cfg.isDiscrete) {
      chosenAction = actions[Math.floor(Math.random() * actions.length)];
    } else {
      chosenAction = env.sampleAction(rootState);
    }

    const confidence = totalAlive > 0 ? bestCount / totalAlive : 0;

    // Compute reward variance among survivors
    const aliveRewards = walkers.filter(w => w.alive).map(w => w.cumReward);
    let rewardVariance = 0;
    if (aliveRewards.length > 1) {
      const m = aliveRewards.reduce((a, b) => a + b, 0) / aliveRewards.length;
      rewardVariance = Math.sqrt(
        aliveRewards.reduce((a, b) => a + (b - m) ** 2, 0) / aliveRewards.length
      );
    }

    return {
      action: chosenAction,
      confidence,
      counts: Object.fromEntries(
        Array.from(counts.entries()).map(([k, v]) => [k, v / Math.max(1, totalAlive)])
      ),
      walkers,
      trajectories,
      virtualRewards: VR,
      rewards,
      distances,
      rewardVariance,
      aliveCount: totalAlive,
    };
  }

  return { decide, relativize, l2 };
})();

// Export for ES module / Node usage if available
if (typeof module !== 'undefined' && module.exports) {
  module.exports = FMC;
}
