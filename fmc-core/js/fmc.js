/**
 * fmc-core JS port — Fractal Monte Carlo math layer.
 *
 * Mirror of Python src/fmc/core.py. Pure-math functions (relativize,
 * virtualReward, ess, effectiveBranching) produce bit-for-bit identical
 * output as the Python version when given the same float64 inputs.
 *
 * cloneStep accepts an externally provided uniform-random vector to allow
 * cross-language testing without relying on platform-specific RNG.
 */

const EPS = 1e-10;

// --------------------------------------------------------------------------
// Helpers (since plain JS arrays lack vectorized ops).
// --------------------------------------------------------------------------

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function std(arr) {
  const mu = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - mu;
    s += d * d;
  }
  return Math.sqrt(s / arr.length); // population std, ddof=0 (matches NumPy default).
}

// --------------------------------------------------------------------------
// Definition 2 — relativize
// --------------------------------------------------------------------------

function relativize(vector) {
  const n = vector.length;
  const sd = std(vector);
  if (sd === 0) {
    return new Float64Array(n).fill(1.0);
  }
  const mu = mean(vector);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const z = (vector[i] - mu) / sd;
    if (z > 0) {
      out[i] = Math.log(1 + z) + 1;
    } else {
      out[i] = Math.exp(z);
    }
  }
  return out;
}

// --------------------------------------------------------------------------
// Definition 3 — virtual reward
// --------------------------------------------------------------------------

function virtualReward(rewards, states, partners, alpha = 1.0, beta = 1.0) {
  const n = rewards.length;
  // states: 2D array of shape [n, dim] (array of arrays or Float64Array).
  // Compute pairwise euclidean distances.
  const dist = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const j = partners[i];
    let s = 0;
    const xi = states[i];
    const xj = states[j];
    for (let k = 0; k < xi.length; k++) {
      const d = xi[k] - xj[k];
      s += d * d;
    }
    dist[i] = Math.sqrt(s);
  }
  const rHat = relativize(rewards);
  const dHat = relativize(dist);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.pow(rHat[i], alpha) * Math.pow(dHat[i], beta);
  }
  return out;
}

// --------------------------------------------------------------------------
// Definition 5 — effective sample size
// --------------------------------------------------------------------------

function effectiveSampleSize(vr) {
  let s = 0, s2 = 0;
  for (let i = 0; i < vr.length; i++) {
    s += vr[i];
    s2 += vr[i] * vr[i];
  }
  if (s2 === 0) return vr.length;
  return (s * s) / s2;
}

// --------------------------------------------------------------------------
// Definition 6 — effective branching factor
// --------------------------------------------------------------------------

function effectiveBranchingFactor(labels) {
  if (labels.length === 0) return 1.0;
  const counts = new Map();
  for (let i = 0; i < labels.length; i++) {
    const k = labels[i];
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const total = labels.length;
  let H = 0;
  for (const c of counts.values()) {
    if (c === 0) continue;
    const p = c / total;
    H -= p * Math.log(p);
  }
  return Math.exp(H);
}

// --------------------------------------------------------------------------
// Definition 4 — cloning step
//
// Uniform-random samples [0, 1) and pre-chosen partners are passed in to
// allow deterministic cross-language tests.
// --------------------------------------------------------------------------

function cloneStep(vr, partners, uniformSamples) {
  const n = vr.length;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const k = partners[i];
    let pClone;
    if (vr[i] === 0) {
      pClone = 1.0;
    } else if (vr[k] <= vr[i]) {
      pClone = 0.0;
    } else {
      pClone = (vr[k] - vr[i]) / vr[i];
      if (pClone > 1.0) pClone = 1.0;
    }
    out[i] = uniformSamples[i] < pClone ? k : i;
  }
  return out;
}

// --------------------------------------------------------------------------
// Decide
// --------------------------------------------------------------------------

function decide(labels) {
  const counts = new Map();
  let bestLabel = labels[0];
  let bestCount = -1;
  for (let i = 0; i < labels.length; i++) {
    const k = labels[i];
    const c = (counts.get(k) || 0) + 1;
    counts.set(k, c);
    if (c > bestCount) {
      bestCount = c;
      bestLabel = k;
    }
  }
  return bestLabel;
}

// --------------------------------------------------------------------------
// Module export (works in both Node and browser <script>).
// --------------------------------------------------------------------------

const FMC = {
  relativize,
  virtualReward,
  effectiveSampleSize,
  effectiveBranchingFactor,
  cloneStep,
  decide,
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = FMC;
}
if (typeof window !== "undefined") {
  window.FMC = FMC;
}
