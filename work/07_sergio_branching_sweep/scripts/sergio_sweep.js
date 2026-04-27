// α-sweep on the validated rocket simulator. For each α in [0, 1],
// measure b_eff (Sergio's effective branching) averaged over multiple seeds,
// then identify the α* that lands closest to Sergio's target of 6.

const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(
  '/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/simulations/rocket_validated.html',
  'utf8'
);
const m = html.match(/<script>\s*"use strict";([\s\S]*?)<\/script>/);
if (!m) { console.error('script not found'); process.exit(1); }

// Mock DOM (same trick as run_rocket_tests.js).
const mockCtx = new Proxy({}, { get: () => () => {} });
const mockEl = {
  innerHTML: '', textContent: '', style: {},
  classList: { add: () => {}, remove: () => {}, contains: () => false },
  appendChild: () => {}, querySelector: () => mockEl,
  addEventListener: () => {},
  children: [], dataset: {}, disabled: false, value: '64',
  getContext: () => mockCtx,
};
mockEl.children = new Proxy([], {
  get(_, p) { return p === 'length' ? 0 : (typeof p === 'string' && /^\d+$/.test(p) ? mockEl : undefined); }
});
global.document = {
  getElementById: () => ({...mockEl, querySelector: () => mockEl}),
  createElement: () => ({...mockEl, querySelector: () => mockEl}),
};
global.window = { addEventListener: () => {} };
global.performance = { now: () => Date.now() };
global.requestAnimationFrame = (fn) => setTimeout(fn, 0);

const wrapped = `
${m[1]}
globalThis.__FMC = FMC;
globalThis.__PH = Physics;
`;
new Function(wrapped)();
const FMC = globalThis.__FMC;
const Physics = globalThis.__PH;

// Build the env once.
const env = {
  cloneState: Physics.cloneState,
  step: Physics.step,
  observe: Physics.observe,
  sampleAction: Physics.sampleAction,
  labelAction: Physics.labelAction,
  perturbInitAction: Physics.perturbInitAction,
};

// Sweep configuration.
const CONFIG = {
  N: 64, M: 30,
  beta: 1.0,
  essThreshold: 0.70,
  alphas: [],     // populated below
  seedsPerAlpha: 12,
  // Try fine resolution near where the value crosses 6.
};
for (let a = 0; a <= 1.001; a += 0.05) CONFIG.alphas.push(Number(a.toFixed(2)));

const root = Physics.initialState();
root.x = 450; root.y = 270;

console.log(`Sergio's α-sweep — target b_eff ≈ 6`);
console.log(`N=${CONFIG.N}, M=${CONFIG.M}, β=${CONFIG.beta}, ${CONFIG.seedsPerAlpha} seeds per α`);
console.log(`Sweeping α in [${CONFIG.alphas[0]}, ${CONFIG.alphas[CONFIG.alphas.length-1]}]`);
console.log('');
console.log('  α     b_eff (mean ± stddev)   gap from 6   diagnostic');
console.log('  ──── ───────────────────────  ──────────  ─────────────────────');

const results = [];
for (const alpha of CONFIG.alphas) {
  const branchings = [];
  const t0 = Date.now();
  for (let seed = 0; seed < CONFIG.seedsPerAlpha; seed++) {
    const r = FMC.decide(env, root, {
      nWalkers: CONFIG.N, timeHorizon: CONFIG.M,
      alpha, beta: CONFIG.beta,
      essThreshold: CONFIG.essThreshold,
      recordTrajectories: false,
      seed: 1000 + seed,
    });
    branchings.push(r.branching);
  }
  const mean = branchings.reduce((a,b) => a+b, 0) / branchings.length;
  const variance = branchings.reduce((a,b) => a + (b-mean)**2, 0) / branchings.length;
  const stddev = Math.sqrt(variance);
  const gap = mean - 6.0;
  const diag = mean <= 1.5 ? 'palmera'
             : mean >= 7.5 ? 'matorral'
             : Math.abs(gap) <= 0.5 ? 'ON SERGIO TARGET'
             : (mean < 6 ? 'tunable, needs more diversity' : 'tunable, needs more selection');
  results.push({ alpha, mean, stddev, gap, branchings });
  const ms = Date.now() - t0;
  console.log(`  ${alpha.toFixed(2)}  ${mean.toFixed(2)} ± ${stddev.toFixed(2)} (${ms}ms)        ${gap >= 0 ? '+' : ''}${gap.toFixed(2)}     ${diag}`);
}

console.log('');
// Find α* whose mean is closest to 6.
results.sort((a, b) => Math.abs(a.mean - 6) - Math.abs(b.mean - 6));
const best = results[0];
console.log(`★ Best α: ${best.alpha.toFixed(2)} → b_eff = ${best.mean.toFixed(2)} ± ${best.stddev.toFixed(2)} (gap from 6: ${best.gap >= 0 ? '+' : ''}${best.gap.toFixed(2)})`);

// Find the lowest α that lands under 6 (i.e. the regime change).
results.sort((a, b) => a.alpha - b.alpha);
let crossLo = null, crossHi = null;
for (let i = 0; i < results.length - 1; i++) {
  if (results[i].mean >= 6 && results[i+1].mean < 6) {
    crossLo = results[i]; crossHi = results[i+1];
    break;
  }
}
if (crossLo) {
  console.log(`Crossing of b_eff=6: between α=${crossLo.alpha.toFixed(2)} (b=${crossLo.mean.toFixed(2)}) and α=${crossHi.alpha.toFixed(2)} (b=${crossHi.mean.toFixed(2)})`);
  // Linear interpolation.
  const tCross = (crossLo.mean - 6) / (crossLo.mean - crossHi.mean);
  const alphaStar = crossLo.alpha + tCross * (crossHi.alpha - crossLo.alpha);
  console.log(`Interpolated α* (where b_eff=6 exactly): ${alphaStar.toFixed(3)}`);
}

// Save full sweep as JSON for the HTML to consume.
const out = {
  config: CONFIG,
  date: new Date().toISOString(),
  results: results.map(r => ({
    alpha: r.alpha,
    mean: r.mean,
    stddev: r.stddev,
    branchings: r.branchings,
  })),
};
fs.writeFileSync('/tmp/sergio_sweep_result.json', JSON.stringify(out, null, 2));
console.log('');
console.log(`Saved full sweep to /tmp/sergio_sweep_result.json`);
