// Fine sweep around (α=0, β∈[0, 1]) to nail the empirical sweet spot.
const fs = require('fs');
const html = fs.readFileSync(
  '/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/simulations/rocket_validated.html',
  'utf8'
);
const m = html.match(/<script>\s*"use strict";([\s\S]*?)<\/script>/);
const mockCtx = new Proxy({}, { get: () => () => {} });
const mockEl = {
  innerHTML: '', textContent: '', style: {},
  classList: { add: () => {}, remove: () => {}, contains: () => false },
  appendChild: () => {}, querySelector: () => mockEl,
  addEventListener: () => {}, children: [], dataset: {}, disabled: false, value: '64',
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
new Function(`${m[1]}\nglobalThis.__FMC=FMC;globalThis.__PH=Physics;`)();
const FMC = globalThis.__FMC;
const Physics = globalThis.__PH;

const env = {
  cloneState: Physics.cloneState, step: Physics.step, observe: Physics.observe,
  sampleAction: Physics.sampleAction, labelAction: Physics.labelAction,
  perturbInitAction: Physics.perturbInitAction,
};
const root = Physics.initialState();
root.x = 450; root.y = 270;

console.log('Fine sweep — (α=0, β varying), N=64, M=30, 20 seeds');
console.log('  β       b_eff (mean ± sd)   gap vs 6');
console.log('  ──────  ──────────────────  ────────');

const SEEDS = 20;
const grid = [];
for (let b = 0.0; b <= 1.005; b += 0.1) {
  const beta = Number(b.toFixed(2));
  const branchings = [];
  for (let s = 0; s < SEEDS; s++) {
    const r = FMC.decide(env, root, {
      nWalkers: 64, timeHorizon: 30,
      alpha: 0.0, beta,
      essThreshold: 0.70, recordTrajectories: false,
      seed: 3000 + s,
    });
    branchings.push(r.branching);
  }
  const mean = branchings.reduce((a,b)=>a+b,0)/SEEDS;
  const sd = Math.sqrt(branchings.reduce((a,b)=>a+(b-mean)**2,0)/SEEDS);
  const gap = mean - 6.0;
  const m2 = Math.abs(gap) <= 0.5 ? '  ★ ON SERGIO TARGET' : '';
  console.log(`  ${beta.toFixed(2).padEnd(6)}  ${mean.toFixed(2)} ± ${sd.toFixed(2)}     ${gap >= 0 ? '+' : ''}${gap.toFixed(2)}${m2}`);
  grid.push({ beta, mean, sd });
}

console.log('');
// Same for α=0.1, α=0.2 to rule out isolation effect
for (const alpha of [0.1, 0.2, 0.3]) {
  console.log(`Same sweep at α=${alpha}:`);
  console.log('  β       b_eff (mean ± sd)   gap');
  for (let b = 0.0; b <= 1.005; b += 0.2) {
    const beta = Number(b.toFixed(2));
    const branchings = [];
    for (let s = 0; s < SEEDS; s++) {
      const r = FMC.decide(env, root, {
        nWalkers: 64, timeHorizon: 30,
        alpha, beta,
        essThreshold: 0.70, recordTrajectories: false,
        seed: 4000 + s,
      });
      branchings.push(r.branching);
    }
    const mean = branchings.reduce((a,b)=>a+b,0)/SEEDS;
    const sd = Math.sqrt(branchings.reduce((a,b)=>a+(b-mean)**2,0)/SEEDS);
    const gap = mean - 6.0;
    const m2 = Math.abs(gap) <= 0.5 ? '  ★ ON SERGIO TARGET' : '';
    console.log(`  ${beta.toFixed(2).padEnd(6)}  ${mean.toFixed(2)} ± ${sd.toFixed(2)}     ${gap >= 0 ? '+' : ''}${gap.toFixed(2)}${m2}`);
  }
  console.log('');
}
