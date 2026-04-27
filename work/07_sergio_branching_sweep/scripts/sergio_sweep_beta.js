// β-sweep at α=0 to test if increasing exploration pressure pushes b_eff toward 6.
// Also tests at the optimal α from the first sweep.

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

const wrapped = `${m[1]}\nglobalThis.__FMC = FMC; globalThis.__PH = Physics;`;
new Function(wrapped)();
const FMC = globalThis.__FMC;
const Physics = globalThis.__PH;

const env = {
  cloneState: Physics.cloneState, step: Physics.step, observe: Physics.observe,
  sampleAction: Physics.sampleAction, labelAction: Physics.labelAction,
  perturbInitAction: Physics.perturbInitAction,
};

const root = Physics.initialState();
root.x = 450; root.y = 270;

function sweep(alpha, betas, seeds) {
  const N = 64, M = 30;
  const out = [];
  for (const beta of betas) {
    const branchings = [];
    for (let s = 0; s < seeds; s++) {
      const r = FMC.decide(env, root, {
        nWalkers: N, timeHorizon: M,
        alpha, beta,
        essThreshold: 0.70,
        recordTrajectories: false,
        seed: 2000 + s,
      });
      branchings.push(r.branching);
    }
    const mean = branchings.reduce((a,b) => a+b, 0) / branchings.length;
    const sd = Math.sqrt(branchings.reduce((a,b) => a + (b-mean)**2, 0) / branchings.length);
    out.push({ alpha, beta, mean, sd });
  }
  return out;
}

console.log('β-sweep at fixed α — test if exploration pressure can reach Sergio target b_eff=6');
console.log('N=64, M=30, 12 seeds per β');
console.log('');

for (const alpha of [0.0, 0.5, 1.0]) {
  console.log(`── α = ${alpha} ──`);
  console.log('  β     b_eff (mean ± sd)');
  const betas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
  const results = sweep(alpha, betas, 12);
  for (const r of results) {
    const marker = Math.abs(r.mean - 6) <= 0.5 ? '  ★ ON TARGET' : '';
    console.log(`  ${r.beta.toFixed(1).padEnd(4)}  ${r.mean.toFixed(2)} ± ${r.sd.toFixed(2)}${marker}`);
  }
  console.log('');
}
