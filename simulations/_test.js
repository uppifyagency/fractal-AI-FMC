// End-to-end smoke test: load fmc.js and run a few decisions on a tiny env.
// Validates that the algorithm doesn't crash and produces sensible output.

const FMC = require('./fmc.js');

const env = {
  cloneState: s => ({ ...s }),
  step(state, action) {
    const s = { ...state };
    s.x += action;
    const reward = s.x; // higher x = better
    return { state: s, reward, terminal: false };
  },
  observe(s) { return [s.x]; },
  availableActions() { return [-1, 0, 1]; },
};

const initialState = { x: 0 };

console.log('Running 30 FMC decisions on a 1D toy env...');
let state = initialState;
let cumReward = 0;
for (let i = 0; i < 30; i++) {
  const dec = FMC.decide(env, state, {
    nWalkers: 30, timeHorizon: 10, balance: 1.0, distanceCoef: 1.0,
  });
  if (dec.action === null) { console.error('FAIL: action is null'); process.exit(1); }
  const r = env.step(state, dec.action);
  state = r.state;
  cumReward += r.reward;
  if (i % 10 === 0) {
    console.log(`  step ${i}: action=${dec.action} x=${state.x.toFixed(1)} conf=${(dec.confidence*100).toFixed(0)}%`);
  }
}
console.log(`Final x=${state.x}, cumulative reward=${cumReward.toFixed(1)}`);
if (state.x < 25) {
  console.error('FAIL: agent should have moved right (action=+1) most of the time');
  process.exit(1);
}
console.log('OK ✓ — FMC algorithm runs and chooses optimal action');
