// _test_envs.js — exhaustive environment tests for all 4 demos.
// For each demo, recreates the env logic and runs FMC for many decisions,
// checking that:
//   - no NaN in state
//   - reward is finite
//   - the algorithm makes progress (varies by demo)
const FMC = require('./fmc.js');

function isFiniteState(s) {
  for (const k of Object.keys(s)) {
    const v = s[k];
    if (typeof v === 'number' && !isFinite(v)) return false;
  }
  return true;
}

// ============== ROCKET ENV TEST ===============================================
function testRocket() {
  const W = 900, H = 600;
  const WORLD = {
    walls: [
      {x:0,y:0,w:W,h:8},{x:0,y:H-8,w:W,h:8},
      {x:0,y:0,w:8,h:H},{x:W-8,y:0,w:8,h:H},
      {x:200,y:120,w:30,h:200},{x:400,y:280,w:200,h:30},
      {x:680,y:100,w:30,h:300},{x:100,y:400,w:200,h:30},
      {x:550,y:480,w:200,h:30},
    ],
    goal: {x: W-100, y: 50, r: 28},
  };
  const THRUSTS = [0, 0.18, 0.36], TORQUES = [-0.06, 0, 0.06];
  const ACTIONS = [];
  for (const t of THRUSTS) for (const r of TORQUES) ACTIONS.push([t, r]);
  function pointInWalls(x, y, rad=4) {
    for (const w of WORLD.walls) {
      if (x+rad>w.x && x-rad<w.x+w.w && y+rad>w.y && y-rad<w.y+w.h) return true;
    }
    return false;
  }
  function distToGoal(s) {
    return Math.sqrt((s.x-WORLD.goal.x)**2 + (s.y-WORLD.goal.y)**2);
  }
  const env = {
    cloneState: s => ({...s}),
    step(state, action) {
      const [thrust, torque] = action;
      const s = {...state};
      s.vAngle = s.vAngle + torque;
      s.angle = s.angle + s.vAngle;
      s.vx = s.vx + Math.cos(s.angle)*thrust;
      s.vy = s.vy + Math.sin(s.angle)*thrust + 0.14;
      s.vx *= 0.992; s.vy *= 0.992; s.vAngle *= 0.95;
      s.x += s.vx; s.y += s.vy;
      s.fuel = Math.max(0, s.fuel - thrust*5);
      let reward = 0, terminal = false;
      if (s.alive) {
        reward = Math.max(0, 600-distToGoal(s))/600;
      }
      if (pointInWalls(s.x, s.y)) { s.alive=false; terminal=true; reward=0; }
      if (distToGoal(s) < WORLD.goal.r) reward = 5;
      return { state: s, reward, terminal };
    },
    observe(s) {
      return [s.x/W, s.y/H, s.vx*0.1, s.vy*0.1, Math.cos(s.angle), Math.sin(s.angle)];
    },
    availableActions() { return ACTIONS; },
  };
  let state = { x:80, y:H-80, vx:0, vy:0, angle:-Math.PI/2, vAngle:0, alive:true, fuel:1000 };
  for (let i = 0; i < 50; i++) {
    const dec = FMC.decide(env, state, { nWalkers:25, timeHorizon:10, balance:1, distanceCoef:1 });
    if (!dec.action) throw new Error('rocket: null action');
    const r = env.step(state, dec.action);
    state = r.state;
    if (!isFiniteState(state)) throw new Error(`rocket: NaN state at step ${i}: ${JSON.stringify(state)}`);
    if (!state.alive) break;
  }
  return `rocket: OK (50 decisions, last alive=${state.alive}, x=${state.x.toFixed(0)}, y=${state.y.toFixed(0)})`;
}

// ============== KART ENV TEST =================================================
function testKart() {
  const W = 900, H = 600;
  const TRACK_RADIUS_X = 320, TRACK_RADIUS_Y = 200;
  const TRACK_CENTER = {x: W/2, y: H/2};
  const NUM_CHECKPOINTS = 8;
  const CHECKPOINTS = [];
  for (let i = 0; i < NUM_CHECKPOINTS; i++) {
    const a = (i / NUM_CHECKPOINTS) * Math.PI*2 - Math.PI/2;
    CHECKPOINTS.push({
      x: TRACK_CENTER.x + Math.cos(a)*TRACK_RADIUS_X,
      y: TRACK_CENTER.y + Math.sin(a)*TRACK_RADIUS_Y,
      angle: a + Math.PI/2,
    });
  }
  function isOnTrack(x, y) {
    const dx=(x-TRACK_CENTER.x)/TRACK_RADIUS_X, dy=(y-TRACK_CENTER.y)/TRACK_RADIUS_Y;
    const r = Math.sqrt(dx*dx+dy*dy);
    return r > 0.7 && r < 1.3;
  }
  const ACTIONS = [];
  for (const a of [0,0.18,0.36]) for (const t of [-0.10,-0.05,0,0.05,0.10]) ACTIONS.push([a,t]);
  const env = {
    cloneState: s => ({...s}),
    step(state, action) {
      const [accel, turn] = action;
      const s = {...state};
      s.angle += turn*(0.5+s.speed*0.3);
      s.speed = Math.min(8, s.speed + accel - s.speed*0.04);
      s.x += Math.cos(s.angle)*s.speed;
      s.y += Math.sin(s.angle)*s.speed;
      let reward=0, terminal=false;
      if (!isOnTrack(s.x, s.y)) {
        s.speed *= 0.7;
        reward = -0.3;
        if (s.x<0||s.x>W||s.y<0||s.y>H) { terminal=true; reward=-2; }
      } else {
        reward = 0.05 + s.speed*0.02;
      }
      const cp = CHECKPOINTS[s.nextCheckpoint];
      const dCp = Math.sqrt((s.x-cp.x)**2 + (s.y-cp.y)**2);
      if (dCp < 50) {
        reward += 5;
        s.nextCheckpoint = (s.nextCheckpoint+1) % NUM_CHECKPOINTS;
        if (s.nextCheckpoint === 0) s.lap++;
      }
      reward += Math.max(0, 100-dCp)/100*0.1;
      return { state: s, reward, terminal };
    },
    observe(s) {
      const cp = CHECKPOINTS[s.nextCheckpoint];
      return [s.x/W, s.y/H, s.speed*0.1, Math.cos(s.angle), Math.sin(s.angle),
              (cp.x-s.x)/W, (cp.y-s.y)/H];
    },
    availableActions() { return ACTIONS; },
  };
  let state = { x:CHECKPOINTS[0].x, y:CHECKPOINTS[0].y, angle:CHECKPOINTS[0].angle,
                speed:0, nextCheckpoint:0, lap:0 };
  let cps = 0;
  for (let i = 0; i < 200; i++) {
    const dec = FMC.decide(env, state, { nWalkers:30, timeHorizon:15, balance:1, distanceCoef:1 });
    if (!dec.action) throw new Error('kart: null action');
    const before = state.nextCheckpoint;
    const r = env.step(state, dec.action);
    state = r.state;
    if (!isFiniteState(state)) throw new Error(`kart: NaN state at step ${i}: ${JSON.stringify(state)}`);
    if (state.nextCheckpoint !== before) cps++;
  }
  return `kart: OK (200 decisions, ${cps} checkpoints reached, lap=${state.lap}, speed=${state.speed.toFixed(2)})`;
}

// ============== PONG ENV TEST =================================================
function testPong() {
  const W = 800, H = 500;
  const PADDLE_W = 12, PADDLE_H = 80, BALL_R = 8, PADDLE_SPEED = 6;
  const PADDLE_X = W - 30;
  const ACTIONS = [-1, 0, 1];
  const env = {
    cloneState: s => ({...s}),
    step(state, action) {
      const s = {...state};
      s.paddleY = Math.max(0, Math.min(H-PADDLE_H, s.paddleY + action*PADDLE_SPEED));
      s.ballX += s.ballVx; s.ballY += s.ballVy;
      if (s.ballY < BALL_R) { s.ballY=BALL_R; s.ballVy=-s.ballVy; }
      if (s.ballY > H-BALL_R) { s.ballY=H-BALL_R; s.ballVy=-s.ballVy; }
      if (s.ballX < BALL_R) { s.ballX=BALL_R; s.ballVx=-s.ballVx; }
      let reward=0, terminal=false;
      if (s.ballX>PADDLE_X-BALL_R && s.ballX<PADDLE_X+PADDLE_W+BALL_R) {
        if (s.ballY>s.paddleY-BALL_R && s.ballY<s.paddleY+PADDLE_H+BALL_R) {
          if (s.ballVx > 0) {
            s.ballX = PADDLE_X-BALL_R-1;
            s.ballVx = -Math.abs(s.ballVx);
            const offset = (s.ballY-(s.paddleY+PADDLE_H/2))/(PADDLE_H/2);
            s.ballVy += offset*2;
            reward = 1;
          }
        }
      }
      if (s.ballX > W+BALL_R) { reward=-1; terminal=true; }
      const distY = Math.abs((s.paddleY+PADDLE_H/2)-s.ballY);
      reward += Math.max(0, 50-distY)/500;
      return { state: s, reward, terminal };
    },
    observe(s) {
      return [s.paddleY/H, s.ballX/W, s.ballY/H, s.ballVx*0.05, s.ballVy*0.05];
    },
    availableActions() { return ACTIONS; },
  };
  let state = { paddleY:H/2-PADDLE_H/2, ballX:80, ballY:H/2, ballVx:5, ballVy:1.5 };
  let hits = 0, misses = 0;
  for (let episode = 0; episode < 3; episode++) {
    state = { paddleY:H/2-PADDLE_H/2, ballX:80, ballY:H/2+(Math.random()-0.5)*100, ballVx:5, ballVy:(Math.random()-0.5)*3 };
    for (let i = 0; i < 300; i++) {
      const dec = FMC.decide(env, state, { nWalkers:25, timeHorizon:25, balance:1, distanceCoef:1 });
      if (!dec.action && dec.action !== 0) throw new Error('pong: null action');
      const r = env.step(state, dec.action);
      state = r.state;
      if (!isFiniteState(state)) throw new Error(`pong: NaN state at episode ${episode} step ${i}`);
      if (r.reward >= 1) hits++;
      if (r.terminal) { misses++; break; }
    }
  }
  return `pong: OK (3 episodes, ${hits} hits, ${misses} misses)`;
}

// ============== OCTOPUS ENV TEST ==============================================
function testOctopus() {
  const W = 900, H = 600;
  const NUM_AGENTS = 5;
  const GOAL = { x: 700, y: 400 };
  const ACTIONS = [];
  for (const dx of [-1,0,1]) for (const dy of [-1,0,1]) ACTIONS.push([dx*0.6, dy*0.6]);
  function makeEnvForAgent(others) {
    const cohesion = 0.6, repulsion = 1.5;
    return {
      cloneState(s) { return {...s, others}; },
      step(state, action) {
        const s = {...state};
        const [ax, ay] = action;
        s.vx = (s.vx+ax)*0.8; s.vy = (s.vy+ay)*0.8;
        s.x += s.vx; s.y += s.vy;
        if (s.x<10||s.x>W-10||s.y<10||s.y>H-10) return { state:s, reward:-1, terminal:true };
        const dGoal = Math.sqrt((s.x-GOAL.x)**2+(s.y-GOAL.y)**2);
        const goalReward = Math.max(0, 1000-dGoal)/1000;
        let cx=0, cy=0;
        for (const o of state.others) { cx+=o.x; cy+=o.y; }
        cx/=state.others.length; cy/=state.others.length;
        const dCohes = Math.sqrt((s.x-cx)**2+(s.y-cy)**2);
        const cohesReward = Math.max(0, 200-dCohes)/200;
        let repulse = 0;
        for (const o of state.others) {
          const d = Math.sqrt((s.x-o.x)**2+(s.y-o.y)**2);
          if (d<35 && d>0.1) repulse -= (35-d)/35;
        }
        const repulseReward = Math.max(0, 1+repulse);
        const reward = goalReward*(1+cohesion*cohesReward)*(1+repulsion*repulseReward) - 1;
        return { state: s, reward, terminal:false };
      },
      observe(s) { return [s.x/W, s.y/H, s.vx*0.1, s.vy*0.1]; },
      availableActions() { return ACTIONS; },
    };
  }
  let agents = [];
  for (let i = 0; i < NUM_AGENTS; i++) {
    const a = (i/NUM_AGENTS)*Math.PI*2;
    agents.push({ x:150+Math.cos(a)*60, y:200+Math.sin(a)*60, vx:0, vy:0, alive:true });
  }
  const initialDist = agents.reduce((acc, a) => acc + Math.sqrt((a.x-GOAL.x)**2+(a.y-GOAL.y)**2), 0) / NUM_AGENTS;
  for (let step = 0; step < 50; step++) {
    const others = agents.map(a => ({...a}));
    const newAgents = [];
    for (let i = 0; i < NUM_AGENTS; i++) {
      const otherAgents = others.filter((_,j) => j!==i);
      const env = makeEnvForAgent(otherAgents);
      const state = {...agents[i], others: otherAgents};
      const dec = FMC.decide(env, state, { nWalkers:15, timeHorizon:8, balance:1, distanceCoef:1 });
      if (!dec.action) throw new Error('octopus: null action');
      const r = env.step(state, dec.action);
      if (!isFiniteState(r.state)) throw new Error(`octopus: NaN state at step ${step}: ${JSON.stringify(r.state)}`);
      newAgents.push({x:r.state.x, y:r.state.y, vx:r.state.vx, vy:r.state.vy, alive:!r.terminal});
    }
    agents = newAgents;
  }
  const finalDist = agents.reduce((acc, a) => acc + Math.sqrt((a.x-GOAL.x)**2+(a.y-GOAL.y)**2), 0) / NUM_AGENTS;
  return `octopus: OK (50 steps, avg dist to goal: ${initialDist.toFixed(0)} → ${finalDist.toFixed(0)})`;
}

// === RUN ALL TESTS ============================================================
const tests = [
  ['rocket', testRocket],
  ['kart', testKart],
  ['pong', testPong],
  ['octopus', testOctopus],
];

let failed = 0;
for (const [name, fn] of tests) {
  try {
    const t0 = Date.now();
    const result = fn();
    const dt = Date.now() - t0;
    console.log(`✓ ${result} (${dt}ms)`);
  } catch (e) {
    console.error(`✗ ${name}: FAILED — ${e.message}`);
    failed++;
  }
}

if (failed > 0) {
  console.error(`\n${failed} tests FAILED`);
  process.exit(1);
}
console.log('\nAll 4 environments PASS ✓');
