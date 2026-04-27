/**
 * Node test runner. Mirrors the Python tests and prints a summary.
 *
 * Usage: node _test.js
 */

const FMC = require("./fmc.js");

let passed = 0;
let failed = 0;
const failures = [];

function approxEqual(a, b, tol = 1e-12) {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

function arraysApproxEqual(a, b, tol = 1e-12) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!approxEqual(a[i], b[i], tol)) return false;
  }
  return true;
}

function test(name, fn) {
  try {
    fn();
    passed++;
    process.stdout.write(".");
  } catch (e) {
    failed++;
    failures.push({ name, message: e.message });
    process.stdout.write("F");
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || "assertion failed");
}

// ---------------- Definition 2: relativize ----------------

test("relativize positivity", () => {
  const inputs = [
    [-100, -10, -1, 0, 1, 10],
    [1e-12, 1e-12, 1e-12],
    [-1e9, -1e9, 1e9],
  ];
  for (const r of inputs) {
    const out = FMC.relativize(r);
    for (const x of out) assert(x > 0, `non-positive: ${x}`);
  }
});

test("relativize constant input returns ones", () => {
  const out = FMC.relativize([5, 5, 5, 5]);
  for (const x of out) assert(approxEqual(x, 1.0));
});

test("relativize affine invariance", () => {
  const r = [0.5, 1.0, 2.0, 4.0, 8.0];
  const base = FMC.relativize(r);
  const cases = [[1, 0], [2, 5], [100, -50], [0.001, 1000]];
  for (const [a, b] of cases) {
    const transformed = r.map((x) => a * x + b);
    const out = FMC.relativize(transformed);
    assert(arraysApproxEqual(base, out, 1e-10), `affine invariance fail at a=${a},b=${b}`);
  }
});

// ---------------- Definition 5: ESS ----------------

test("ess uniform weights gives N", () => {
  const vr = new Array(50).fill(1);
  assert(approxEqual(FMC.effectiveSampleSize(vr), 50));
});

test("ess single dominant gives one", () => {
  const vr = new Array(50).fill(0);
  vr[7] = 1e9;
  assert(approxEqual(FMC.effectiveSampleSize(vr), 1.0, 1e-6));
});

// ---------------- Definition 6: branching ----------------

test("branching palmera = 1", () => {
  assert(approxEqual(FMC.effectiveBranchingFactor([3, 3, 3, 3]), 1.0));
});

test("branching matorral = K", () => {
  const K = 9;
  const labels = [];
  for (let k = 0; k < K; k++) for (let n = 0; n < 10; n++) labels.push(k);
  assert(approxEqual(FMC.effectiveBranchingFactor(labels), K, 1e-9));
});

test("branching 50/50 split = 2", () => {
  const labels = [];
  for (let i = 0; i < 32; i++) labels.push(0);
  for (let i = 0; i < 32; i++) labels.push(1);
  assert(approxEqual(FMC.effectiveBranchingFactor(labels), 2.0, 1e-9));
});

// ---------------- Definition 4: clone_step ----------------

test("clone uniform vr means stay", () => {
  const vr = new Array(64).fill(1.0);
  const partners = Array.from({ length: 64 }, (_, i) => (i + 1) % 64);
  const samples = new Array(64).fill(0.5);
  const out = FMC.cloneStep(vr, partners, samples);
  for (let i = 0; i < 64; i++) assert(out[i] === i, `expected stay at ${i}, got ${out[i]}`);
});

test("clone zero vr always clones", () => {
  const vr = [0.0, 1.0, 2.0, 3.0];
  const partners = [1, 2, 3, 0];
  const samples = [0.99, 0.99, 0.99, 0.99];
  const out = FMC.cloneStep(vr, partners, samples);
  assert(out[0] === 1, `walker 0 should clone, got idx=${out[0]}`);
});

// ---------------- Decide ----------------

test("decide argmax bincount", () => {
  assert(FMC.decide([0, 0, 1, 1, 1, 2]) === 1);
});

// ---------------- Cross-language fixture (optional) ----------------

const fs = require("fs");
const path = require("path");
const fixturePath = path.join(__dirname, "..", "tests", "_fixture_relativize.json");
if (fs.existsSync(fixturePath)) {
  const fixtures = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  test("relativize matches Python fixture (bit-for-bit)", () => {
    for (const fx of fixtures) {
      const out = FMC.relativize(fx.input);
      assert(
        arraysApproxEqual(out, fx.expected, 1e-12),
        `fixture mismatch on input length ${fx.input.length}`,
      );
    }
  });

  test("virtualReward matches Python fixture (bit-for-bit)", () => {
    for (const fx of fixtures) {
      if (!fx.virtual_reward) continue;
      const states = fx.states;
      const out = FMC.virtualReward(
        fx.input,
        states,
        fx.partners,
        fx.alpha,
        fx.beta,
      );
      assert(
        arraysApproxEqual(out, fx.virtual_reward, 1e-12),
        `VR fixture mismatch`,
      );
    }
  });
}

console.log("");
if (failed === 0) {
  console.log(`OK ${passed} passed, 0 failed`);
  process.exit(0);
} else {
  console.log(`FAIL ${passed} passed, ${failed} failed`);
  for (const f of failures) console.log(`  - ${f.name}: ${f.message}`);
  process.exit(1);
}
