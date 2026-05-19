/**
 * RNG abstraction shared by browser and offline simulator.
 *
 * - `defaultRng` uses Math.random for production gameplay (matches legacy behavior).
 * - `createRng(seed)` returns a deterministic Mulberry32-backed generator for
 *   reproducible offline data generation.
 *
 * Every consumer in src/sim/* MUST go through an `Rng` instance instead of
 * calling `Math.random()` directly so offline runs can be deterministic.
 */

export function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function makeApi(randomFn, seed = null) {
  const api = {
    seed,
    random: randomFn,
    randInt(a, b) {
      return Math.floor(randomFn() * (b - a + 1)) + a;
    },
    spread(maxAbs) {
      return (randomFn() * 2 - 1) * maxAbs;
    },
    choice(arr) {
      if (!arr || arr.length === 0) return undefined;
      return arr[Math.floor(randomFn() * arr.length)];
    },
    chance(p) {
      return randomFn() < p;
    }
  };
  return api;
}

export const defaultRng = makeApi(Math.random, null);

/**
 * Mulberry32 PRNG. Deterministic given a 32-bit unsigned integer seed.
 * Good enough statistical quality for gameplay simulation and BC dataset gen.
 */
export function createRng(seed) {
  if (seed == null) return makeApi(Math.random, null);
  let s = (seed >>> 0) || 0x9e3779b9;
  function next() {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  return makeApi(next, seed);
}
