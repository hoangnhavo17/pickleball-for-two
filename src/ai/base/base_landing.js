/**
 * Base ball-landing predictor (shared by every CPU model).
 *
 * Currently this is a thin classifier-style wrapper around the deterministic
 * physics simulator in `src/sim/physics.js`. Wrapping it here means a future
 * learned regressor can drop in with the same contract without touching the
 * runtime or any concrete policy.
 *
 * Public API:
 *   predictFirstBounce(ball)              -> { x, y, x01, y01 } | null
 *   predictReachableAtCpu(state, ball)    -> { x, y, x01, y01 } | null
 *   predictLanding({ ball, state? })      -> { kind, x, y, x01, y01, confidence } | null
 */

import {
  WIDTH,
  HEIGHT,
  predictFirstGroundBouncePoint,
  predictBallAtCpu
} from "../../sim/index.js";

function withNormalized(p) {
  if (!p) return null;
  return {
    x: p.x,
    y: p.y,
    x01: WIDTH > 0 ? p.x / WIDTH : 0,
    y01: HEIGHT > 0 ? p.y / HEIGHT : 0
  };
}

export function predictFirstBounce(ball) {
  return withNormalized(predictFirstGroundBouncePoint(ball));
}

export function predictReachableAtCpu(state, ball) {
  return withNormalized(predictBallAtCpu(state, ball));
}

/**
 * Unified landing prediction contract. Tries the "reachable at CPU" path
 * first (which respects two-bounce rules) and falls back to the raw first
 * ground bounce. `confidence` reflects which branch produced the answer
 * (deterministic for now: 1.0 if reachable, 0.6 if first-bounce only).
 */
export function predictLanding({ ball, state }) {
  if (!ball) return null;
  if (state) {
    const reach = predictReachableAtCpu(state, ball);
    if (reach) return { kind: "reachable_at_cpu", confidence: 1.0, ...reach };
  }
  const bounce = predictFirstBounce(ball);
  if (bounce) return { kind: "first_bounce", confidence: 0.6, ...bounce };
  return null;
}
