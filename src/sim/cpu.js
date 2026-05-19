/**
 * Pure CPU AI helpers shared between browser and offline simulator.
 *
 * - Heuristic intercept target (clean, no movement noise).
 * - 24-dim feature extraction matching ml/config/feature_schema.json.
 * - Expert action label (used as the supervised target for behavior cloning).
 * - Hard-state score for replay upweighting.
 *
 * The action grid (5x5) and `actionToDelta` mapping are passed in by the
 * caller so this module stays decoupled from the policy implementation.
 */

import {
  ACTION_DIM,
  COURT_SPEED_SCALE,
  CPU_HEURISTIC_LAND_BIAS_X,
  CPU_HEURISTIC_NET_GUARD_X,
  CPU_RALLY_MAX_X,
  CPU_RALLY_MIN_X,
  HALF_COURT_DEPTH,
  HEIGHT,
  KITCHEN_HALF,
  LEGACY_HALF_COURT_DEPTH,
  MAX_CONTACT_HEIGHT,
  NET_X,
  PADDLE_REACH_BOTTOM,
  PADDLE_REACH_TOP,
  WIDTH
} from "./constants.js";
import {
  paddleHitHeight01,
  predictBallAtCpu,
  predictFirstGroundBouncePoint
} from "./physics.js";
import { clamp } from "./rng.js";
import { isInCpuKitchen, mustLetBallBounceBeforeReturn } from "./rules.js";

/** Action index -> (dx, dy) on a 5x5 grid in {-1, -0.5, 0, 0.5, 1}. */
export function actionToDelta(a, grid = 5) {
  const center = (grid - 1) / 2;
  const ix = a % grid;
  const iy = Math.floor(a / grid);
  return [(ix - center) / center, (iy - center) / center];
}

/** Inverse of actionToDelta. */
export function deltaToAction(dx, dy, grid = 5) {
  const center = (grid - 1) / 2;
  const sx = Math.round(clamp(dx, -1, 1) * center) + center;
  const sy = Math.round(clamp(dy, -1, 1) * center) + center;
  return sx + sy * grid;
}

/**
 * Clean intercept target for imitation + ML features. No tracking noise.
 * @returns {{hasTarget:boolean, tx?:number, ty?:number}}
 */
export function getCpuHeuristicTarget(state, ball, cpu) {
  if (!(ball.vx > 0.5 && ball.lastHitter !== "cpu")) return { hasTarget: false };
  if (mustLetBallBounceBeforeReturn(state, ball)) {
    const land = predictFirstGroundBouncePoint(ball);
    if (!land) return { hasTarget: false };
    const tx = clamp(land.x + CPU_HEURISTIC_LAND_BIAS_X, NET_X + CPU_HEURISTIC_NET_GUARD_X, CPU_RALLY_MAX_X);
    const ty = clamp(land.y, PADDLE_REACH_TOP, PADDLE_REACH_BOTTOM);
    return { hasTarget: true, tx, ty };
  }
  const target = predictBallAtCpu(state, ball);
  if (!target) return { hasTarget: false };
  const tx = clamp(target.x, CPU_RALLY_MIN_X, CPU_RALLY_MAX_X);
  const ty = clamp(target.y, PADDLE_REACH_TOP, PADDLE_REACH_BOTTOM);
  return { hasTarget: true, tx, ty };
}

/** Fill `f` (Float32Array(24)) with normalized features matching the schema. */
export function extractCpuMLFeatures(f, ball, cpu, player, state, hasTarget, tx, ty) {
  f[0] = ball.x / WIDTH;
  f[1] = ball.y / HEIGHT;
  const velFeat = 0.12 / COURT_SPEED_SCALE;
  f[2] = clamp(ball.vx * velFeat, -1, 1);
  f[3] = clamp(ball.vy * velFeat, -1, 1);
  f[4] = cpu.x / WIDTH;
  f[5] = cpu.y / HEIGHT;
  f[6] = player.x / WIDTH;
  f[7] = player.y / HEIGHT;
  if (hasTarget) {
    f[8] = clamp((tx - cpu.x) / WIDTH, -1, 1);
    f[9] = clamp((ty - cpu.y) / HEIGHT, -1, 1);
  } else {
    f[8] = 0;
    f[9] = 0;
  }
  f[10] = hasTarget ? 1 : 0;
  f[11] = ball.lastHitter === "player" ? 1 : 0;
  f[12] = ball.lastHitter === "cpu" ? 1 : 0;
  f[13] = ball.lastHitter == null ? 1 : 0;
  f[14] = clamp(ball.bouncesSinceHit / 3, 0, 1);
  f[15] = clamp(state.bouncesNeeded / 2, 0, 1);
  f[16] = mustLetBallBounceBeforeReturn(state, ball) ? 1 : 0;
  f[17] = ball.vx > 0.5 && ball.lastHitter !== "cpu" ? 1 : 0;
  f[18] = clamp((ball.x - cpu.x) / WIDTH, -1, 1);
  f[19] = clamp((ball.y - cpu.y) / HEIGHT, -1, 1);
  f[20] = clamp(state.hitsThisRally / 14, 0, 1);
  f[21] = clamp((player.x - NET_X) / WIDTH, -1, 1);
  f[22] = isInCpuKitchen(cpu.x) ? 1 : 0;
  f[23] = paddleHitHeight01(cpu.y);
  return f;
}

/**
 * Expert action label: best 5x5 paddle delta to reach (tx, ty) with kitchen
 * margin and neutral-court bias mixed in. Returns an action index in
 * [0, ACTION_DIM).
 */
export function chooseExpertCpuAction(state, ball, cpu, tx, ty, stepSpeed, stepSpeedY = stepSpeed) {
  const needsBounceFirst = mustLetBallBounceBeforeReturn(state, ball);
  const kitchenCenterX = NET_X + KITCHEN_HALF * 0.5;
  let bestAction = 0;
  let bestScore = Infinity;

  for (let a = 0; a < ACTION_DIM; a += 1) {
    const [mx, my] = actionToDelta(a);
    const nx = clamp(cpu.x + mx * stepSpeed, CPU_RALLY_MIN_X, CPU_RALLY_MAX_X);
    const ny = clamp(cpu.y + my * stepSpeedY, PADDLE_REACH_TOP, PADDLE_REACH_BOTTOM);
    const dx = nx - tx;
    const dy = ny - ty;
    let score = dx * dx + dy * dy;

    if (!needsBounceFirst && nx < NET_X + KITCHEN_HALF) {
      const kitchenDepth = (NET_X + KITCHEN_HALF - nx) / KITCHEN_HALF;
      score += 140 * kitchenDepth * kitchenDepth;
    }

    const centerBias = (nx - kitchenCenterX) / (CPU_RALLY_MAX_X - CPU_RALLY_MIN_X);
    score += 2.5 * centerBias * centerBias;

    if (score < bestScore) {
      bestScore = score;
      bestAction = a;
    }
  }
  return bestAction;
}

/** Difficulty score for the CPU's current state, used to upweight rare/hard frames. */
export function estimateHardStateScore(state, ball, cpu, tx, ty) {
  const speed01 = clamp(Math.hypot(ball.vx, ball.vy) / (9.2 * COURT_SPEED_SCALE), 0, 1);
  const sidelineDist = Math.min(Math.abs(tx - CPU_RALLY_MIN_X), Math.abs(CPU_RALLY_MAX_X - tx));
  const edge01 = 1 - clamp(sidelineDist / (46 * (HALF_COURT_DEPTH / LEGACY_HALF_COURT_DEPTH)), 0, 1);
  const lowContact01 = clamp((ty - PADDLE_REACH_TOP) / (MAX_CONTACT_HEIGHT * (40 / 110)), 0, 1);
  const volleyPressure01 = mustLetBallBounceBeforeReturn(state, ball) ? 0 : 1;
  const kitchenPressure01 = cpu.x < NET_X + KITCHEN_HALF ? 1 : 0;
  return clamp(
    0.38 * speed01 +
      0.25 * edge01 +
      0.2 * volleyPressure01 +
      0.1 * kitchenPressure01 +
      0.07 * (1 - lowContact01),
    0,
    1
  );
}
