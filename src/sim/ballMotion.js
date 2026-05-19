/**
 * Shared per-frame ball integration and floor bounce.
 * Used by step.js, trajectory predictors, and offline physics tuners.
 *
 * Floor COR follows USAP ball bounce limits (see constants.js BALL_COURT_COR_*).
 */

import {
  AIR_DRAG,
  BALL_COURT_COR_MAX,
  BALL_COURT_COR_MIN,
  BALL_GRAVITY,
  FLOOR_BOUNCE_MIN_VY,
  FLOOR_BOUNCE_SPEED_REF,
  FLOOR_BOUNCE_VX_MUL,
  FLOOR_BOUNCE_VY_MUL,
  FLOOR_BOUNCE_VY_MUL_MAX,
  FLOOR_ROLL_VX_MUL
} from "./constants.js";
import { clamp } from "./rng.js";

/** Advance velocity one frame (air drag + gravity). */
export function integrateBallVelocity(vx, vy, opts = {}) {
  const drag = opts.airDrag ?? AIR_DRAG;
  const gravity = opts.ballGravity ?? BALL_GRAVITY;
  return {
    vx: vx * drag,
    vy: vy * drag + gravity
  };
}

/**
 * Court bounce: COR = rebound speed / inbound speed on the surface normal.
 * Direct vertical impacts use BALL_COURT_COR_MIN..MAX (USAP 30–34 in off 78 in drop).
 * Glancing impacts reduce COR (less normal component).
 */
export function courtBounceCor(vx, vy, opts = {}) {
  const minVy = opts.floorMinVy ?? FLOOR_BOUNCE_MIN_VY;
  const corMin = opts.floorCorMin ?? FLOOR_BOUNCE_VY_MUL ?? BALL_COURT_COR_MIN;
  const corMax = opts.floorCorMax ?? FLOOR_BOUNCE_VY_MUL_MAX ?? BALL_COURT_COR_MAX;
  const speedRef = opts.floorSpeedRef ?? FLOOR_BOUNCE_SPEED_REF;

  const inbound = Math.abs(vy);
  if (inbound <= minVy) return 0;

  const total = Math.hypot(vx, vy);
  const normalWeight = clamp(inbound / Math.max(total, 0.01), 0.2, 1);
  const speed01 = clamp((inbound - minVy) / Math.max(speedRef - minVy, 0.01), 0, 1);
  const corDirect = corMin + (corMax - corMin) * speed01;
  return corMin + (corDirect - corMin) * normalWeight;
}

/**
 * Floor bounce on GROUND_Y. Returns null if the ball rolls to rest (no bounce).
 * @returns {{ vx: number, vy: number, bounced: boolean } | null}
 */
export function applyFloorBounce(vx, vy, opts = {}) {
  const minVy = opts.floorMinVy ?? FLOOR_BOUNCE_MIN_VY;
  const vxMul = opts.floorVxMul ?? FLOOR_BOUNCE_VX_MUL;
  const inbound = Math.abs(vy);
  if (inbound <= minVy) {
    return null;
  }
  const cor = courtBounceCor(vx, vy, opts);
  if (cor <= 0) {
    return null;
  }
  return {
    vx: vx * vxMul,
    vy: -vy * cor,
    bounced: true
  };
}

/** Roll friction when vertical speed is below bounce threshold. */
export function applyFloorRoll(vx, opts = {}) {
  const vxMul = opts.floorRollVxMul ?? FLOOR_ROLL_VX_MUL;
  return vx * vxMul;
}
