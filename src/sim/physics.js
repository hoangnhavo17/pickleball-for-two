/**
 * Pure physics helpers used by both browser game and offline simulator.
 *
 * No DOM, no globals, no Math.random. All randomness is injected via the
 * `rng` argument so callers can switch between Math.random and a seeded
 * Mulberry32 generator.
 */

import { integrateBallVelocity, applyFloorBounce, applyFloorRoll } from "./ballMotion.js";
import {
  BALL_R,
  CONTACT_PAD,
  COURT_LEFT,
  COURT_RIGHT,
  CPU_RALLY_MAX_X,
  CPU_RETURN_ANGLE_MAX,
  COURT_SPEED_SCALE,
  GROUND_Y,
  MAX_CONTACT_HEIGHT,
  NET_CLEAR_ANGLE_MAX,
  NET_PLANE_HALF_WIDTH,
  NET_TAPE_CLIP_PX,
  NET_TOP,
  NET_X,
  PADDLE_H,
  PADDLE_PLAY_H,
  PADDLE_PLAY_W,
  PADDLE_REACH_BOTTOM,
  PADDLE_REACH_TOP,
  PLAYER_RETURN_ANGLE_MAX,
  POWER_MAX,
  POWER_MIN,
  RETURN_POSITION_WEIGHT,
  RETURN_POWER_BASE,
  RETURN_SPEED_MULT,
  SERVE_POWER_BASE
} from "./constants.js";
import { clamp } from "./rng.js";
import { mustLetBallBounceBeforeReturn, powerStrength01 } from "./rules.js";

export function servePowerEffective(displayChoice) {
  const norm01 = powerStrength01(displayChoice);
  return SERVE_POWER_BASE + norm01 * (POWER_MAX - SERVE_POWER_BASE);
}

export function speedFromPowerOnly(power, isReturn) {
  let u = isReturn
    ? powerStrength01(power)
    : (servePowerEffective(power) - SERVE_POWER_BASE) / (POWER_MAX - SERVE_POWER_BASE);
  if (!isReturn) {
    u = Math.min(1, u + 0.11 * (1 - u) * (1 - u));
  }
  let s = 1.52 + 5.85 * u + 5.5 * u * u;
  if (!isReturn) s *= 1.035;
  else s *= 1.08;
  return s * COURT_SPEED_SCALE;
}

export function paddleHitHeight01(paddleY) {
  return clamp((GROUND_Y - paddleY) / MAX_CONTACT_HEIGHT, 0, 1);
}

export function returnPowerFromPaddleY(paddleY) {
  const t = paddleHitHeight01(paddleY);
  return clamp(
    Math.round(RETURN_POWER_BASE + t * RETURN_POSITION_WEIGHT),
    RETURN_POWER_BASE,
    POWER_MAX
  );
}

export function angleFromPaddleY(p, who = "player") {
  const heightAbove = clamp(GROUND_Y - p.y, 0, MAX_CONTACT_HEIGHT);
  const maxAngle = who === "cpu" ? CPU_RETURN_ANGLE_MAX : PLAYER_RETURN_ANGLE_MAX;
  return clamp(60 - (heightAbove / MAX_CONTACT_HEIGHT) * 40, 20, maxAngle);
}

export function computeLaunchVelocity(who, power, angleDeg, paddleY, isReturn) {
  const p = clamp(power, POWER_MIN, POWER_MAX);
  let speed = speedFromPowerOnly(p, isReturn);
  if (isReturn) speed *= RETURN_SPEED_MULT;
  const dir = who === "player" ? 1 : -1;
  const rad = (angleDeg * Math.PI) / 180;
  const h = paddleHitHeight01(paddleY);
  const d0 = isReturn ? 0.97 : 0.87;
  const k = isReturn ? 0.39 : 0.45;
  const distMul = d0 + k * h;
  const arcMul = d0 + k * (1 - h);
  return {
    vx: dir * Math.cos(rad) * speed * distMul,
    vy: -Math.sin(rad) * speed * arcMul
  };
}

/**
 * Simulate a return until first bounce. Used for shot tuning and diagnostics.
 * @returns {{ landX: number, landY: number, hitsNet: boolean, inBounds: boolean } | null}
 */
export function simulateReturnLanding(startX, startY, paddleY, who, power, angleDeg) {
  const v = computeLaunchVelocity(who, power, angleDeg, paddleY, true);
  let x = startX;
  let y = startY;
  let vx = v.vx;
  let vy = v.vy;
  let hitsNet = false;
  for (let step = 0; step < 480; step += 1) {
    const prevX = x;
    ({ vx, vy } = integrateBallVelocity(vx, vy));
    x += vx;
    y += vy;
    const crossedNet = (prevX - NET_X) * (x - NET_X) <= 0 && prevX !== x;
    const inNetSlab =
      x > NET_X - NET_PLANE_HALF_WIDTH &&
      x < NET_X + NET_PLANE_HALF_WIDTH &&
      y >= NET_TOP &&
      y <= GROUND_Y;
    if (inNetSlab || (crossedNet && y >= NET_TOP && y <= GROUND_Y)) {
      hitsNet = true;
    }
    if (y >= GROUND_Y) {
      return { landX: x, landY: GROUND_Y, hitsNet, inBounds: x >= COURT_LEFT && x <= COURT_RIGHT };
    }
  }
  return null;
}

export function trajectoryHitsNetBody(startX, startY, vx, vy) {
  let x = startX;
  let y = startY;
  let vx0 = vx;
  let vy0 = vy;
  for (let step = 0; step < 520; step += 1) {
    ({ vx: vx0, vy: vy0 } = integrateBallVelocity(vx0, vy0));
    x += vx0;
    y += vy0;
    if (x > NET_X - NET_PLANE_HALF_WIDTH && x < NET_X + NET_PLANE_HALF_WIDTH && y >= NET_TOP && y <= GROUND_Y) {
      const tapeClip = y <= NET_TOP + NET_TAPE_CLIP_PX && vy0 > 0;
      if (!tapeClip) return true;
    }
    if (y >= GROUND_Y) return false;
    if (y < NET_TOP - PADDLE_H / 3 && (x < NET_X - PADDLE_H / 2 || x > NET_X + PADDLE_H / 2)) return false;
  }
  return false;
}

export function netClearAssistProbability(power, isReturn) {
  if (!isReturn) {
    const eff = servePowerEffective(power);
    if (eff >= 56) return 1;
    const u = (eff - SERVE_POWER_BASE) / (POWER_MAX - SERVE_POWER_BASE);
    if (u >= 0.14) return 1;
    return 0.2 + 0.8 * (u / 0.14) * (u / 0.14);
  }
  const u = powerStrength01(power);
  if (u >= 0.24) return 1;
  return 0.18 + 0.82 * (u / 0.24) * (u / 0.24);
}

export function adjustAngleToClearNet(rng, who, power, angleDeg, paddleY, isReturn, startX, startY) {
  if (rng.random() >= netClearAssistProbability(power, isReturn)) return angleDeg;
  let a = angleDeg;
  while (a < NET_CLEAR_ANGLE_MAX) {
    const v = computeLaunchVelocity(who, power, a, paddleY, isReturn);
    if (!trajectoryHitsNetBody(startX, startY, v.vx, v.vy)) return a;
    a += 1.2;
  }
  return NET_CLEAR_ANGLE_MAX;
}

export function canClearNetAndLandIn(ball, opponentSide, who, power, angle, paddleY) {
  const v = computeLaunchVelocity(who, power, angle, paddleY, true);
  if (trajectoryHitsNetBody(ball.x, ball.y, v.vx, v.vy)) return false;
  let x = ball.x;
  let y = ball.y;
  let vx = v.vx;
  let vy = v.vy;
  for (let step = 0; step < 380; step += 1) {
    ({ vx, vy } = integrateBallVelocity(vx, vy));
    x += vx;
    y += vy;
    if (y >= GROUND_Y) {
      if (x < COURT_LEFT || x > COURT_RIGHT) return false;
      if (opponentSide === "cpu") return x > NET_X + 6;
      return x < NET_X - 6;
    }
  }
  return false;
}

export function paddleHalfExtents() {
  const hw = PADDLE_PLAY_W / 2 + CONTACT_PAD + BALL_R;
  const hh = PADDLE_PLAY_H / 2 + CONTACT_PAD + BALL_R;
  return { hw, hh };
}

export function paddleContactsAt(px, py, p) {
  const { hw, hh } = paddleHalfExtents();
  return Math.abs(px - p.x) < hw && Math.abs(py - p.y) < hh;
}

export function paddleContacts(ball, p) {
  return paddleContactsAt(ball.x, ball.y, p);
}

export function paddleApproachOk(who, segDx) {
  if (Math.abs(segDx) < 0.12) return true;
  return who === "player" ? segDx < 0 : segDx > 0;
}

/**
 * Earliest t in [0,1] where segment (x0,y0)->(x1,y1) enters expanded paddle AABB,
 * or null. Liang–Barsky-style slab clipping on the segment parameter.
 */
export function segmentPaddleHitT(x0, y0, x1, y1, p) {
  const { hw, hh } = paddleHalfExtents();
  const minX = p.x - hw;
  const maxX = p.x + hw;
  const minY = p.y - hh;
  const maxY = p.y + hh;
  const dx = x1 - x0;
  const dy = y1 - y0;
  let t0 = 0;
  let t1 = 1;

  function clipSlab(p0, d, lo, hi) {
    if (Math.abs(d) < 1e-8) {
      if (p0 < lo || p0 > hi) return false;
      return true;
    }
    const inv = 1 / d;
    let tNear = (lo - p0) * inv;
    let tFar = (hi - p0) * inv;
    if (tNear > tFar) {
      const tmp = tNear;
      tNear = tFar;
      tFar = tmp;
    }
    t0 = Math.max(t0, tNear);
    t1 = Math.min(t1, tFar);
    return t0 <= t1;
  }

  if (!clipSlab(x0, dx, minX, maxX)) return null;
  if (!clipSlab(y0, dy, minY, maxY)) return null;
  if (t0 > 1 || t1 < 0) return null;
  const tEnter = Math.max(0, t0);
  const tExit = Math.min(1, t1);
  if (tEnter > tExit) return null;
  return tEnter;
}

export function paddleContactTAlongStep(x0, y0, x1, y1, paddle) {
  let t = segmentPaddleHitT(x0, y0, x1, y1, paddle);
  if (t == null && paddleContactsAt(x1, y1, paddle)) t = 1;
  return t;
}

export function predictFirstGroundBouncePoint(ball) {
  let simX = ball.x;
  let simY = ball.y;
  let simVx = ball.vx;
  let simVy = ball.vy;
  for (let step = 0; step < 220; step += 1) {
    ({ vx: simVx, vy: simVy } = integrateBallVelocity(simVx, simVy));
    simX += simVx;
    simY += simVy;
    if (simY >= GROUND_Y) {
      if (simX < COURT_LEFT || simX > COURT_RIGHT) return null;
      return { x: simX, y: GROUND_Y - BALL_R * 3.5 };
    }
    if (simX < COURT_LEFT - 2 * PADDLE_H || simX > COURT_RIGHT + 2 * PADDLE_H) return null;
  }
  return null;
}

export function predictBallAtCpu(state, ball) {
  let simX = ball.x;
  let simY = ball.y;
  let simVx = ball.vx;
  let simVy = ball.vy;
  let bounced = ball.bouncesSinceHit;
  const needsBounceFirst = state.bouncesNeeded > 0 && bounced === 0;

  for (let step = 0; step < 220; step += 1) {
    ({ vx: simVx, vy: simVy } = integrateBallVelocity(simVx, simVy));
    simX += simVx;
    simY += simVy;

    if (simY >= GROUND_Y) {
      simY = GROUND_Y;
      const bounce = applyFloorBounce(simVx, simVy);
      if (bounce) {
        simVx = bounce.vx;
        simVy = bounce.vy;
        bounced += 1;
      } else {
        simVy = 0;
        simVx = applyFloorRoll(simVx);
        return null;
      }
    }

    if (simX < COURT_LEFT - 1.5 * PADDLE_H || simX > COURT_RIGHT + 1.5 * PADDLE_H) return null;

    const reachable =
      simX >= NET_X + 20 && simX <= CPU_RALLY_MAX_X && GROUND_Y - simY <= MAX_CONTACT_HEIGHT;
    const bounceOk = !needsBounceFirst || bounced > 0;
    if (reachable && bounceOk) {
      return { x: simX, y: simY };
    }
  }
  return null;
}

/** Mutates ball in place to launch it from `who` at the given parameters. */
export function launchBall(ball, who, power, angleDeg, paddleY, isReturn = false) {
  const v = computeLaunchVelocity(who, power, angleDeg, paddleY, isReturn);
  ball.vx = v.vx;
  ball.vy = v.vy;
  ball.lastHitter = who;
  ball.bouncesSinceHit = 0;
  ball.bouncedOnSide = null;
  ball.firstBounceInAfterHit = false;
}

export {
  mustLetBallBounceBeforeReturn,
  powerStrength01
};
