/**
 * Shot intent: paddle height → Smash / Drive / Lob.
 * Soft mode (Q toggle): kitchen → Dink, box → Drop at any paddle height.
 */

import {
  CPU_MODE_PRO,
  HEIGHT,
  KITCHEN_HALF,
  NET_X,
  POWER_MAX,
  POWER_MIN,
  SHOT_PROFILES,
  WIDTH
} from "./constants.js";
import { angleFromPaddleY, paddleHitHeight01 } from "./physics.js";
import {
  isInCpuKitchen,
  isInPlayerKitchen,
  softShotFromCourtPosition
} from "./rules.js";
import { clamp } from "./rng.js";

/** Paddle-height bands (0 = low, 1 = high): bottom ¼ lob, middle ½ drive, top ¼ smash. */
export const PADDLE_HIGH_BAND = 3 / 4;
export const PADDLE_MID_BAND = 1 / 4;

export const SOFT_SHOT_DROP = "Drop";
export const SOFT_SHOT_DINK = "Dink";

/** Sample tuned launch parameters for a shot label. */
export function sampleShotParams(intent, rng) {
  const profile = SHOT_PROFILES[intent];
  if (!profile) {
    return { power: 50, angle: 35 };
  }
  return {
    power: rng.randInt(profile.powerMin, profile.powerMax),
    angle: rng.randInt(profile.angleMin, profile.angleMax)
  };
}

/**
 * Smash / Drive / Lob from paddle height only.
 * @param {number} h paddleHitHeight01
 */
export function intentFromPaddleHeight(h) {
  if (h >= PADDLE_HIGH_BAND) return "Smash";
  if (h >= PADDLE_MID_BAND) return "Drive";
  return "Lob";
}

/**
 * Player return: soft on → kitchen Dink / box Drop (any paddle height).
 * Soft off → Smash / Drive / Lob from paddle height.
 */
export function resolvePlayerReturnIntent(p, state) {
  if (state.playerSoftShotQueued) {
    return softShotFromCourtPosition("player", p.x);
  }
  return intentFromPaddleHeight(paddleHitHeight01(p.y));
}

/**
 * CPU shot picker (Pro oracle may override).
 */
export function pickShotIntent(who, p, ball, state, rng, oracleFn = null, skill01 = 0) {
  if (who === "cpu" && state.cpuMode === CPU_MODE_PRO && typeof oracleFn === "function") {
    const oracleShot = oracleFn(p.x / WIDTH, p.y / HEIGHT);
    if (oracleShot && rng.random() < 0.7) return oracleShot;
  }

  const h = paddleHitHeight01(p.y);
  const inFrontCourt =
    who === "cpu"
      ? p.x < NET_X + KITCHEN_HALF + 4 || isInCpuKitchen(p.x)
      : p.x > NET_X - KITCHEN_HALF - 4 || isInPlayerKitchen(p.x);

  if (inFrontCourt && h < PADDLE_MID_BAND) {
    return softShotFromCourtPosition(who, p.x);
  }
  return intentFromPaddleHeight(h);
}

/**
 * Apply a shot intent label to base (power, angle), returning final values.
 */
export function applyShotIntent(who, p, basePower, baseAngle, forcedIntent, ball, state, rng, oracleFn, skill01) {
  const intent =
    forcedIntent || pickShotIntent(who, p, ball, state, rng, oracleFn, skill01);
  const skill = who === "cpu" ? clamp(skill01 ?? 0, 0, 1) : 0.65;
  const tuned = sampleShotParams(intent, rng);
  let power = tuned.power;
  let angle = tuned.angle;

  if (intent !== "Smash" && who !== "player") {
    const paddleAngle = angleFromPaddleY(p, who);
    const h = paddleHitHeight01(p.y);
    const w = intent === "Lob" || intent === "Dink" ? 0.28 + 0.12 * (1 - h) : 0.18 + 0.1 * h;
    angle = angle * (1 - w) + paddleAngle * w;
    const noise = 5.4 - 4.0 * skill;
    angle += rng.spread(noise);
  }

  return { power: clamp(power, POWER_MIN, POWER_MAX), angle, intent };
}

export { KITCHEN_HALF as _KITCHEN_HALF };
