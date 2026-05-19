/**
 * Pure rule predicates and shot-type classification.
 */

import { COURT_LEFT, COURT_RIGHT, KITCHEN_HALF, NET_X, POWER_MAX, POWER_MIN } from "./constants.js";
import { clamp } from "./rng.js";

export function powerStrength01(p) {
  return clamp(p, POWER_MIN, POWER_MAX) / (POWER_MAX - POWER_MIN);
}

export function shotTypeFromPower(p) {
  const u = powerStrength01(p);
  if (u < 0.26) return "Dink";
  if (u < 0.48) return "Drop";
  if (u < 0.7) return "Drive";
  if (u < 0.88) return "Lob";
  return "Smash";
}

export function isInPlayerKitchen(x) {
  return x > NET_X - KITCHEN_HALF && x < NET_X;
}

export function isInCpuKitchen(x) {
  return x > NET_X && x < NET_X + KITCHEN_HALF;
}

/** Service box on player side (baseline through kitchen line, excluding NVZ). */
export function isInPlayerBox(x) {
  return x >= COURT_LEFT && x <= NET_X - KITCHEN_HALF;
}

/** Service box on CPU side (kitchen line through baseline, excluding NVZ). */
export function isInCpuBox(x) {
  return x >= NET_X + KITCHEN_HALF && x <= COURT_RIGHT;
}

/**
 * Soft shot from court depth: kitchen (NVZ) → Dink, service box → Drop.
 * @param {"player"|"cpu"} who
 */
export function softShotFromCourtPosition(who, x) {
  if (who === "player") {
    return isInPlayerKitchen(x) ? "Dink" : "Drop";
  }
  return isInCpuKitchen(x) ? "Dink" : "Drop";
}

export function mustLetBallBounceBeforeReturn(state, ball) {
  return state.bouncesNeeded > 0 && ball.bouncesSinceHit === 0;
}
