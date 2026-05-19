/**
 * Pure contact resolution: when ball intersects a paddle, build the return
 * (or detect kitchen/volley faults) and update ball state in place.
 *
 * Returns an event object describing what happened so the caller (browser or
 * offline simulator) can update HUD/messages and dataset rows accordingly.
 */

import { adjustAngleToClearNet, launchBall, returnPowerFromPaddleY, angleFromPaddleY } from "./physics.js";
import {
  isInCpuKitchen,
  isInPlayerKitchen,
  mustLetBallBounceBeforeReturn,
  shotTypeFromPower
} from "./rules.js";
import { applyShotIntent, resolvePlayerReturnIntent } from "./shots.js";

/**
 * @returns {{type:"fault", who, winner, reason}|{type:"return", who, power, angle, intent}}
 */
export function handleContact(state, ball, player, cpu, who, rng, oracleFn = null, skill01 = 0) {
  const p = who === "player" ? player : cpu;
  if (who === "player") player.swingPhase = 1;
  else cpu.swingPhase = 1;

  if (mustLetBallBounceBeforeReturn(state, ball)) {
    const winner = who === "player" ? "cpu" : "player";
    return { type: "fault", who, winner, reason: "volley_before_bounce" };
  }

  const inKitchen =
    (who === "player" && isInPlayerKitchen(p.x)) ||
    (who === "cpu" && isInCpuKitchen(p.x));
  const isVolley = ball.bouncesSinceHit === 0;
  if (inKitchen && isVolley) {
    const winner = who === "player" ? "cpu" : "player";
    return { type: "fault", who, winner, reason: "kitchen_volley" };
  }

  let power = returnPowerFromPaddleY(p.y);
  let angle = angleFromPaddleY(p, who);
  const forcedIntent =
    who === "cpu" ? state.cpuPlannedShotType : resolvePlayerReturnIntent(p, state);
  const pick = applyShotIntent(who, p, power, angle, forcedIntent, ball, state, rng, oracleFn, skill01);
  power = pick.power;
  angle = pick.angle;
  const shotType = pick.intent;

  if (who === "cpu") state.cpuPlannedShotType = null;

  const useNetAssist = pick.intent !== "Smash";
  if (useNetAssist) {
    angle = adjustAngleToClearNet(rng, who, power, angle, p.y, true, ball.x, ball.y);
  }
  launchBall(ball, who, power, angle, p.y, true);
  if (state.bouncesNeeded > 0) state.bouncesNeeded -= 1;
  state.hitsThisRally += 1;
  state.lastShotType = shotType;
  state.contactCooldown = 14;
  return { type: "return", who, power, angle, intent: shotType };
}

export { shotTypeFromPower };
