/**
 * Single ball-physics step shared by browser and offline simulator.
 *
 * Mutates `state` and `ball` in place. Returns an event object describing
 * the outcome of the step so callers can update UI/messages or dataset
 * trackers without reaching into module-level globals.
 *
 * Contract:
 * - Caller is responsible for placing the ball at the server position when
 *   `state.awaitingServe` is true (this function only `return`s in that case).
 * - On a rally-ending fault/score, return value is `{ type: "rallyEnd", winner, reason }`.
 *   The caller invokes `endRally` from rally.js to apply scoring/serve transitions.
 * - On contact during the step, return value is `{ type: "contact", contact }` where
 *   `contact` is the contact result (see contact.js).
 * - Otherwise return value is `{ type: "step" }`.
 */

import {
  COURT_LEFT,
  COURT_RIGHT,
  GROUND_Y,
  NET_PLANE_HALF_WIDTH,
  NET_TAPE_CLIP_PX,
  NET_TOP,
  NET_X,
  WIDTH
} from "./constants.js";
import { integrateBallVelocity, applyFloorBounce, applyFloorRoll } from "./ballMotion.js";
import { handleContact } from "./contact.js";
import { paddleApproachOk, paddleContactTAlongStep } from "./physics.js";
import {
  isInCpuKitchen,
  isInPlayerKitchen,
  mustLetBallBounceBeforeReturn
} from "./rules.js";
import { clamp } from "./rng.js";

/**
 * @returns {object} Step event:
 *   - `{type:"idle"}` when awaiting serve
 *   - `{type:"step"}` normal step with no events
 *   - `{type:"contact", contact}` on a paddle contact
 *   - `{type:"rallyEnd", winner, reason}` on rally end
 */
export function physicsStep(state, ball, player, cpu, rng, oracleFn = null, skill01 = 0) {
  if (state.awaitingServe) {
    return { type: "idle" };
  }

  if (state.contactCooldown > 0) state.contactCooldown -= 1;

  const prevX = ball.x;
  const prevY = ball.y;
  ({ vx: ball.vx, vy: ball.vy } = integrateBallVelocity(ball.vx, ball.vy));
  ball.x += ball.vx;
  ball.y += ball.vy;

  ball.trail.push({ x: ball.x, y: ball.y });
  if (ball.trail.length > 24) ball.trail.shift();

  if (state.contactCooldown === 0) {
    const segDx = ball.x - prevX;
    const mostlyVertical = Math.abs(segDx) < 0.12;

    let best = null;
    if (ball.lastHitter !== "player") {
      const tP = paddleContactTAlongStep(prevX, prevY, ball.x, ball.y, player);
      if (tP != null && paddleApproachOk("player", segDx) && (mostlyVertical || ball.vx <= 0)) {
        best = { who: "player", t: tP };
      }
    }
    if (ball.lastHitter !== "cpu" && !mustLetBallBounceBeforeReturn(state, ball)) {
      const tC = paddleContactTAlongStep(prevX, prevY, ball.x, ball.y, cpu);
      if (tC != null && paddleApproachOk("cpu", segDx) && (mostlyVertical || ball.vx >= 0)) {
        const cand = { who: "cpu", t: tC };
        if (!best || cand.t < best.t - 1e-6) best = cand;
      }
    }

    if (best) {
      const segDy = ball.y - prevY;
      ball.x = prevX + best.t * (ball.x - prevX);
      ball.y = prevY + best.t * segDy;
      const contact = handleContact(state, ball, player, cpu, best.who, rng, oracleFn, skill01);
      if (contact.type === "fault") {
        return { type: "rallyEnd", winner: contact.winner, reason: contact.reason };
      }
      return { type: "contact", contact };
    }
  }

  const crossedNetPlane = (prevX - NET_X) * (ball.x - NET_X) <= 0 && prevX !== ball.x;
  const segmentT = crossedNetPlane ? (NET_X - prevX) / (ball.x - prevX) : 0;
  const yAtNetCross = crossedNetPlane ? prevY + (ball.y - prevY) * clamp(segmentT, 0, 1) : ball.y;
  const inNetBodyNow =
    ball.x > NET_X - NET_PLANE_HALF_WIDTH &&
    ball.x < NET_X + NET_PLANE_HALF_WIDTH &&
    ball.y >= NET_TOP &&
    ball.y <= GROUND_Y;
  const hitNetByCrossing = crossedNetPlane && yAtNetCross >= NET_TOP && yAtNetCross <= GROUND_Y;
  if (inNetBodyNow || hitNetByCrossing) {
    const hitsTapeTop = yAtNetCross <= NET_TOP + NET_TAPE_CLIP_PX && ball.vy > 0;
    if (hitsTapeTop) {
      ball.x = NET_X + (ball.lastHitter === "player" ? 1 : -1) * 5;
      ball.y = NET_TOP - 1;
      ball.vy = -Math.max(0.8, Math.abs(ball.vy) * 0.45);
      ball.vx *= 0.82;
    } else {
      const winner = ball.lastHitter === "player" ? "cpu" : "player";
      return { type: "rallyEnd", winner, reason: "net_hit" };
    }
  }

  if (ball.x < -20 || ball.x > WIDTH + 20) {
    if (ball.lastHitter && ball.bouncesSinceHit >= 1 && ball.firstBounceInAfterHit) {
      const winner = ball.lastHitter;
      return { type: "rallyEnd", winner, reason: "off_court_after_bounce_in" };
    }
    const winner = ball.lastHitter === "player" ? "cpu" : "player";
    return { type: "rallyEnd", winner, reason: "off_court" };
  }

  if (ball.y >= GROUND_Y) {
    ball.y = GROUND_Y;
    if (ball.x < COURT_LEFT || ball.x > COURT_RIGHT) {
      if (ball.lastHitter && ball.bouncesSinceHit >= 1 && ball.firstBounceInAfterHit) {
        const winner = ball.lastHitter;
        return { type: "rallyEnd", winner, reason: "second_bounce_out_after_in" };
      }
      const winner = ball.lastHitter === "player" ? "cpu" : "player";
      return { type: "rallyEnd", winner, reason: "out" };
    }
    const bounce = applyFloorBounce(ball.vx, ball.vy);
    if (bounce) {
      ball.vy = bounce.vy;
      ball.vx = bounce.vx;
      ball.bouncesSinceHit += 1;
      ball.bouncedOnSide = ball.x < NET_X ? "player" : "cpu";

      if (ball.bouncesSinceHit === 1 && ball.bouncedOnSide === ball.lastHitter) {
        const winner = ball.lastHitter === "player" ? "cpu" : "player";
        return { type: "rallyEnd", winner, reason: "didnt_clear_net" };
      }

      if (state.hitsThisRally === 1 && ball.bouncesSinceHit === 1) {
        const serveInOppKitchen =
          (ball.lastHitter === "player" &&
            ball.bouncedOnSide === "cpu" &&
            isInCpuKitchen(ball.x)) ||
          (ball.lastHitter === "cpu" &&
            ball.bouncedOnSide === "player" &&
            isInPlayerKitchen(ball.x));
        if (serveInOppKitchen) {
          const winner = ball.lastHitter === "player" ? "cpu" : "player";
          return { type: "rallyEnd", winner, reason: "serve_in_kitchen" };
        }
      }

      if (ball.bouncesSinceHit === 1) {
        ball.firstBounceInAfterHit = true;
      }

      if (ball.bouncesSinceHit >= 2) {
        const loser = ball.bouncedOnSide;
        const winner = loser === "player" ? "cpu" : "player";
        return { type: "rallyEnd", winner, reason: "second_bounce" };
      }
    } else {
      ball.vy = 0;
      ball.vx = applyFloorRoll(ball.vx);
      if (Math.abs(ball.vx) < 0.4) {
        const loser = ball.x < NET_X ? "player" : "cpu";
        const winner = loser === "player" ? "cpu" : "player";
        return { type: "rallyEnd", winner, reason: "ball_died" };
      }
    }
  }

  return { type: "step" };
}
