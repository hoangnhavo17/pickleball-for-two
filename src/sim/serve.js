/**
 * Serve flows shared between browser and offline simulator.
 *
 * The browser separately drives the oscillator/UI, while the offline runner
 * calls these directly with chosen parameters. Each function mutates ball
 * state in place and returns metadata for HUD/dataset bookkeeping.
 */

import {
  CPU_SERVE_DEFAULT_X,
  CPU_SERVE_DISPLAY_POWER_MAX,
  CPU_SERVE_DISPLAY_POWER_MIN,
  GROUND_Y,
  PADDLE_PLAY_H,
  PLAYER_SERVE_DEFAULT_X,
  PADDLE_PLAY_W,
  POWER_MAX,
  SERVE_ANGLE_PRESETS
} from "./constants.js";
import { adjustAngleToClearNet, launchBall } from "./physics.js";
import { clamp } from "./rng.js";
import { shotTypeFromPower } from "./rules.js";

function standBehindBaselineY() {
  return GROUND_Y - (PADDLE_PLAY_H / 2 + 2);
}

/**
 * One-shot placement when a new serve is pending: server centered in their
 * behind-baseline strip, receiver on the opposite baseline.
 */
export function resetServePositions(state, player, cpu) {
  const y0 = standBehindBaselineY();
  if (state.server === "player") {
    player.x = PLAYER_SERVE_DEFAULT_X;
    player.y = y0;
    cpu.x = CPU_SERVE_DEFAULT_X;
    cpu.y = y0;
  } else {
    cpu.x = CPU_SERVE_DEFAULT_X;
    cpu.y = y0;
    player.x = PLAYER_SERVE_DEFAULT_X;
    player.y = y0;
  }
}

export function placeBallOnServer(state, ball, player, cpu) {
  const server = state.server === "player" ? player : cpu;
  const dir = state.server === "player" ? 1 : -1;
  ball.x = server.x + dir * (PADDLE_PLAY_W + 2);
  ball.y = server.y;
  ball.vx = 0;
  ball.vy = 0;
  ball.lastHitter = null;
  ball.bouncesSinceHit = 0;
  ball.bouncedOnSide = null;
  ball.firstBounceInAfterHit = false;
  ball.trail.length = 0;
}

export function pickCpuServeDisplayPower(rng, skill01 = 0) {
  const skill = clamp(skill01, 0, 1);
  const min = Math.round(CPU_SERVE_DISPLAY_POWER_MIN + 8 * skill);
  const max = Math.round(CPU_SERVE_DISPLAY_POWER_MAX + 4 * skill);
  return rng.randInt(min, Math.min(POWER_MAX, max));
}

export function pickCpuServeAngle(rng, skill01 = 0) {
  const skill = clamp(skill01, 0, 1);
  const centerPreset =
    rng.random() < 0.7 + 0.2 * skill
      ? SERVE_ANGLE_PRESETS[rng.randInt(1, 2)]
      : SERVE_ANGLE_PRESETS[rng.randInt(0, SERVE_ANGLE_PRESETS.length - 1)];
  const jitter = Math.round(2 - 1.2 * skill);
  return centerPreset + rng.randInt(-jitter, jitter);
}

/**
 * Headless training serve: bias the distribution toward harder cases while
 * keeping a reasonable mix.
 */
export function pickPlayerTrainingServe(rng, serveAngleOptions) {
  const displayPower = rng.random() < 0.4 ? rng.randInt(82, 100) : rng.randInt(42, 98);
  const options =
    serveAngleOptions && serveAngleOptions.length === 4
      ? serveAngleOptions
      : [18, 30, 42, 54];
  const pickedAngle =
    rng.random() < 0.35
      ? rng.random() < 0.5
        ? rng.randInt(8, 20)
        : rng.randInt(52, 68)
      : options[rng.randInt(0, options.length - 1)];
  return { displayPower, pickedAngle };
}

/**
 * Player serve from a chosen angle/power. Returns metadata with the resolved
 * shot type and final launch parameters.
 */
export function serveByPlayer(state, ball, player, cpu, displayPower, angle, rng) {
  player.swingPhase = 1;
  ball.x = player.x + (PADDLE_PLAY_W + 2);
  ball.y = player.y;
  const finalAngle = adjustAngleToClearNet(rng, "player", displayPower, angle, player.y, false, ball.x, ball.y);
  launchBall(ball, "player", displayPower, finalAngle, player.y, false);
  state.mlEpisodeId += 1;
  state.awaitingServe = false;
  state.rallyActive = true;
  state.bouncesNeeded = 2;
  state.hitsThisRally = 1;
  state.lastShotType = shotTypeFromPower(displayPower);
  state.contactCooldown = 14;
  return {
    who: "player",
    displayPower,
    angle: finalAngle,
    shotType: state.lastShotType
  };
}

export function serveByCpu(state, ball, player, cpu, rng, skill01 = 0) {
  cpu.swingPhase = 1;
  const displayPower = pickCpuServeDisplayPower(rng, skill01);
  ball.x = cpu.x - (PADDLE_PLAY_W + 2);
  ball.y = cpu.y;
  const angle = pickCpuServeAngle(rng, skill01);
  const finalAngle = adjustAngleToClearNet(rng, "cpu", displayPower, angle, cpu.y, false, ball.x, ball.y);
  launchBall(ball, "cpu", displayPower, finalAngle, cpu.y, false);
  state.mlEpisodeId += 1;
  state.awaitingServe = false;
  state.rallyActive = true;
  state.bouncesNeeded = 2;
  state.hitsThisRally = 1;
  state.lastShotType = shotTypeFromPower(displayPower);
  state.contactCooldown = 14;
  return {
    who: "cpu",
    displayPower,
    angle: finalAngle,
    shotType: state.lastShotType
  };
}

/**
 * Headless training-style player serve, biased toward hard coverage.
 */
export function serveByPlayerTraining(state, ball, player, cpu, rng) {
  const { displayPower, pickedAngle } = pickPlayerTrainingServe(rng, state.serveAngleOptions);
  return serveByPlayer(state, ball, player, cpu, displayPower, pickedAngle, rng);
}
