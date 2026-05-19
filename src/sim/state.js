/**
 * Pure state factories used by both browser and offline simulator.
 *
 * No DOM or storage access: just plain objects with the shape consumed by
 * the rest of src/sim/*.
 */

import {
  CPU_BASE_X,
  CPU_MODE_NORMAL,
  GROUND_Y,
  PADDLE_PLAY_H,
  PADDLE_PLAY_W,
  PLAYER_BASE_X,
  PLAYER_SERVE_DEFAULT_X
} from "./constants.js";

const paddleStandY = () => GROUND_Y - (PADDLE_PLAY_H / 2 + 2);

export function createPlayer() {
  return { x: PLAYER_SERVE_DEFAULT_X, y: paddleStandY(), swingPhase: 0 };
}

export function createCpu() {
  return { x: CPU_BASE_X, y: paddleStandY(), swingPhase: 0 };
}

export function createBall() {
  return {
    x: PLAYER_SERVE_DEFAULT_X + PADDLE_PLAY_W + 2,
    y: paddleStandY(),
    vx: 0,
    vy: 0,
    lastHitter: null,
    bouncesSinceHit: 0,
    bouncedOnSide: null,
    firstBounceInAfterHit: false,
    trail: []
  };
}

export function createGameState() {
  return {
    playerScore: 0,
    cpuScore: 0,
    server: "player",
    rallyActive: false,
    awaitingServe: true,
    bouncesNeeded: 2,
    hitsThisRally: 0,
    selectedServeAngle: null,
    serveAngleOptions: null,
    prevServeAngleOptions: null,
    serveAngleLabelCache: null,
    prevServeAngleLabelCache: null,
    serveAngleLabelGen: 0,
    cpuMode: CPU_MODE_NORMAL,
    cpuSkill: 0,
    lastShotType: "",
    message: "Pick serve angle 1\u20134, then time Space on the power bar.",
    gameOver: false,
    contactCooldown: 0,
    cpuPlannedShotType: null,
    /** Q toggles soft shot (kitchen→dink, box→drop at any paddle height) */
    playerSoftShotQueued: false,
    /** Pro ML (per-rally): { x: Float32Array(24), action, expertAction, hardState } */
    mlRallySamples: [],
    /** DAgger aggregated dataset across rallies for the active model */
    mlDataset: [],
    mlTick: 0,
    mlEpisodeId: 0,
    training: {
      active: false,
      targetRallies: 0,
      completedRallies: 0,
      prevCpuMode: CPU_MODE_NORMAL
    },
    /** Session-only counters for ML HUD */
    ml: {
      policyGradSteps: 0,
      lastRallyFrames: 0,
      daggerBeta: 0.85
    }
  };
}
