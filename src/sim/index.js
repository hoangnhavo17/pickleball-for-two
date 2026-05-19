/**
 * Public entry point for the shared simulation core.
 *
 * Both the browser game (`src/game/game.js`) and the headless offline runner
 * (`ml/sim/run_offline_sim.js`) import from this module so changes stay in
 * lockstep across environments.
 */

export * from "./constants.js";
export * from "./rng.js";
export * from "./state.js";
export * from "./rules.js";
export {
  servePowerEffective,
  speedFromPowerOnly,
  paddleHitHeight01,
  returnPowerFromPaddleY,
  angleFromPaddleY,
  computeLaunchVelocity,
  trajectoryHitsNetBody,
  netClearAssistProbability,
  adjustAngleToClearNet,
  canClearNetAndLandIn,
  paddleHalfExtents,
  paddleContactsAt,
  paddleContacts,
  paddleApproachOk,
  segmentPaddleHitT,
  paddleContactTAlongStep,
  predictFirstGroundBouncePoint,
  predictBallAtCpu,
  launchBall
} from "./physics.js";
export {
  pickShotIntent,
  applyShotIntent,
  resolvePlayerReturnIntent,
  intentFromPaddleHeight,
  sampleShotParams,
  SOFT_SHOT_DROP,
  SOFT_SHOT_DINK
} from "./shots.js";
export { simulateReturnLanding } from "./physics.js";
export {
  actionToDelta,
  deltaToAction,
  getCpuHeuristicTarget,
  extractCpuMLFeatures,
  chooseExpertCpuAction,
  estimateHardStateScore
} from "./cpu.js";
export { handleContact } from "./contact.js";
export { physicsStep } from "./step.js";
export {
  placeBallOnServer,
  resetServePositions,
  pickCpuServeDisplayPower,
  pickCpuServeAngle,
  pickPlayerTrainingServe,
  serveByPlayer,
  serveByCpu,
  serveByPlayerTraining
} from "./serve.js";
export { endRally } from "./rally.js";
