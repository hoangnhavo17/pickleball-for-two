/**
 * End-of-rally bookkeeping: scoring, side-out, game-over detection.
 *
 * Pure: takes (state, ball, player, cpu, winner). Mutates state in place and
 * returns a summary object the caller can use for UI/dataset bookkeeping.
 *
 * Note: ML weight updates and dataset aggregation live in src/game/game.js
 * (browser path) and ml/sim/run_offline_sim.js (offline path) so this module
 * stays free of policy/runtime dependencies.
 */

import { WIN_SCORE } from "./constants.js";
import { placeBallOnServer, resetServePositions } from "./serve.js";

export function endRally(state, ball, player, cpu, winner) {
  if (state.training.active) {
    state.training.completedRallies += 1;
  }

  let pointAwardedTo = null;
  let sideOutTo = null;
  if (winner === state.server) {
    if (winner === "player") state.playerScore += 1;
    else state.cpuScore += 1;
    pointAwardedTo = winner;
  } else {
    state.server = winner;
    sideOutTo = winner;
  }

  state.rallyActive = false;
  state.awaitingServe = true;
  state.cpuPlannedShotType = null;
  state.bouncesNeeded = 2;
  state.hitsThisRally = 0;
  state.contactCooldown = 0;
  state.selectedServeAngle = null;

  let gameOver = false;
  if (state.playerScore >= WIN_SCORE && state.playerScore - state.cpuScore >= 2) {
    gameOver = true;
  } else if (state.cpuScore >= WIN_SCORE && state.cpuScore - state.playerScore >= 2) {
    gameOver = true;
  }
  state.gameOver = gameOver;

  resetServePositions(state, player, cpu);
  placeBallOnServer(state, ball, player, cpu);

  return {
    winner,
    pointAwardedTo,
    sideOutTo,
    gameOver,
    playerScore: state.playerScore,
    cpuScore: state.cpuScore,
    server: state.server
  };
}
