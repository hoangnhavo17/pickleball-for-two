/**
 * Baseline rule-based CPU movement: step toward a target (tx, ty) on a 3×3 grid.
 * The full game still uses richer heuristics in game.js; this module is the
 * clean “portfolio baseline” you can swap in or compare against ML later.
 *
 * @param {number} cpuX
 * @param {number} cpuY
 * @param {number} tx
 * @param {number} ty
 * @returns {{ dx: number, dy: number }} each in {-1, 0, 1}
 */
export function ruleBasedStepToward(cpuX, cpuY, tx, ty) {
  const dx = tx - cpuX;
  const dy = ty - cpuY;
  const sx = dx === 0 ? 0 : dx > 0 ? 1 : -1;
  const sy = dy === 0 ? 0 : dy > 0 ? 1 : -1;
  return { dx: sx, dy: sy };
}
