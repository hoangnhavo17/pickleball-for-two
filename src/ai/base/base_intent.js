/**
 * Base shot intent classifier (shared by every CPU model).
 *
 * Today this is a lookup over the offline-generated shot oracle artifact
 * (`public/models/pro_cpu_v1/shot_oracle_v1.json`). The interface is
 * model-agnostic on purpose so a future learned classifier can drop in
 * without touching the model runtime or any concrete policy.
 *
 * Public API:
 *   loadShotIntentClassifier(url?)   -> Promise<boolean>
 *   predictShotIntent(cpuX01, cpuY01) -> string | null   (back-compat signature)
 *   predictShotIntentDetailed(input) -> { best, scores } | null
 *   isShotIntentReady()              -> boolean
 */

const DEFAULT_ARTIFACT_URL = new URL(
  "../../../public/models/pro_cpu_v1/shot_oracle_v1.json",
  import.meta.url
);

let oracle = null;
let ready = false;

function parseOracle(raw) {
  if (!raw || typeof raw !== "object") return null;
  const grid = raw.grid || {};
  const xBins = Number(grid.x_bins);
  const yBins = Number(grid.y_bins);
  if (!Number.isInteger(xBins) || !Number.isInteger(yBins) || xBins <= 0 || yBins <= 0) return null;
  if (!Array.isArray(raw.cells) || raw.cells.length !== xBins * yBins) return null;
  const cellMap = new Map();
  for (let i = 0; i < raw.cells.length; i += 1) {
    const c = raw.cells[i];
    if (!c || typeof c !== "object") continue;
    const xi = Number(c.x_bin);
    const yi = Number(c.y_bin);
    if (!Number.isInteger(xi) || !Number.isInteger(yi)) continue;
    if (typeof c.best_shot !== "string") continue;
    cellMap.set(`${xi}:${yi}`, { best: c.best_shot, scores: c.scores || null });
  }
  return { xBins, yBins, cellMap };
}

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

export async function loadShotIntentClassifier(url) {
  try {
    const target = url ? new URL(url, import.meta.url) : DEFAULT_ARTIFACT_URL;
    const resp = await fetch(target.href, { cache: "no-store" });
    if (!resp.ok) {
      oracle = null;
      ready = false;
      return false;
    }
    const parsed = parseOracle(await resp.json());
    if (!parsed) {
      oracle = null;
      ready = false;
      return false;
    }
    oracle = parsed;
    ready = true;
    return true;
  } catch {
    oracle = null;
    ready = false;
    return false;
  }
}

export function isShotIntentReady() {
  return ready;
}

/**
 * Back-compat signature used by `pickShotIntent` and `physicsStep` in the
 * shared sim core. Returns the recommended best shot type as a string.
 */
export function predictShotIntent(cpuX01, cpuY01) {
  if (!ready || !oracle) return null;
  const x = Math.min(oracle.xBins - 1, Math.floor(clamp01(cpuX01) * oracle.xBins));
  const y = Math.min(oracle.yBins - 1, Math.floor(clamp01(cpuY01) * oracle.yBins));
  const cell = oracle.cellMap.get(`${x}:${y}`);
  return cell ? cell.best : null;
}

/**
 * Richer signature for callers that want full per-shot scores.
 * Future learned classifiers can also implement this contract.
 */
export function predictShotIntentDetailed({ courtX01, courtY01 }) {
  if (!ready || !oracle) return null;
  const x = Math.min(oracle.xBins - 1, Math.floor(clamp01(courtX01) * oracle.xBins));
  const y = Math.min(oracle.yBins - 1, Math.floor(clamp01(courtY01) * oracle.yBins));
  const cell = oracle.cellMap.get(`${x}:${y}`);
  if (!cell) return null;
  return { best: cell.best, scores: cell.scores };
}
