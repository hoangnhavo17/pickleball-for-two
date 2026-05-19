/**
 * In-memory dataset for behavior cloning (state/features -> action labels).
 * Export with downloadAsCsv() during dev; Python pipeline reads CSV from ml/data/raw/.
 */

const rows = [];
const FEATURE_DIM = 24;
const CSV_HEADER = [
  ...Array.from({ length: FEATURE_DIM }, (_, i) => `f${i}`),
  "action",
  "expert_action",
  "policy_action",
  "use_expert",
  "episode_id",
  "frame_idx"
].join(",");

/**
 * @param {ArrayLike<number>} features24 normalized feature vector f0..f23
 * @param {number} actionIndex chosen policy action id (0..24)
 * @param {number} expertActionIndex expert/teacher action id (0..24)
 * @param {{policy_action?: number, use_expert?: number, episode_id?: number, frame_idx?: number}} meta
 */
export function recordFeatureRow(features24, actionIndex, expertActionIndex, meta = {}) {
  if (!features24 || typeof features24.length !== "number" || features24.length < FEATURE_DIM) return;
  const row = {};
  for (let i = 0; i < FEATURE_DIM; i += 1) row[`f${i}`] = Number(features24[i]);
  row.action = actionIndex;
  row.expert_action = expertActionIndex;
  row.policy_action = meta.policy_action ?? actionIndex;
  row.use_expert = meta.use_expert ?? (actionIndex === expertActionIndex ? 1 : 0);
  row.episode_id = meta.episode_id ?? -1;
  row.frame_idx = meta.frame_idx ?? -1;
  rows.push(row);
}

export function clearRecording() {
  rows.length = 0;
}

export function getRowCount() {
  return rows.length;
}

function rowToCsvLine(r) {
  const cols = [];
  for (let i = 0; i < FEATURE_DIM; i += 1) cols.push(r[`f${i}`]);
  cols.push(r.action, r.expert_action, r.policy_action, r.use_expert, r.episode_id, r.frame_idx);
  return cols.join(",");
}

/** Download accumulated rows as CSV (browser only). */
export function downloadAsCsv(filename = "gameplay_export.csv") {
  const body = [CSV_HEADER, ...rows.map(rowToCsvLine)].join("\n");
  const blob = new Blob([body], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** JSON export alternative for debugging. */
export function downloadAsJson(filename = "gameplay_export.json") {
  const blob = new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

globalThis.__pickleballRecording = {
  recordFeatureRow,
  clearRecording,
  getRowCount,
  downloadAsCsv,
  downloadAsJson
};
