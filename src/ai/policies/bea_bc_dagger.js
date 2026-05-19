/**
 * Bea Haverkrone — concrete CPU policy.
 *
 * Approach: behavior cloning + DAgger on top of the shared base layer
 * (intent classifier + landing predictor). This module owns ONLY the
 * policy head (a small 2-layer MLP) and its safety gates. Loading the
 * shot oracle / landing physics is the runtime's job.
 *
 * Contract (`src/ai/cpu_model_runtime.js`):
 *   loadPolicy()          -> Promise<boolean>
 *   inferAction(features) -> { action, trusted, confidence?, margin?, reason }
 *   isReady()             -> boolean
 *   meta                  -> { id, displayName, approach }
 *
 * Artifact: `public/models/pro_cpu_v1/bea-haverkrone.json`
 */

const ARTIFACT_URL = new URL(
  "../../../public/models/pro_cpu_v1/bea-haverkrone.json",
  import.meta.url
);

const PRO_CONFIDENCE_MIN = 0.42;
const PRO_MARGIN_MIN = 0.08;
const PRO_OOD_ABS_LIMIT = 1.35;

const EXPECTED_RUNTIME_CONTRACT = {
  feature_dim: 24,
  action_dim: 25,
  schema_name: "pro_cpu_features_v1",
  schema_version: 1,
  physics: {
    gravity: 0.18,
    ball_weight: 1.12,
    air_drag: 0.995,
    restitution: 0.56
  }
};

let model = null;
let ready = false;

function isFiniteNumberArray(arr, expectedLen) {
  return Array.isArray(arr) && arr.length === expectedLen && arr.every((v) => Number.isFinite(v));
}

function approxEqual(a, b, eps = 1e-6) {
  return Math.abs(a - b) <= eps;
}

function contractCompatible(contract, expected) {
  if (!contract || typeof contract !== "object") return true;
  if (Number(contract.feature_dim) !== expected.feature_dim) return false;
  if (Number(contract.action_dim) !== expected.action_dim) return false;
  if (String(contract.schema_name || "") !== expected.schema_name) return false;
  if (Number(contract.schema_version) !== expected.schema_version) return false;
  const p = contract.physics || {};
  const e = expected.physics;
  return (
    approxEqual(Number(p.gravity), e.gravity) &&
    approxEqual(Number(p.ball_weight), e.ball_weight) &&
    approxEqual(Number(p.air_drag), e.air_drag) &&
    approxEqual(Number(p.restitution), e.restitution)
  );
}

function parseModelArtifact(raw) {
  if (!raw || typeof raw !== "object") return null;
  const inputDim = Number(raw.input_dim);
  const hiddenDim = Number(raw.hidden_dim);
  const actionDim = Number(raw.action_dim);
  if (!Number.isInteger(inputDim) || !Number.isInteger(hiddenDim) || !Number.isInteger(actionDim)) return null;
  if (inputDim <= 0 || hiddenDim <= 0 || actionDim <= 0) return null;
  const w1Len = inputDim * hiddenDim;
  const w2Len = hiddenDim * actionDim;
  if (!isFiniteNumberArray(raw.W1, w1Len)) return null;
  if (!isFiniteNumberArray(raw.b1, hiddenDim)) return null;
  if (!isFiniteNumberArray(raw.W2, w2Len)) return null;
  if (!isFiniteNumberArray(raw.b2, actionDim)) return null;
  return {
    inputDim,
    hiddenDim,
    actionDim,
    W1: Float32Array.from(raw.W1),
    b1: Float32Array.from(raw.b1),
    W2: Float32Array.from(raw.W2),
    b2: Float32Array.from(raw.b2),
    h: new Float32Array(hiddenDim),
    logits: new Float32Array(actionDim)
  };
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i += 1) if (arr[i] > arr[best]) best = i;
  return best;
}

function softmax(logits) {
  let max = logits[0];
  for (let i = 1; i < logits.length; i += 1) if (logits[i] > max) max = logits[i];
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  const inv = 1 / (sum + 1e-8);
  for (let i = 0; i < logits.length; i += 1) out[i] *= inv;
  return out;
}

function secondBest(arr, bestIdx) {
  let v = -Infinity;
  for (let i = 0; i < arr.length; i += 1) {
    if (i === bestIdx) continue;
    if (arr[i] > v) v = arr[i];
  }
  return v;
}

export const meta = Object.freeze({
  id: "bea-haverkrone",
  displayName: "Bea Haverkrone",
  approach: "behavior_cloning+dagger"
});

export async function loadPolicy() {
  try {
    const resp = await fetch(ARTIFACT_URL.href, { cache: "no-store" });
    if (!resp.ok) {
      model = null;
      ready = false;
      return false;
    }
    const parsed = await resp.json();
    if (!contractCompatible(parsed.runtime_contract, EXPECTED_RUNTIME_CONTRACT)) {
      model = null;
      ready = false;
      return false;
    }
    const next = parseModelArtifact(parsed);
    if (!next) {
      model = null;
      ready = false;
      return false;
    }
    model = next;
    ready = true;
    return true;
  } catch {
    model = null;
    ready = false;
    return false;
  }
}

export function isReady() {
  return ready;
}

/**
 * Infer a discrete action id (0..action_dim-1) from a normalized feature
 * vector. Returns a structured result rather than just a number so the
 * runtime can implement safety fallback decisions consistently.
 */
export function inferAction(features) {
  if (!ready || !model || !features || typeof features.length !== "number") {
    return { action: null, trusted: false, reason: "model_unavailable" };
  }
  if (features.length < model.inputDim) return { action: null, trusted: false, reason: "shape" };
  for (let i = 0; i < model.inputDim; i += 1) {
    const v = Number(features[i]);
    if (!Number.isFinite(v)) return { action: null, trusted: false, reason: "nan" };
    if (Math.abs(v) > PRO_OOD_ABS_LIMIT) return { action: null, trusted: false, reason: "ood" };
  }
  const { inputDim, hiddenDim, actionDim, W1, b1, W2, b2, h, logits } = model;
  for (let i = 0; i < hiddenDim; i += 1) {
    let s = b1[i];
    const row = i * inputDim;
    for (let d = 0; d < inputDim; d += 1) s += W1[row + d] * features[d];
    h[i] = s > 0 ? s : 0;
  }
  for (let k = 0; k < actionDim; k += 1) {
    let s = b2[k];
    const row = k * hiddenDim;
    for (let i = 0; i < hiddenDim; i += 1) s += W2[row + i] * h[i];
    logits[k] = s;
  }
  const probs = softmax(logits);
  const action = argmax(probs);
  const pBest = probs[action];
  const pSecond = secondBest(probs, action);
  const trusted = pBest >= PRO_CONFIDENCE_MIN && (pBest - pSecond) >= PRO_MARGIN_MIN;
  return {
    action,
    trusted,
    confidence: pBest,
    margin: pBest - pSecond,
    reason: trusted ? "ok" : "low_confidence"
  };
}
