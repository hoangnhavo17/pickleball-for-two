/**
 * Browser CPU policy: 2-layer MLP + softmax over 25 move actions (5x5 paddle deltas).
 * Training: basic behavior cloning (supervised imitation of geometry expert actions).
 * Weights persist in localStorage (per model id); export/import as JSON for backups or Python tooling.
 */
const CpuPolicy = (function globalCpuPolicy() {
  const INPUT_DIM = 24;
  const HIDDEN_DIM = 32;
  const ACTION_GRID = 5;
  const ACTION_CENTER = (ACTION_GRID - 1) / 2;
  const ACTION_DIM = ACTION_GRID * ACTION_GRID;
  const STORAGE_KEY_PREFIX = "pickleball_cpu_policy_v1";
  const DEFAULT_MODEL_ID = "Bea Haverkrone";
  const LEGACY_DEFAULT_MODEL_IDS = ["b-havior-klone", "bc-baseline", "default"];
  let activeModelId = DEFAULT_MODEL_ID;
  /** Tracks whether the currently selected model id has been initialized/loaded this session. */
  let initializedModelId = null;

  function buildStorageKey(modelId) {
    return `${STORAGE_KEY_PREFIX}:${modelId || DEFAULT_MODEL_ID}`;
  }

  /** action index -> (dx, dy) each in {-1,-0.5,0,0.5,1} for paddle step direction */
  function actionToDelta(a) {
    const ix = a % ACTION_GRID;
    const iy = Math.floor(a / ACTION_GRID);
    const dx = (ix - ACTION_CENTER) / ACTION_CENTER;
    const dy = (iy - ACTION_CENTER) / ACTION_CENTER;
    return [dx, dy];
  }

  function deltaToAction(dx, dy) {
    const sx = Math.round(Math.max(-1, Math.min(1, dx)) * ACTION_CENTER) + ACTION_CENTER;
    const sy = Math.round(Math.max(-1, Math.min(1, dy)) * ACTION_CENTER) + ACTION_CENTER;
    return sx + sy * ACTION_GRID;
  }

  let W1 = new Float32Array(HIDDEN_DIM * INPUT_DIM);
  let b1 = new Float32Array(HIDDEN_DIM);
  let W2 = new Float32Array(ACTION_DIM * HIDDEN_DIM);
  let b2 = new Float32Array(ACTION_DIM);

  function randn() {
    return (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 0.5;
  }

  function initRandom() {
    const scale1 = Math.sqrt(2 / (INPUT_DIM + HIDDEN_DIM));
    const scale2 = Math.sqrt(2 / (HIDDEN_DIM + ACTION_DIM));
    for (let i = 0; i < W1.length; i++) W1[i] = randn() * scale1;
    for (let i = 0; i < b1.length; i++) b1[i] = randn() * scale1 * 0.1;
    for (let i = 0; i < W2.length; i++) W2[i] = randn() * scale2;
    for (let i = 0; i < b2.length; i++) b2[i] = randn() * scale2 * 0.1;
  }

  function softmax(logits, outProbs) {
    let max = logits[0];
    for (let k = 1; k < ACTION_DIM; k++) if (logits[k] > max) max = logits[k];
    let sum = 0;
    for (let k = 0; k < ACTION_DIM; k++) {
      outProbs[k] = Math.exp(logits[k] - max);
      sum += outProbs[k];
    }
    const inv = 1 / (sum + 1e-8);
    for (let k = 0; k < ACTION_DIM; k++) outProbs[k] *= inv;
  }

  function forward(x, out) {
    const { z1, h, logits, probs } = out;
    for (let i = 0; i < HIDDEN_DIM; i++) {
      let s = b1[i];
      const row = i * INPUT_DIM;
      for (let d = 0; d < INPUT_DIM; d++) s += W1[row + d] * x[d];
      z1[i] = s;
      h[i] = s > 0 ? s : 0;
    }
    for (let k = 0; k < ACTION_DIM; k++) {
      let s = b2[k];
      const row = k * HIDDEN_DIM;
      for (let i = 0; i < HIDDEN_DIM; i++) s += W2[row + i] * h[i];
      logits[k] = s;
    }
    softmax(logits, probs);
  }

  function sampleAction(probs, temperature) {
    if (temperature !== 1) {
      const scaled = new Float32Array(ACTION_DIM);
      let max = -Infinity;
      for (let k = 0; k < ACTION_DIM; k++) {
        scaled[k] = probs[k] > 0 ? Math.log(probs[k] + 1e-8) / temperature : -100;
        if (scaled[k] > max) max = scaled[k];
      }
      let sum = 0;
      for (let k = 0; k < ACTION_DIM; k++) {
        scaled[k] = Math.exp(scaled[k] - max);
        sum += scaled[k];
      }
      let r = Math.random() * sum;
      for (let k = 0; k < ACTION_DIM; k++) {
        r -= scaled[k];
        if (r <= 0) return k;
      }
      return ACTION_DIM - 1;
    }
    let r = Math.random();
    for (let k = 0; k < ACTION_DIM; k++) {
      r -= probs[k];
      if (r <= 0) return k;
    }
    return ACTION_DIM - 1;
  }

  function argmax(probs) {
    let best = 0;
    for (let k = 1; k < ACTION_DIM; k++) if (probs[k] > probs[best]) best = k;
    return best;
  }

  function backward(x, z1, h, logits, probs, dLogits, dW1, db1, dW2, db2) {
    for (let k = 0; k < ACTION_DIM; k++) db2[k] = dLogits[k];

    for (let k = 0; k < ACTION_DIM; k++) {
      const row = k * HIDDEN_DIM;
      for (let i = 0; i < HIDDEN_DIM; i++) dW2[row + i] = dLogits[k] * h[i];
    }

    const dh = new Float32Array(HIDDEN_DIM);
    for (let i = 0; i < HIDDEN_DIM; i++) {
      let s = 0;
      for (let k = 0; k < ACTION_DIM; k++) s += W2[k * HIDDEN_DIM + i] * dLogits[k];
      dh[i] = s;
    }

    const dz1 = new Float32Array(HIDDEN_DIM);
    for (let i = 0; i < HIDDEN_DIM; i++) dz1[i] = z1[i] > 0 ? dh[i] : 0;

    for (let i = 0; i < HIDDEN_DIM; i++) db1[i] = dz1[i];

    for (let i = 0; i < HIDDEN_DIM; i++) {
      const row = i * INPUT_DIM;
      for (let d = 0; d < INPUT_DIM; d++) dW1[row + d] = dz1[i] * x[d];
    }
  }

  function clipGrad(arr, maxNorm) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
    const n = Math.sqrt(sum);
    if (n <= maxNorm || n < 1e-8) return;
    const s = maxNorm / n;
    for (let i = 0; i < arr.length; i++) arr[i] *= s;
  }

  const scratch = {
    z1: new Float32Array(HIDDEN_DIM),
    h: new Float32Array(HIDDEN_DIM),
    logits: new Float32Array(ACTION_DIM),
    probs: new Float32Array(ACTION_DIM),
    dLogits: new Float32Array(ACTION_DIM),
    dW1: new Float32Array(HIDDEN_DIM * INPUT_DIM),
    db1: new Float32Array(HIDDEN_DIM),
    dW2: new Float32Array(ACTION_DIM * HIDDEN_DIM),
    db2: new Float32Array(ACTION_DIM),
  };

  const LR_BC = 0.03;
  const GRAD_CLIP = 1.2;

  function trainRally(samples) {
    if (!samples || samples.length === 0) return;
    const accW1 = new Float32Array(W1.length);
    const accB1 = new Float32Array(b1.length);
    const accW2 = new Float32Array(W2.length);
    const accB2 = new Float32Array(b2.length);
    let used = 0;

    for (let s = 0; s < samples.length; s++) {
      const { x, expertAction } = samples[s];
      if (!(expertAction >= 0 && expertAction < ACTION_DIM)) continue;
      forward(x, scratch);
      const { z1, h, logits, probs } = scratch;

      for (let k = 0; k < ACTION_DIM; k++) scratch.dLogits[k] = 0;

      for (let k = 0; k < ACTION_DIM; k++) {
        const eE = k === expertAction ? 1 : 0;
        scratch.dLogits[k] += (eE - probs[k]);
      }
      used += 1;

      const dW1 = new Float32Array(W1.length);
      const db1 = new Float32Array(b1.length);
      const dW2 = new Float32Array(W2.length);
      const db2 = new Float32Array(b2.length);
      backward(x, z1, h, logits, probs, scratch.dLogits, dW1, db1, dW2, db2);

      for (let i = 0; i < accW1.length; i++) accW1[i] += dW1[i];
      for (let i = 0; i < accB1.length; i++) accB1[i] += db1[i];
      for (let i = 0; i < accW2.length; i++) accW2[i] += dW2[i];
      for (let i = 0; i < accB2.length; i++) accB2[i] += db2[i];
    }
    if (used === 0) return;

    const inv = 1 / used;
    for (let i = 0; i < accW1.length; i++) accW1[i] *= inv;
    for (let i = 0; i < accB1.length; i++) accB1[i] *= inv;
    for (let i = 0; i < accW2.length; i++) accW2[i] *= inv;
    for (let i = 0; i < accB2.length; i++) accB2[i] *= inv;

    clipGrad(accW1, GRAD_CLIP);
    clipGrad(accB1, GRAD_CLIP);
    clipGrad(accW2, GRAD_CLIP);
    clipGrad(accB2, GRAD_CLIP);

    for (let i = 0; i < W1.length; i++) W1[i] += LR_BC * accW1[i];
    for (let i = 0; i < b1.length; i++) b1[i] += LR_BC * accB1[i];
    for (let i = 0; i < W2.length; i++) W2[i] += LR_BC * accW2[i];
    for (let i = 0; i < b2.length; i++) b2[i] += LR_BC * accB2[i];
  }

  function serialize() {
    return {
      W1: Array.from(W1),
      b1: Array.from(b1),
      W2: Array.from(W2),
      b2: Array.from(b2),
    };
  }

  function deserialize(data) {
    if (!data || !data.W1 || data.W1.length !== W1.length) return false;
    W1.set(data.W1);
    b1.set(data.b1);
    W2.set(data.W2);
    b2.set(data.b2);
    return true;
  }

  function save() {
    try {
      localStorage.setItem(buildStorageKey(activeModelId), JSON.stringify(serialize()));
    } catch (e) {
      /* ignore quota */
    }
  }

  function load() {
    try {
      const raw = localStorage.getItem(buildStorageKey(activeModelId));
      if (!raw) return false;
      const data = JSON.parse(raw);
      return deserialize(data);
    } catch (e) {
      return false;
    }
  }

  function resetBrain() {
    initRandom();
    try {
      localStorage.removeItem(buildStorageKey(activeModelId));
    } catch (e) { /* ignore */ }
  }

  function exportModelObject() {
    return serialize();
  }

  function importModelObject(data) {
    return deserialize(data);
  }

  function ensureLoaded() {
    // One-time semantics per model id for this tab session.
    // Repeated calls should be no-ops, otherwise we can re-randomize before first save.
    if (initializedModelId === activeModelId) return;
    if (load()) {
      initializedModelId = activeModelId;
      return;
    }
    // One-time migration: carry old baseline ids into the new named baseline model.
    if (activeModelId === DEFAULT_MODEL_ID) {
      try {
        for (let i = 0; i < LEGACY_DEFAULT_MODEL_IDS.length; i++) {
          const legacyRaw = localStorage.getItem(buildStorageKey(LEGACY_DEFAULT_MODEL_IDS[i]));
          if (legacyRaw) {
            const data = JSON.parse(legacyRaw);
            if (deserialize(data)) {
              save();
              initializedModelId = activeModelId;
              return;
            }
          }
        }
      } catch (e) {
        /* ignore */
      }
    }
    initRandom();
    initializedModelId = activeModelId;
  }

  function setModelId(modelId) {
    activeModelId = modelId || DEFAULT_MODEL_ID;
    // Force one ensureLoaded() pass when switching models.
    if (initializedModelId !== activeModelId) initializedModelId = null;
  }

  function getModelId() {
    return activeModelId;
  }

  function deleteModelById(modelId) {
    try {
      localStorage.removeItem(buildStorageKey(modelId || DEFAULT_MODEL_ID));
      return true;
    } catch (e) {
      return false;
    }
  }

  function duplicateModelById(fromId, toId) {
    try {
      const raw = localStorage.getItem(buildStorageKey(fromId || DEFAULT_MODEL_ID));
      if (!raw) return false;
      localStorage.setItem(buildStorageKey(toId || DEFAULT_MODEL_ID), raw);
      return true;
    } catch (e) {
      return false;
    }
  }

  return {
    INPUT_DIM,
    HIDDEN_DIM,
    ACTION_DIM,
    DEFAULT_MODEL_ID,
    actionToDelta,
    deltaToAction,
    forward(x, out) {
      forward(x, out || scratch);
    },
    sampleAction,
    argmax,
    trainRally,
    save,
    load,
    exportModelObject,
    importModelObject,
    resetBrain,
    ensureLoaded,
    setModelId,
    getModelId,
    deleteModelById,
    duplicateModelById,
  };
})();

export { CpuPolicy };
if (typeof globalThis !== "undefined") globalThis.CpuPolicy = CpuPolicy;
