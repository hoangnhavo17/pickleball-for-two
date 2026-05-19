/**
 * Backward-compatibility shim for the legacy `mlAI` API.
 *
 * The real wiring now lives in:
 *   - `src/ai/base/base_intent.js`        (shared shot intent classifier)
 *   - `src/ai/base/base_landing.js`       (shared ball landing predictor)
 *   - `src/ai/policies/bea_bc_dagger.js`  (Bea — BC + DAgger policy head)
 *   - `src/ai/cpu_model_runtime.js`       (orchestrator + model registry)
 *
 * Existing callers keep working through this shim. New code should import
 * from `cpu_model_runtime.js` instead.
 */

import {
  initCpuModelRuntime,
  inferAction,
  isPolicyReady,
  predictShotIntent
} from "./cpu_model_runtime.js";

import { ruleBasedStepToward } from "./ruleBasedAI.js";

export async function tryLoadProCpuModel() {
  await initCpuModelRuntime();
  return isPolicyReady();
}

export function inferProCpuAction(features) {
  return inferAction(features);
}

export function inferProCpuMove(cpuX, cpuY, tx, ty) {
  return ruleBasedStepToward(cpuX, cpuY, tx, ty);
}

export function isProCpuModelLoaded() {
  return isPolicyReady();
}

export function inferProShotIntent(cpuX01, cpuY01) {
  return predictShotIntent(cpuX01, cpuY01);
}

void tryLoadProCpuModel();
