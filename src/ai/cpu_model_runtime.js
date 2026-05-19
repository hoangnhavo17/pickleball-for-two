/**
 * CPU model runtime — composes the shared base layer (shot intent
 * classifier + ball landing predictor) with a model-specific policy
 * head. New models register a policy module that implements the
 * { meta, loadPolicy, isReady, inferAction } contract.
 *
 * Public API:
 *   initCpuModelRuntime()              -> Promise<void>
 *   listAvailableModels()              -> Array<{ id, displayName, approach }>
 *   getActiveModelId()                 -> string
 *   setActiveModelId(id)               -> Promise<boolean>
 *
 *   // Base layer (model-agnostic):
 *   predictShotIntent(cpuX01, cpuY01)  -> string | null
 *   predictShotIntentDetailed(input)   -> { best, scores } | null
 *   predictLanding({ ball, state })    -> { kind, x, y, x01, y01, confidence } | null
 *
 *   // Policy layer (delegates to active model):
 *   inferAction(features)              -> { action, trusted, ... }
 *   isPolicyReady()                    -> boolean
 */

import {
  loadShotIntentClassifier,
  predictShotIntent as baseIntentPredict,
  predictShotIntentDetailed as baseIntentPredictDetailed,
  isShotIntentReady
} from "./base/base_intent.js";

import {
  predictLanding as basePredictLanding,
  predictFirstBounce as basePredictFirstBounce,
  predictReachableAtCpu as basePredictReachableAtCpu
} from "./base/base_landing.js";

import * as BeaPolicy from "./policies/bea_bc_dagger.js";

const POLICIES = new Map();

function registerPolicy(mod) {
  if (!mod || !mod.meta || !mod.meta.id) return;
  POLICIES.set(mod.meta.id, mod);
}

registerPolicy(BeaPolicy);

const DEFAULT_MODEL_ID = BeaPolicy.meta.id;
let activeModelId = DEFAULT_MODEL_ID;

function getActivePolicy() {
  return POLICIES.get(activeModelId) || null;
}

let runtimeReadyPromise = null;

async function loadActivePolicyOnce() {
  const policy = getActivePolicy();
  if (!policy) return false;
  try {
    return await policy.loadPolicy();
  } catch {
    return false;
  }
}

export function initCpuModelRuntime() {
  if (!runtimeReadyPromise) {
    runtimeReadyPromise = Promise.all([loadShotIntentClassifier(), loadActivePolicyOnce()]).then(
      () => undefined
    );
  }
  return runtimeReadyPromise;
}

export function listAvailableModels() {
  return Array.from(POLICIES.values()).map((p) => ({ ...p.meta }));
}

export function getActiveModelId() {
  return activeModelId;
}

export async function setActiveModelId(id) {
  if (!POLICIES.has(id)) return false;
  activeModelId = id;
  runtimeReadyPromise = null;
  await initCpuModelRuntime();
  return true;
}

/* ---------- Base layer pass-throughs ------------------------------------ */

export function predictShotIntent(cpuX01, cpuY01) {
  return baseIntentPredict(cpuX01, cpuY01);
}

export function predictShotIntentDetailed(input) {
  return baseIntentPredictDetailed(input);
}

export function predictLanding(input) {
  return basePredictLanding(input);
}

export function predictFirstBounce(ball) {
  return basePredictFirstBounce(ball);
}

export function predictReachableAtCpu(state, ball) {
  return basePredictReachableAtCpu(state, ball);
}

export function isShotIntentClassifierReady() {
  return isShotIntentReady();
}

/* ---------- Policy layer (delegates to active model) -------------------- */

export function inferAction(features) {
  const policy = getActivePolicy();
  if (!policy) return { action: null, trusted: false, reason: "no_active_model" };
  return policy.inferAction(features);
}

export function isPolicyReady() {
  const policy = getActivePolicy();
  return !!policy && policy.isReady();
}

void initCpuModelRuntime();
