import { inferProCpuAction, inferProShotIntent } from "../ai/mlAI.js";
import {
  ACTION_DIM,
  BALL_R,
  CPU_BASE_X,
  CPU_HEURISTIC_LAND_BIAS_X,
  CPU_HEURISTIC_NET_GUARD_X,
  CPU_MODE_NORMAL,
  CPU_MODE_PRO,
  CPU_RALLY_MAX_X,
  CPU_RALLY_MIN_X,
  CPU_SERVE_DEFAULT_X,
  CPU_SERVE_MAX_X,
  CPU_SERVE_MIN_X,
  CPU_SPEED,
  CPU_VERTICAL_SPEED,
  COURT_LEFT,
  COURT_RIGHT,
  FEATURE_DIM,
  GROUND_Y,
  HEIGHT,
  KITCHEN_HALF,
  NET_TOP,
  NET_X,
  PADDLE_PLAY_H,
  PADDLE_REACH_BOTTOM,
  PADDLE_REACH_TOP,
  PADDLE_PLAY_W,
  PLAYER_BASE_X,
  PLAYER_RALLY_MAX_X,
  PLAYER_RALLY_MIN_X,
  PLAYER_SERVE_MAX_X,
  PLAYER_SERVE_MIN_X,
  PLAYER_SPEED,
  PLAYER_VERTICAL_SPEED,
  POWER_MAX,
  POWER_MIN,
  SERVE_ANGLE_PRESETS,
  SERVE_PADDLE_REACH_TOP,
  SERVE_POWER_OSCILLATION_MS,
  WIDTH,
  BROWSER_ANIMATION_TIME_SCALE,
  defaultRng,
  clamp,
  createPlayer,
  createCpu,
  createBall,
  createGameState,
  mustLetBallBounceBeforeReturn,
  servePowerEffective,
  returnPowerFromPaddleY,
  predictFirstGroundBouncePoint,
  predictBallAtCpu,
  pickShotIntent,
  getCpuHeuristicTarget,
  extractCpuMLFeatures,
  chooseExpertCpuAction,
  estimateHardStateScore,
  physicsStep,
  placeBallOnServer,
  resetServePositions,
  serveByPlayer as simServeByPlayer,
  serveByCpu as simServeByCpu,
  serveByPlayerTraining as simServeByPlayerTraining,
  endRally as simEndRally
} from "../sim/index.js";

const canvas = document.getElementById("court");
const ctx = canvas.getContext("2d");
const scoreEl = document.getElementById("score");
const statusEl = document.getElementById("status");
const powerOptionsEl = document.getElementById("powerOptions");
const serveControlsLabel = document.getElementById("serveControlsLabel");
const servePowerBarBlock = document.getElementById("servePowerBarBlock");
const servePowerFill = document.getElementById("servePowerFill");
const servePowerReadout = document.getElementById("servePowerReadout");
const newGameBtn = document.getElementById("newGameBtn");
const cpuModelSelect = document.getElementById("cpuModelSelect");
const mainGameEl = document.querySelector("main.game");

const CPU_MODEL_LIST_KEY = "pickleball_cpu_model_profiles_v1";
const CPU_MODEL_ACTIVE_KEY = "pickleball_cpu_model_active_v1";
const DEFAULT_CPU_MODEL_ID = "Bea Haverkrone";
const LEGACY_DEFAULT_CPU_MODEL_IDS = ["b-havior-klone", "bc-baseline", "default"];

const MAX_DAGGER_DATASET = 4000;
const DAGGER_BATCH_SIZE = 320;
const DAGGER_BETA_START = 0.85;
const DAGGER_BETA_MIN = 0.05;
const DAGGER_BETA_DECAY = 0.985;
const DAGGER_RECENCY_EXP = 2.6;
const DAGGER_UNIFORM_MIX = 0.22;
const DAGGER_EVAL_BATCH_SIZE = 128;
const DAGGER_KEEP_DELTA = 0.002;

const heldKeys = { left: false, right: false, up: false, down: false };

/** Wall-clock stretch for ball sim only (physics constants unchanged). */
let physicsStepAccum = 0;
const MAX_PHYSICS_STEPS_PER_FRAME = 4;

const player = createPlayer();
const cpu = createCpu();
const ball = createBall();
const state = createGameState();

const rng = defaultRng;

const _cpuFeat = new Float32Array(FEATURE_DIM);
const _cpuPolOut = {
  z1: new Float32Array(32),
  h: new Float32Array(32),
  logits: new Float32Array(typeof CpuPolicy !== "undefined" ? CpuPolicy.ACTION_DIM : ACTION_DIM),
  probs: new Float32Array(typeof CpuPolicy !== "undefined" ? CpuPolicy.ACTION_DIM : ACTION_DIM)
};

function loadCpuModelList() {
  try {
    const raw = localStorage.getItem(CPU_MODEL_LIST_KEY);
    if (!raw) return [DEFAULT_CPU_MODEL_ID];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return [DEFAULT_CPU_MODEL_ID];
    const clean = parsed
      .map((v) => String(v).trim())
      .map((v) => (LEGACY_DEFAULT_CPU_MODEL_IDS.includes(v) ? DEFAULT_CPU_MODEL_ID : v))
      .filter((v, i, arr) => v.length > 0 && arr.indexOf(v) === i);
    if (!clean.includes(DEFAULT_CPU_MODEL_ID)) clean.unshift(DEFAULT_CPU_MODEL_ID);
    return clean;
  } catch (e) {
    return [DEFAULT_CPU_MODEL_ID];
  }
}

function saveCpuModelList(list) {
  try {
    localStorage.setItem(CPU_MODEL_LIST_KEY, JSON.stringify(list));
  } catch (e) {
    /* ignore quota issues */
  }
}

function getActiveCpuModelId() {
  try {
    const id = localStorage.getItem(CPU_MODEL_ACTIVE_KEY);
    if (!id || !id.trim()) return DEFAULT_CPU_MODEL_ID;
    const clean = id.trim();
    return LEGACY_DEFAULT_CPU_MODEL_IDS.includes(clean) ? DEFAULT_CPU_MODEL_ID : clean;
  } catch (e) {
    return DEFAULT_CPU_MODEL_ID;
  }
}

function setActiveCpuModelId(id) {
  try {
    localStorage.setItem(CPU_MODEL_ACTIVE_KEY, id);
  } catch (e) {
    /* ignore quota issues */
  }
}

function refreshCpuModelSelectUI() {
  if (!cpuModelSelect) return;
  const models = loadCpuModelList();
  const active = getActiveCpuModelId();
  cpuModelSelect.innerHTML = "";
  models.forEach((id) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    cpuModelSelect.appendChild(opt);
  });
  cpuModelSelect.value = models.includes(active) ? active : DEFAULT_CPU_MODEL_ID;
}

function switchCpuModel(id) {
  if (typeof CpuPolicy === "undefined") return;
  const models = loadCpuModelList();
  const next = models.includes(id) ? id : DEFAULT_CPU_MODEL_ID;
  CpuPolicy.setModelId(next);
  CpuPolicy.ensureLoaded();
  setActiveCpuModelId(next);
  refreshCpuModelSelectUI();
  state.mlRallySamples = [];
  state.mlDataset = [];
  state.mlTick = 0;
  state.ml.daggerBeta = DAGGER_BETA_START;
  state.message = `CPU model switched to "${next}".`;
  updateHud();
}

function randomChoice(arr) { return arr[rng.randInt(0, arr.length - 1)]; }

function sampleDaggerBatch() {
  if (!state.mlDataset.length) return [];
  const n = Math.min(DAGGER_BATCH_SIZE, state.mlDataset.length);
  const out = [];
  const len = state.mlDataset.length;
  for (let i = 0; i < n; i += 1) {
    if (rng.random() < DAGGER_UNIFORM_MIX) {
      out.push(randomChoice(state.mlDataset));
      continue;
    }
    const u = rng.random();
    const recency01 = 1 - Math.pow(1 - u, DAGGER_RECENCY_EXP);
    const idx = clamp(Math.floor(recency01 * len), 0, len - 1);
    out.push(state.mlDataset[idx]);
  }
  return out;
}

function sampleUniformBatch(arr, n) {
  if (!arr || arr.length === 0) return [];
  const take = Math.min(n, arr.length);
  const out = [];
  for (let i = 0; i < take; i += 1) out.push(randomChoice(arr));
  return out;
}

function policyExpertMatchOn(samples) {
  if (!samples || samples.length === 0 || typeof CpuPolicy === "undefined") return 0;
  let ok = 0;
  let used = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const s = samples[i];
    if (!s || !s.x || !(s.expertAction >= 0 && s.expertAction < CpuPolicy.ACTION_DIM)) continue;
    CpuPolicy.forward(s.x, _cpuPolOut);
    const a = CpuPolicy.argmax(_cpuPolOut.probs);
    if (a === s.expertAction) ok += 1;
    used += 1;
  }
  return used > 0 ? ok / used : 0;
}

function serveOscillator01() {
  const t = performance.now() / SERVE_POWER_OSCILLATION_MS;
  return Math.sin(t * Math.PI * 2) * 0.5 + 0.5;
}

function serveOscillatorPowerInt() {
  return clamp(Math.round(serveOscillator01() * POWER_MAX), POWER_MIN, POWER_MAX);
}

function cpuSkill01() {
  if (state.cpuMode !== CPU_MODE_PRO) return 0;
  return clamp(state.cpuSkill, 0, 1);
}

function updateServePowerBarDom() {
  if (!servePowerBarBlock || servePowerBarBlock.hidden || !servePowerFill || !servePowerReadout) return;
  const v = serveOscillator01();
  servePowerFill.style.width = `${v * 100}%`;
  servePowerReadout.textContent = String(Math.round(v * POWER_MAX));
}

function invalidateServeAngleLabelCache() {
  if (state.serveAngleOptions && state.serveAngleOptions.length === 4) {
    state.prevServeAngleOptions = [...state.serveAngleOptions];
  }
  state.serveAngleOptions = null;
  if (state.serveAngleLabelCache && state.serveAngleLabelCache.length === SERVE_ANGLE_PRESETS.length) {
    state.prevServeAngleLabelCache = [...state.serveAngleLabelCache];
  }
  state.serveAngleLabelCache = null;
  state.serveAngleLabelGen += 1;
}

function generateServeAngleOptions() {
  const minGap = 4;
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const picked = [];
    let tries = 0;
    while (picked.length < 4 && tries < 220) {
      tries += 1;
      const candidate = rng.randInt(1, 89);
      if (picked.some((v) => Math.abs(v - candidate) < minGap)) continue;
      picked.push(candidate);
    }
    if (picked.length === 4) {
      picked.sort((a, b) => a - b);
      return picked;
    }
  }
  return [12, 28, 46, 62];
}

function ensureServeAngleOptions() {
  if (state.serveAngleOptions && state.serveAngleOptions.length === 4) return;
  let next = null;
  for (let tries = 0; tries < 10; tries += 1) {
    next = generateServeAngleOptions();
    const prev = state.prevServeAngleOptions;
    if (!prev || prev.length !== next.length) break;
    let same = true;
    for (let i = 0; i < next.length; i += 1) {
      if (next[i] !== prev[i]) {
        same = false;
        break;
      }
    }
    if (!same) break;
  }
  state.serveAngleOptions = next;
}

function ensureServeAngleLabelCache() {
  ensureServeAngleOptions();
  if (!state.serveAngleLabelCache || state.serveAngleLabelCache.length !== SERVE_ANGLE_PRESETS.length) {
    const prev = state.prevServeAngleLabelCache;
    let next = null;
    for (let tries = 0; tries < 8; tries += 1) {
      next = state.serveAngleOptions.map((a) => formatServeAngleChallenge(a));
      if (!prev || prev.length !== next.length) break;
      let same = true;
      for (let i = 0; i < next.length; i += 1) {
        if (next[i] !== prev[i]) {
          same = false;
          break;
        }
      }
      if (!same) break;
    }
    state.serveAngleLabelCache = next;
  }
}

function clearServePanel() {
  state.selectedServeAngle = null;
  invalidateServeAngleLabelCache();
  powerOptionsEl.innerHTML = "";
  if (servePowerBarBlock) servePowerBarBlock.hidden = true;
}

function formatServeAngleChallenge(angleDeg) {
  return `${angleDeg}\u00B0`;
}

function syncCpuModeBtnStyles() {
  document.querySelectorAll(".cpu-mode-btn").forEach((btn) => {
    btn.classList.toggle("is-selected", btn.dataset.cpuMode === state.cpuMode);
  });
}

function syncCpuProUiVisibility() {
  const pro = state.cpuMode === CPU_MODE_PRO;
  if (mainGameEl) mainGameEl.classList.toggle("cpu-mode-pro", pro);
}

function setCpuMode(mode) {
  if (mode !== CPU_MODE_NORMAL && mode !== CPU_MODE_PRO) return;
  state.cpuMode = mode;
  if (mode === CPU_MODE_PRO) {
    state.cpuSkill = Math.max(state.cpuSkill, 0.12);
    if (typeof CpuPolicy !== "undefined") CpuPolicy.ensureLoaded();
  } else {
    state.mlRallySamples = [];
    state.mlTick = 0;
  }
  syncCpuModeBtnStyles();
  syncCpuProUiVisibility();
  updateHud();
}

function renderServeAngleButtons() {
  ensureServeAngleLabelCache();
  const useExpr = false;
  const gen = state.serveAngleLabelGen;
  const firstBtn = powerOptionsEl.querySelector("button");
  const canPatchOnly =
    powerOptionsEl.children.length === SERVE_ANGLE_PRESETS.length &&
    firstBtn &&
    firstBtn.dataset.labelGen === String(gen);

  if (canPatchOnly) {
    state.serveAngleOptions.forEach((angleDeg, i) => {
      const btn = powerOptionsEl.children[i];
      const selected = state.selectedServeAngle === angleDeg;
      btn.className = "power-btn" + (selected ? " selected" : "") + (useExpr ? " angle-expr" : "");
    });
    return;
  }

  powerOptionsEl.innerHTML = "";
  state.serveAngleOptions.forEach((angleDeg, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.labelGen = String(gen);
    const selected = state.selectedServeAngle === angleDeg;
    btn.className = "power-btn" + (selected ? " selected" : "") + (useExpr ? " angle-expr" : "");
    btn.textContent = state.serveAngleLabelCache[i];
    btn.addEventListener("click", () => {
      state.selectedServeAngle = angleDeg;
      syncServeControlPanel();
      state.message = "Press Space when the bar is at the power you want.";
      updateHud();
    });
    powerOptionsEl.appendChild(btn);
  });
}

function syncServeControlPanel() {
  if (state.gameOver) {
    clearServePanel();
    return;
  }
  if (state.awaitingServe && state.server === "player") {
    if (serveControlsLabel) {
      serveControlsLabel.textContent =
        state.selectedServeAngle == null ? "Serve angle (1\u20134)" : "Time your power \u2014 Space";
    }
    renderServeAngleButtons();
    if (servePowerBarBlock) {
      servePowerBarBlock.hidden = state.selectedServeAngle == null;
    }
    if (state.selectedServeAngle != null) updateServePowerBarDom();
  } else {
    clearServePanel();
  }
}

function updateHud() {
  const sLabel = state.server === "player" ? "You" : "CPU";
  const sScore = state.server === "player" ? state.playerScore : state.cpuScore;
  const rScore = state.server === "player" ? state.cpuScore : state.playerScore;
  scoreEl.textContent =
    `${sLabel} ${sScore}-${rScore}  \u00B7  You ${state.playerScore} - ${state.cpuScore} CPU (singles)`;
  statusEl.textContent = state.message;
}

/* -------------------------------------------------------------------- */
/* Rally end / message formatting                                        */
/* -------------------------------------------------------------------- */

function rallyEndMessage(reason, winner, lastHitter, bouncedOnSide) {
  const lastWord = lastHitter === "player" ? "Your" : "CPU";
  const loser = bouncedOnSide;
  const loserPossessive = loser === "player" ? "your" : "CPU";
  const lastPossessivePerson = lastHitter === "player" ? "Your" : "CPU";
  switch (reason) {
    case "volley_before_bounce": {
      const who = winner === "player" ? "CPU" : "You";
      return `${who} volleyed before the bounce - fault.`;
    }
    case "kitchen_volley": {
      const who = winner === "player" ? "CPU" : "You";
      return `${who} contacted in kitchen before bounce - fault.`;
    }
    case "net_hit":
      return `${lastWord} shot hit the net.`;
    case "off_court_after_bounce_in":
      return `${lastWord} shot: first bounce in, then out of play \u2014 point ${winner === "player" ? "you" : "CPU"}.`;
    case "off_court":
      return `${lastWord} shot flew off the court.`;
    case "second_bounce_out_after_in":
      return `${lastWord} shot: first bounce in, second out \u2014 point ${winner === "player" ? "you" : "CPU"}.`;
    case "out":
      return `${lastWord} shot landed out.`;
    case "didnt_clear_net":
      return `${lastWord} shot didn't clear the net.`;
    case "serve_in_kitchen":
      return `${lastPossessivePerson} serve landed in the kitchen - out.`;
    case "second_bounce":
      return `${loser === "player" ? "You" : "CPU"} failed to return.`;
    case "ball_died":
      return `Ball died on ${loserPossessive} side.`;
    default:
      return `Rally ended (${reason}).`;
  }
}

function endRallyWithMessages(winner, reason) {
  // Snapshot ball metadata BEFORE rally end resets ball state.
  const lastHitter = ball.lastHitter;
  const bouncedOnSide = ball.bouncedOnSide;
  state.message = rallyEndMessage(reason, winner, lastHitter, bouncedOnSide);

  // Pro mode: aggregate samples and run BC/DAgger update before scoring.
  if (state.cpuMode === CPU_MODE_PRO) {
    const rallyLen = clamp(state.hitsThisRally / 10, 0, 1);
    const learnGain = winner === "player"
      ? 0.028 + 0.022 * rallyLen
      : 0.01 + 0.012 * rallyLen;
    state.cpuSkill = clamp(state.cpuSkill + learnGain, 0, 1);
    if (typeof CpuPolicy !== "undefined" && state.mlRallySamples.length > 0) {
      const nFrames = state.mlRallySamples.length;
      for (let i = 0; i < state.mlRallySamples.length; i += 1) {
        const s = state.mlRallySamples[i];
        state.mlDataset.push(s);
        if (s.hardState >= 0.62) state.mlDataset.push(s);
        if (s.hardState >= 0.84) state.mlDataset.push(s);
      }
      if (state.mlDataset.length > MAX_DAGGER_DATASET) {
        state.mlDataset.splice(0, state.mlDataset.length - MAX_DAGGER_DATASET);
      }
      const batch = sampleDaggerBatch();
      const gateEval = sampleUniformBatch(state.mlDataset, DAGGER_EVAL_BATCH_SIZE);
      let keptUpdate = true;
      if (batch.length > 0 && gateEval.length > 0) {
        const snapshot = CpuPolicy.exportModelObject();
        const beforeScore = policyExpertMatchOn(gateEval);
        CpuPolicy.trainRally(batch);
        const afterScore = policyExpertMatchOn(gateEval);
        keptUpdate = afterScore >= beforeScore - DAGGER_KEEP_DELTA;
        if (!keptUpdate) CpuPolicy.importModelObject(snapshot);
      } else if (batch.length > 0) {
        CpuPolicy.trainRally(batch);
      }
      CpuPolicy.save();
      state.ml.policyGradSteps += 1;
      state.ml.lastRallyFrames = nFrames;
      const dynamicBetaMin = clamp(0.05 + 0.22 * (1 - state.cpuSkill), DAGGER_BETA_MIN, 0.27);
      state.ml.daggerBeta = Math.max(dynamicBetaMin, state.ml.daggerBeta * DAGGER_BETA_DECAY);
      if (!keptUpdate) {
        state.message = "Pro update rejected by validation gate; kept previous checkpoint.";
      }
    }
    state.mlRallySamples = [];
    state.mlTick = 0;
  }

  const result = simEndRally(state, ball, player, cpu, winner);
  if (result.gameOver) {
    state.message = result.playerScore > result.cpuScore
      ? `Game! You win ${result.playerScore}-${result.cpuScore}.`
      : `Game! CPU wins ${result.cpuScore}-${result.playerScore}.`;
  } else if (result.pointAwardedTo) {
    state.message += ` Point ${result.pointAwardedTo === "player" ? "you" : "CPU"}.`;
  } else if (result.sideOutTo) {
    state.message += ` Side-out - ${result.sideOutTo === "player" ? "you" : "CPU"} now serve.`;
  }
  invalidateServeAngleLabelCache();
  syncServeControlPanel();
  if (!state.gameOver && state.server === "cpu") {
    state.message += " Press Space when ready for CPU serve.";
  }
  updateHud();
}

/* -------------------------------------------------------------------- */
/* Serve adapters (browser path)                                         */
/* -------------------------------------------------------------------- */

function serveByPlayer() {
  if (state.training.active) return;
  if (state.gameOver || !state.awaitingServe || state.server !== "player") return;
  if (state.selectedServeAngle == null) {
    state.message = "Pick a serve angle (1\u20134 or a button) before serving.";
    updateHud();
    return;
  }
  const displayPower = serveOscillatorPowerInt();
  const effPower = servePowerEffective(displayPower);
  const meta = simServeByPlayer(state, ball, player, cpu, displayPower, state.selectedServeAngle, rng);
  state.message =
    `You serve - ${meta.shotType} \u2014 power ${effPower}, angle ${meta.angle.toFixed(0)}\u00B0 (timed ${displayPower}).`;
  syncServeControlPanel();
  updateHud();
}

function serveByCpu() {
  if (state.gameOver || !state.awaitingServe || state.server !== "cpu") return;
  const meta = simServeByCpu(state, ball, player, cpu, rng, cpuSkill01());
  const effPower = servePowerEffective(meta.displayPower);
  state.message =
    `CPU serves - ${meta.shotType} \u2014 power ${effPower}, angle ${Number(meta.angle).toFixed(0)}\u00B0 (timed ${meta.displayPower}).`;
  if (!state.training.active) updateHud();
}

function serveByPlayerTraining() {
  if (state.gameOver || !state.awaitingServe || state.server !== "player") return;
  simServeByPlayerTraining(state, ball, player, cpu, rng);
}

/* -------------------------------------------------------------------- */
/* Per-frame contact -> message bridge                                   */
/* -------------------------------------------------------------------- */

function applyContactMessage(contact) {
  state.message =
    `${contact.who === "player" ? "You" : "CPU"} return - ${contact.intent} \u2014 power ${contact.power}, angle ${contact.angle.toFixed(0)}\u00B0.`;
  updateHud();
}

function physicsStepBrowser() {
  if (state.awaitingServe) {
    placeBallOnServer(state, ball, player, cpu);
    return;
  }
  const oracleFn = state.cpuMode === CPU_MODE_PRO ? inferProShotIntent : null;
  const ev = physicsStep(state, ball, player, cpu, rng, oracleFn, cpuSkill01());
  if (ev.type === "rallyEnd") {
    endRallyWithMessages(ev.winner, ev.reason);
  } else if (ev.type === "contact") {
    applyContactMessage(ev.contact);
  }
}

/* -------------------------------------------------------------------- */
/* CPU update + ML hooks                                                 */
/* -------------------------------------------------------------------- */

function updateCpu() {
  if (state.gameOver) return;
  cpu.swingPhase = Math.max(cpu.swingPhase, 0);
  const skill = cpuSkill01();
  const proBonus = state.cpuMode === CPU_MODE_PRO ? 1.2 * skill : 0;
  const stepSpeed = CPU_SPEED + proBonus;
  const stepSpeedY = CPU_VERTICAL_SPEED + proBonus * (CPU_VERTICAL_SPEED / CPU_SPEED);
  const idleStep = 2.4;
  const idleStepY = idleStep * (CPU_VERTICAL_SPEED / CPU_SPEED);

  if (state.awaitingServe) {
    if (state.server === "cpu") {
      cpu.x += clamp(CPU_SERVE_DEFAULT_X - cpu.x, -idleStep, idleStep);
      cpu.x = clamp(cpu.x, CPU_SERVE_MIN_X, CPU_SERVE_MAX_X);
    } else {
      cpu.x += clamp(CPU_BASE_X - cpu.x, -idleStep, idleStep);
      cpu.x = clamp(cpu.x, CPU_RALLY_MIN_X, CPU_RALLY_MAX_X);
    }
    const dyIdle = (GROUND_Y - PADDLE_PLAY_H / 2) - cpu.y;
    cpu.y += clamp(dyIdle, -idleStepY, idleStepY);
    const minServeY = state.server === "cpu" ? SERVE_PADDLE_REACH_TOP : PADDLE_REACH_TOP;
    cpu.y = clamp(cpu.y, minServeY, PADDLE_REACH_BOTTOM);
    return;
  }

  if (state.cpuMode === CPU_MODE_PRO && typeof CpuPolicy !== "undefined") {
    CpuPolicy.ensureLoaded();
    const ht = getCpuHeuristicTarget(state, ball, cpu);
    if (ht.hasTarget) {
      state.cpuPlannedShotType = pickShotIntent("cpu", cpu, ball, state, rng, inferProShotIntent, skill);
      extractCpuMLFeatures(_cpuFeat, ball, cpu, player, state, true, ht.tx, ht.ty);
      const expertAction = chooseExpertCpuAction(state, ball, cpu, ht.tx, ht.ty, stepSpeed, stepSpeedY);
      const explore = state.cpuSkill < 0.95 && rng.random() < Math.max(0.1, 0.48 - 0.4 * skill);
      let shipped = inferProCpuAction(_cpuFeat);
      let policyAction = shipped.action;
      if (!(policyAction >= 0 && policyAction < CpuPolicy.ACTION_DIM)) {
        CpuPolicy.forward(_cpuFeat, _cpuPolOut);
        const temp = Math.max(0.5, 1.05 - 0.75 * skill);
        policyAction = explore
          ? CpuPolicy.sampleAction(_cpuPolOut.probs, temp)
          : CpuPolicy.argmax(_cpuPolOut.probs);
        shipped = { trusted: false, reason: "cpu_policy_fallback" };
      }
      const forceExpert =
        shipped &&
        shipped.trusted === false &&
        (shipped.reason === "ood" || shipped.reason === "low_confidence" || shipped.reason === "nan");
      const useExpert = forceExpert || rng.random() < state.ml.daggerBeta;
      const action = useExpert ? expertAction : policyAction;
      const [mx, my] = CpuPolicy.actionToDelta(action);
      const hardState = estimateHardStateScore(state, ball, cpu, ht.tx, ht.ty);
      if (
        globalThis.__pickleballRecording &&
        typeof globalThis.__pickleballRecording.recordFeatureRow === "function"
      ) {
        globalThis.__pickleballRecording.recordFeatureRow(_cpuFeat, action, expertAction, {
          policy_action: policyAction,
          use_expert: useExpert ? 1 : 0,
          episode_id: state.mlEpisodeId,
          frame_idx: state.mlTick
        });
      }
      state.mlTick += 1;
      if (state.mlRallySamples.length < 200 && state.mlTick % 2 === 0) {
        state.mlRallySamples.push({
          x: Float32Array.from(_cpuFeat),
          action,
          expertAction,
          hardState
        });
      }
      cpu.x = clamp(cpu.x + mx * stepSpeed, CPU_RALLY_MIN_X, CPU_RALLY_MAX_X);
      cpu.y = clamp(cpu.y + my * stepSpeedY, PADDLE_REACH_TOP, PADDLE_REACH_BOTTOM);
      return;
    }
  } else if (ball.vx > 0.5 && ball.lastHitter !== "cpu") {
    if (mustLetBallBounceBeforeReturn(state, ball)) {
      const land = predictFirstGroundBouncePoint(ball);
      if (land) {
        const trackingNoise = rng.spread(16);
        const tx = clamp(
          land.x + CPU_HEURISTIC_LAND_BIAS_X + trackingNoise,
          NET_X + CPU_HEURISTIC_NET_GUARD_X,
          CPU_RALLY_MAX_X
        );
        const ty = clamp(land.y, PADDLE_REACH_TOP, PADDLE_REACH_BOTTOM);
        const dx = tx - cpu.x;
        const dy = ty - cpu.y;
        cpu.x += clamp(dx, -stepSpeed, stepSpeed);
        cpu.y += clamp(dy, -stepSpeedY, stepSpeedY);
        cpu.x = clamp(cpu.x, CPU_RALLY_MIN_X, CPU_RALLY_MAX_X);
        return;
      }
    } else {
      const target = predictBallAtCpu(state, ball);
      if (target) {
        const trackingNoise = rng.spread(12);
        const tx = clamp(target.x + trackingNoise, CPU_RALLY_MIN_X, CPU_RALLY_MAX_X);
        const ty = clamp(target.y + rng.spread(2.4), PADDLE_REACH_TOP, PADDLE_REACH_BOTTOM);
        const dx = tx - cpu.x;
        const dy = ty - cpu.y;
        cpu.x += clamp(dx, -stepSpeed, stepSpeed);
        cpu.y += clamp(dy, -stepSpeedY, stepSpeedY);
        cpu.x = clamp(cpu.x, CPU_RALLY_MIN_X, CPU_RALLY_MAX_X);
        return;
      }
    }
  }

  const dxN = CPU_BASE_X - cpu.x;
  const dyN = (GROUND_Y - PADDLE_PLAY_H / 2) - cpu.y;
  cpu.x += clamp(dxN, -2, 2);
  cpu.y += clamp(dyN, -idleStepY, idleStepY);
  cpu.x = clamp(cpu.x, CPU_RALLY_MIN_X, CPU_RALLY_MAX_X);
}

function updatePlayerInput() {
  if (state.gameOver || state.training.active) return;
  let dx = 0, dy = 0;
  if (heldKeys.left) dx -= PLAYER_SPEED;
  if (heldKeys.right) dx += PLAYER_SPEED;
  if (heldKeys.up) dy -= PLAYER_VERTICAL_SPEED;
  if (heldKeys.down) dy += PLAYER_VERTICAL_SPEED;
  if (dx !== 0 || dy !== 0) {
    let minX;
    let maxX;
    if (state.awaitingServe) {
      if (state.server === "player") {
        minX = PLAYER_SERVE_MIN_X;
        maxX = PLAYER_SERVE_MAX_X;
      } else {
        minX = PLAYER_RALLY_MIN_X;
        maxX = PLAYER_RALLY_MAX_X;
      }
    } else {
      minX = PLAYER_RALLY_MIN_X;
      maxX = PLAYER_RALLY_MAX_X;
    }
    const minY =
      state.awaitingServe && state.server === "player" ? SERVE_PADDLE_REACH_TOP : PADDLE_REACH_TOP;
    player.x = clamp(player.x + dx, minX, maxX);
    player.y = clamp(player.y + dy, minY, PADDLE_REACH_BOTTOM);
  }
}

/* -------------------------------------------------------------------- */
/* Rendering                                                             */
/* -------------------------------------------------------------------- */

function drawCourt() {
  ctx.clearRect(0, 0, WIDTH, HEIGHT);

  ctx.save();
  ctx.globalAlpha = 0.04;
  ctx.fillStyle = "#5fff8a";
  for (let y = 0; y < HEIGHT; y += 3) ctx.fillRect(0, y, WIDTH, 1);
  ctx.restore();

  ctx.shadowColor = "#5fff8a";
  ctx.shadowBlur = 8;
  ctx.strokeStyle = "#5fff8a";
  ctx.lineWidth = 1.6;

  ctx.beginPath();
  ctx.moveTo(COURT_LEFT, GROUND_Y);
  ctx.lineTo(COURT_RIGHT, GROUND_Y);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(COURT_LEFT, GROUND_Y);
  ctx.lineTo(COURT_LEFT, GROUND_Y - 10);
  ctx.moveTo(COURT_RIGHT, GROUND_Y);
  ctx.lineTo(COURT_RIGHT, GROUND_Y - 10);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(NET_X, NET_TOP - 4);
  ctx.lineTo(NET_X, GROUND_Y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(NET_X - 6, NET_TOP);
  ctx.lineTo(NET_X + 6, NET_TOP);
  ctx.stroke();

  ctx.shadowBlur = 0;
  ctx.strokeStyle = "rgba(95, 255, 138, 0.45)";
  ctx.lineWidth = 0.6;
  for (let y = NET_TOP + 3; y < GROUND_Y; y += 4) {
    ctx.beginPath();
    ctx.moveTo(NET_X - 4, y);
    ctx.lineTo(NET_X + 4, y);
    ctx.stroke();
  }
  for (let dx = -4; dx <= 4; dx += 2) {
    ctx.beginPath();
    ctx.moveTo(NET_X + dx, NET_TOP);
    ctx.lineTo(NET_X + dx, GROUND_Y);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(255, 90, 90, 0.16)";
  ctx.fillRect(NET_X - KITCHEN_HALF, GROUND_Y - 14, KITCHEN_HALF, 14);
  ctx.fillRect(NET_X, GROUND_Y - 14, KITCHEN_HALF, 14);

  ctx.strokeStyle = "rgba(255, 90, 90, 0.8)";
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(NET_X - KITCHEN_HALF, GROUND_Y);
  ctx.lineTo(NET_X - KITCHEN_HALF, GROUND_Y - 14);
  ctx.moveTo(NET_X + KITCHEN_HALF, GROUND_Y);
  ctx.lineTo(NET_X + KITCHEN_HALF, GROUND_Y - 14);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawTrail() {
  if (ball.trail.length < 2) return;
  ctx.save();
  ctx.shadowColor = "#fde68a";
  ctx.shadowBlur = 6;
  for (let i = 1; i < ball.trail.length; i += 1) {
    const a = ball.trail[i - 1];
    const b = ball.trail[i];
    const alpha = (i / ball.trail.length) * 0.55;
    ctx.strokeStyle = `rgba(253, 230, 138, ${alpha.toFixed(3)})`;
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPaddle(p, isPlayer) {
  const dir = isPlayer ? 1 : -1;
  const swing = p.swingPhase;
  const swingTilt = swing * dir * 0.6;

  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.rotate(swingTilt);

  ctx.shadowColor = isPlayer ? "#5fff8a" : "#fde68a";
  ctx.shadowBlur = 7;
  ctx.strokeStyle = isPlayer ? "#5fff8a" : "#fde68a";
  ctx.fillStyle = isPlayer ? "rgba(95, 255, 138, 0.18)" : "rgba(253, 230, 138, 0.18)";
  ctx.lineWidth = 1.6;

  ctx.beginPath();
  ctx.ellipse(0, 0, PADDLE_PLAY_W / 2 + 2, PADDLE_PLAY_H / 2, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = isPlayer ? "rgba(95, 255, 138, 0.6)" : "rgba(253, 230, 138, 0.55)";
  for (let dy = -PADDLE_PLAY_H / 2 + 4; dy <= PADDLE_PLAY_H / 2 - 4; dy += 5) {
    ctx.beginPath();
    ctx.arc(0, dy, 0.7, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.shadowBlur = 0;
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fillRect(-1.5, PADDLE_PLAY_H / 2, 3, 8);
  ctx.fillStyle = "#dc2626";
  ctx.fillRect(-1.5, PADDLE_PLAY_H / 2 + 4, 3, 4);

  ctx.restore();
}

function drawBall() {
  ctx.save();
  ctx.shadowColor = "#fde68a";
  ctx.shadowBlur = 12;
  ctx.fillStyle = "#fde68a";
  ctx.beginPath();
  ctx.arc(ball.x, ball.y, BALL_R, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawHud() {
  ctx.save();
  ctx.font = "11px JetBrains Mono, ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(95, 255, 138, 0.85)";
  const bouncesLabel =
    state.bouncesNeeded === 2 ? "Two-bounce: serve must bounce" :
    state.bouncesNeeded === 1 ? "Two-bounce: return must bounce" :
    "Volleys allowed (mind the kitchen)";
  ctx.fillText(bouncesLabel, 14, 18);

  if (state.cpuMode === CPU_MODE_PRO && state.rallyActive && !state.gameOver) {
    ctx.font = "10px JetBrains Mono, ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(147, 197, 253, 0.92)";
    ctx.fillText(`ML trajectory buffer: ${state.mlRallySamples.length} / 200 frames`, 14, 32);
  }

  ctx.font = "11px JetBrains Mono, ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillStyle = "rgba(253, 230, 138, 0.9)";
  let powerHud;
  if (state.awaitingServe && state.server === "player") {
    if (state.selectedServeAngle == null) {
      powerHud = "Serve angle \u2014";
    } else {
      powerHud = `${state.selectedServeAngle}\u00B0  ${serveOscillatorPowerInt()}`;
    }
  } else {
    powerHud = `Return pwr ${returnPowerFromPaddleY(player.y)}`;
  }
  ctx.fillText(powerHud, WIDTH - 14, 18);

  if (state.lastShotType) {
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(220, 252, 231, 0.7)";
    ctx.fillText(`Last: ${state.lastShotType}`, WIDTH / 2, 18);
  }

  if (state.awaitingServe && !state.gameOver) {
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(95, 255, 138, 0.75)";
    const hint =
      state.server === "player"
        ? state.selectedServeAngle == null
          ? "Pick angle 1\u20134, then Space on the bar"
          : "Space \u2014 serve with current bar level"
        : "Space when ready for CPU serve";
    ctx.fillText(hint, WIDTH / 2, HEIGHT - 18);
  }
  if (!state.gameOver && !state.awaitingServe) {
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(253, 230, 138, 0.95)";
    ctx.fillText(
      `Soft ${state.playerSoftShotQueued ? "ON" : "off"} (Q) \u2014 ${
        state.playerSoftShotQueued
          ? "kitchen dink / box drop (any paddle height)"
          : "high smash \u00B7 mid drive \u00B7 low lob"
      }`,
      14,
      HEIGHT - 18
    );
  }
  ctx.restore();
}

function runBallPhysicsSteps() {
  let steps = 0;
  while (physicsStepAccum >= 1 && steps < MAX_PHYSICS_STEPS_PER_FRAME) {
    physicsStepAccum -= 1;
    physicsStepBrowser();
    if (!state.gameOver) updateCpu();
    steps += 1;
  }
}

function gameLoop() {
  player.swingPhase = Math.max(0, player.swingPhase - 0.08);
  cpu.swingPhase = Math.max(0, cpu.swingPhase - 0.08);

  if (!state.gameOver) updatePlayerInput();

  if (!state.gameOver) {
    if (state.awaitingServe) {
      placeBallOnServer(state, ball, player, cpu);
      updateCpu();
    } else {
      physicsStepAccum += BROWSER_ANIMATION_TIME_SCALE;
      runBallPhysicsSteps();
    }
  }

  drawCourt();
  drawTrail();
  drawPaddle(player, true);
  drawPaddle(cpu, false);
  drawBall();
  drawHud();

  if (state.awaitingServe && state.server === "player" && state.selectedServeAngle != null) {
    updateServePowerBarDom();
  }

  requestAnimationFrame(gameLoop);
}

function resetGame() {
  state.playerScore = 0;
  state.cpuScore = 0;
  state.server = "player";
  state.gameOver = false;
  state.awaitingServe = true;
  state.rallyActive = false;
  state.bouncesNeeded = 2;
  state.hitsThisRally = 0;
  state.lastShotType = "";
  state.contactCooldown = 0;
  state.cpuPlannedShotType = null;
  state.playerSoftShotQueued = false;
  physicsStepAccum = 0;
  state.cpuSkill = state.cpuMode === CPU_MODE_PRO ? 0.12 : 0;
  state.mlRallySamples = [];
  state.mlTick = 0;
  invalidateServeAngleLabelCache();
  state.message = "Pick serve angle 1\u20134, then press Space when the power bar is where you want it.";
  resetServePositions(state, player, cpu);
  placeBallOnServer(state, ball, player, cpu);
  syncServeControlPanel();
  syncCpuModeBtnStyles();
  updateHud();
}

/* -------------------------------------------------------------------- */
/* Input                                                                 */
/* -------------------------------------------------------------------- */

function setHeldKey(key, isDown) {
  if (key === "ArrowLeft") heldKeys.left = isDown;
  else if (key === "ArrowRight") heldKeys.right = isDown;
  else if (key === "ArrowUp") heldKeys.up = isDown;
  else if (key === "ArrowDown") heldKeys.down = isDown;
  else return false;
  return true;
}

window.addEventListener("keydown", (event) => {
  if (state.gameOver || state.training.active) return;

  const k = String(event.key || "").toLowerCase();
  if (k === "q") {
    event.preventDefault();
    state.playerSoftShotQueued = !state.playerSoftShotQueued;
    state.message = state.playerSoftShotQueued
      ? "Soft on: dink at kitchen, drop in box (any paddle height)."
      : "Soft off: high smash, mid drive, low lob.";
    updateHud();
    return;
  }

  if (setHeldKey(event.key, true)) {
    event.preventDefault();
    return;
  }

  if (["1", "2", "3", "4"].includes(event.key)) {
    if (!(state.awaitingServe && state.server === "player")) return;
    const idx = Number(event.key) - 1;
    ensureServeAngleOptions();
    const ang = state.serveAngleOptions[idx];
    if (ang != null) {
      event.preventDefault();
      state.selectedServeAngle = ang;
      syncServeControlPanel();
      state.message = "Press Space when the bar is at the power you want.";
      updateHud();
    }
    return;
  }

  if (event.key === " ") {
    if (state.awaitingServe && state.server === "player") {
      event.preventDefault();
      serveByPlayer();
    } else if (state.awaitingServe && state.server === "cpu") {
      event.preventDefault();
      serveByCpu();
    }
  }
});

window.addEventListener("keyup", (event) => {
  if (setHeldKey(event.key, false)) event.preventDefault();
});

window.addEventListener("blur", () => {
  heldKeys.left = heldKeys.right = heldKeys.up = heldKeys.down = false;
});

newGameBtn.addEventListener("click", () => {
  if (state.training.active) return;
  resetGame();
});

document.querySelectorAll(".cpu-mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => setCpuMode(btn.dataset.cpuMode));
});
if (typeof CpuPolicy !== "undefined") {
  const models = loadCpuModelList();
  saveCpuModelList(models);
  refreshCpuModelSelectUI();
  const activeModel = models.includes(getActiveCpuModelId()) ? getActiveCpuModelId() : DEFAULT_CPU_MODEL_ID;
  CpuPolicy.setModelId(activeModel);
  CpuPolicy.ensureLoaded();
  setActiveCpuModelId(activeModel);
}
if (cpuModelSelect) {
  cpuModelSelect.addEventListener("change", () => switchCpuModel(cpuModelSelect.value));
}
syncCpuModeBtnStyles();
syncCpuProUiVisibility();

resetGame();
gameLoop();
