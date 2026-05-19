/**
 * Pure, environment-agnostic simulation constants.
 *
 * All gameplay physics, court geometry, and rules constants live here so
 * that browser and headless runners share a single source of truth.
 *
 * Court layout uses paddle height (`PADDLE_H`) as the standard unit (30px):
 *   kitchen (NVZ depth) = 7 paddles, box = 15 paddles, half-court depth = 22 paddles,
 *   net body height = 4 paddles, max contact band = 8 paddles, serve band = 4 paddles,
 *   ball radius = 1/4 paddle.
 *
 * On-court paddle sprites and hitboxes use `PADDLE_PLAY_*` (2×) — do not use those
 * for court geometry.
 *
 * Important: do NOT import any DOM/canvas/window APIs here. This module is
 * imported by both the browser game and the Node offline simulator.
 */

/** Canonical court-scaling unit (pixels). Not the on-screen paddle height. */
export const PADDLE_H = 30;
export const PADDLE_W = 12;

/** Gameplay paddle dimensions (render + collision). */
export const PADDLE_PLAY_H = PADDLE_H * 2;
export const PADDLE_PLAY_W = PADDLE_W * 2;

export const KITCHEN_DEPTH_PADDLES = 7;
export const BOX_DEPTH_PADDLES = 15;
export const HALF_COURT_PADDLES = KITCHEN_DEPTH_PADDLES + BOX_DEPTH_PADDLES;

export const NET_HEIGHT_PADDLES = 4;
export const MAX_HIT_PADDLES = 8;
export const SERVE_HEIGHT_PADDLES = 4;

export const KITCHEN_HALF = KITCHEN_DEPTH_PADDLES * PADDLE_H;
export const BOX_DEPTH = BOX_DEPTH_PADDLES * PADDLE_H;
export const HALF_COURT_DEPTH = HALF_COURT_PADDLES * PADDLE_H;

export const NET_BODY_HEIGHT = NET_HEIGHT_PADDLES * PADDLE_H;
export const MAX_CONTACT_HEIGHT = MAX_HIT_PADDLES * PADDLE_H;
/** Allowed vertical span (from baseline reach) while serving. */
export const SERVE_CONTACT_HEIGHT = SERVE_HEIGHT_PADDLES * PADDLE_H;

export const BALL_R = PADDLE_H / 4;
export const CONTACT_PAD = 4;

/** Horizontal runback: canvas edge → baseline on each side, in paddle-heights. */
export const BASELINE_BACKSPACE_PADDLES = 5;
export const COURT_SIDE_INSET = BASELINE_BACKSPACE_PADDLES * PADDLE_H;

export const WIDTH = HALF_COURT_DEPTH * 2 + COURT_SIDE_INSET * 2;
export const NET_X = WIDTH / 2;
export const COURT_LEFT = NET_X - HALF_COURT_DEPTH;
export const COURT_RIGHT = NET_X + HALF_COURT_DEPTH;

/** Half-court depth (net→baseline) from pre-paddle-unit tuning; used for margins and power scaling. */
export const LEGACY_HALF_COURT_DEPTH = 394;

/** Scale launch speeds so similar power levels reach comparable court fractions on the wider court. */
export const COURT_SPEED_SCALE = HALF_COURT_DEPTH / LEGACY_HALF_COURT_DEPTH;

export const PLAYER_BASE_X = COURT_LEFT + Math.round(PADDLE_H / 3);
export const CPU_BASE_X = COURT_RIGHT - Math.round(PADDLE_H / 3);

export const RUN_PAST_COURT = Math.round(HALF_COURT_DEPTH * (170 / LEGACY_HALF_COURT_DEPTH));
export const PLAYER_RALLY_MIN_X = Math.max(8, COURT_LEFT - RUN_PAST_COURT);
export const PLAYER_RALLY_MAX_X = NET_X - Math.round(12 * (HALF_COURT_DEPTH / LEGACY_HALF_COURT_DEPTH));
export const CPU_RALLY_MIN_X = NET_X + Math.round(20 * (HALF_COURT_DEPTH / LEGACY_HALF_COURT_DEPTH));
export const CPU_RALLY_MAX_X = Math.min(WIDTH - 8, COURT_RIGHT + RUN_PAST_COURT);

/** Horizontal strip where the server may stand before serve (behind own baseline). */
export const PLAYER_SERVE_MIN_X = Math.max(8, COURT_LEFT - RUN_PAST_COURT);
export const PLAYER_SERVE_MAX_X = COURT_LEFT;
export const CPU_SERVE_MIN_X = COURT_RIGHT;
export const CPU_SERVE_MAX_X = Math.min(WIDTH - 8, COURT_RIGHT + RUN_PAST_COURT);

export const PLAYER_SERVE_DEFAULT_X = (PLAYER_SERVE_MIN_X + PLAYER_SERVE_MAX_X) / 2;
export const CPU_SERVE_DEFAULT_X = (CPU_SERVE_MIN_X + CPU_SERVE_MAX_X) / 2;

/** Net plane collision slab half-width (horizontal). */
export const NET_PLANE_HALF_WIDTH = Math.max(4, Math.round((4 * PADDLE_H) / 30));
export const NET_TAPE_CLIP_PX = Math.max(2, Math.round((2 * PADDLE_H) / 30));

/** CPU intercept targets: preserve legacy offsets relative to half-court depth. */
export const CPU_HEURISTIC_LAND_BIAS_X = Math.round(22 * (HALF_COURT_DEPTH / LEGACY_HALF_COURT_DEPTH));
export const CPU_HEURISTIC_NET_GUARD_X = Math.round(24 * (HALF_COURT_DEPTH / LEGACY_HALF_COURT_DEPTH));

/** Canvas layout: ground line and net tape derived from paddle-sized net body. */
export const CANVAS_BOTTOM_MARGIN = 56;
/** Vertical space above the net tape for ball flight (legacy layout headroom). */
export const SKY_ABOVE_NET_TOP = 444;
export const HEIGHT = CANVAS_BOTTOM_MARGIN + NET_BODY_HEIGHT + SKY_ABOVE_NET_TOP;
export const GROUND_Y = HEIGHT - CANVAS_BOTTOM_MARGIN;
export const NET_TOP = GROUND_Y - NET_BODY_HEIGHT;

export const PADDLE_REACH_TOP = Math.min(GROUND_Y - MAX_CONTACT_HEIGHT, NET_TOP - 2 * PADDLE_PLAY_H);
export const PADDLE_REACH_BOTTOM = GROUND_Y - PADDLE_PLAY_H / 2 + 4;
export const SERVE_PADDLE_Y_RANGE = SERVE_CONTACT_HEIGHT;
export const SERVE_PADDLE_REACH_TOP = PADDLE_REACH_BOTTOM - SERVE_PADDLE_Y_RANGE;

export const GRAVITY = 0.18;
/** Multiplier on gravity for the ball only (>1 = heavier, faster drop, lower arc apex). */
export const BALL_WEIGHT = 1.12;

/**
 * USAP ball bounce test: drop 78 in → rebound 30–34 in on approved surface.
 * COR (rebound_v / inbound_v) = sqrt(rebound_height / drop_height).
 */
export const USAP_BALL_DROP_HEIGHT_IN = 78;
export const USAP_BALL_REBOUND_MIN_IN = 30;
export const USAP_BALL_REBOUND_MAX_IN = 34;
export const BALL_COURT_COR_MIN = Math.sqrt(USAP_BALL_REBOUND_MIN_IN / USAP_BALL_DROP_HEIGHT_IN);
export const BALL_COURT_COR_MAX = Math.sqrt(USAP_BALL_REBOUND_MAX_IN / USAP_BALL_DROP_HEIGHT_IN);
/** Hard-court ~57% kinetic-energy loss → ~43% retained → COR ≈ sqrt(0.43). */
export const BALL_COURT_ENERGY_RETAINED = 0.43;
export const BALL_COURT_COR_ENERGY = Math.sqrt(BALL_COURT_ENERGY_RETAINED);

/**
 * Paddle–ball COR (PBCoR): USAP limits effective trampoline to ~0.43–0.46;
 * true physical COR with flex/deformation is closer to ~0.70 (not simulated here).
 * Launch speeds are shot-tuned; this is metadata for contracts/docs.
 */
export const PADDLE_BALL_COR_USAP = 0.445;
export const PADDLE_BALL_COR_PHYSICAL = 0.7;

/** @deprecated alias — use BALL_COURT_COR_ENERGY for floor physics */
export const RESTITUTION = BALL_COURT_COR_ENERGY;

/** Legacy per-frame air drag before court scale (see ml/sim/tune_physics.js). */
export const LEGACY_AIR_DRAG = 0.995;

/**
 * In-flight physics scaled for the paddle-sized court.
 * Launch speeds use COURT_SPEED_SCALE; gravity/drag scale so hang time and arcs
 * stay proportional to court width (see ml/sim/tune_physics.js).
 */
export const PHYSICS_COURT_SCALE = HALF_COURT_DEPTH / LEGACY_HALF_COURT_DEPTH;
export const BALL_GRAVITY = GRAVITY * BALL_WEIGHT * Math.pow(PHYSICS_COURT_SCALE, 0.86);
/** Per-frame air drag (0.995 legacy); court-wide scaling over-damped long serves. */
export const AIR_DRAG = LEGACY_AIR_DRAG;

/** Floor COR band (USAP ball test); glancing hits interpolate toward COR_MIN in ballMotion.js. */
export const FLOOR_BOUNCE_VY_MUL = BALL_COURT_COR_MIN;
export const FLOOR_BOUNCE_VY_MUL_MAX = BALL_COURT_COR_MAX;
export const FLOOR_BOUNCE_SPEED_REF = 6;
/** Tangential speed retained on bounce (coupled to ~43% energy on horizontal skim). */
export const FLOOR_BOUNCE_VX_MUL = Math.sqrt(BALL_COURT_ENERGY_RETAINED);
export const FLOOR_BOUNCE_MIN_VY = 0.65;
/** Horizontal friction when the ball skids to rest (no vertical bounce). */
export const FLOOR_ROLL_VX_MUL = 0.85;

export const WIN_SCORE = 11;
/** Pre–paddle-unit contact band; used to scale vertical paddle speed. */
export const LEGACY_MAX_CONTACT_HEIGHT = 110;
/**
 * Browser-only real-time stretch for the ball (see game.js fixed-step accumulator).
 * 1 = real-time sim cadence; 0.78 ≈ 28% more wall-clock time to react. Does not change
 * launch speed, gravity, drag, or bounce formulas — only how many sim steps run per second.
 */
export const BROWSER_ANIMATION_TIME_SCALE = 0.78;

/**
 * Faster paddle up/down on the 8-paddle contact band (player reaction time).
 * Does not change ball physics — only how quickly you can change shot height.
 */
export const PADDLE_VERTICAL_SPEED_BOOST = 1.08;
export const PADDLE_VERTICAL_SPEED_SCALE =
  (MAX_CONTACT_HEIGHT / LEGACY_MAX_CONTACT_HEIGHT) * PADDLE_VERTICAL_SPEED_BOOST;

export const CPU_SPEED = 2.65;
export const PLAYER_SPEED = 2.85;
export const CPU_VERTICAL_SPEED = CPU_SPEED * PADDLE_VERTICAL_SPEED_SCALE;
export const PLAYER_VERTICAL_SPEED = PLAYER_SPEED * PADDLE_VERTICAL_SPEED_SCALE;
export const RETURN_SPEED_MULT = 1.28;
export const NET_CLEAR_ANGLE_MAX = 70;
export const PLAYER_RETURN_ANGLE_MAX = 58;
export const CPU_RETURN_ANGLE_MAX = 50;

export const POWER_MIN = 0;
export const POWER_MAX = 100;
export const SERVE_POWER_BASE = 50;
export const SERVE_ANGLE_PRESETS = [26, 32, 38, 44];
export const SERVE_POWER_OSCILLATION_MS = 2600;
export const CPU_SERVE_DISPLAY_POWER_MIN = 62;
export const CPU_SERVE_DISPLAY_POWER_MAX = 94;
export const RETURN_POWER_BASE = 45;
export const RETURN_POSITION_WEIGHT = POWER_MAX - RETURN_POWER_BASE;

/**
 * Per-shot power/angle bands tuned for scaled court physics (ml/sim/tune_shots.js after tune_physics).
 * Targets: Dink → opp. kitchen, Drop → opp. box (short), Drive → deep box, Lob → baseline, Smash → attacking.
 */
export const SHOT_PROFILES = {
  Dink: { powerMin: 25, powerMax: 33, angleMin: 57, angleMax: 65 },
  Drop: { powerMin: 46, powerMax: 58, angleMin: 33, angleMax: 43 },
  Drive: { powerMin: 51, powerMax: 63, angleMin: 25, angleMax: 35 },
  Lob: { powerMin: 55, powerMax: 67, angleMin: 44, angleMax: 54 },
  Smash: { powerMin: 79, powerMax: 95, angleMin: -37, angleMax: -25 }
};

export const CPU_MODE_NORMAL = "normal";
export const CPU_MODE_PRO = "pro";

export const FEATURE_DIM = 24;
export const ACTION_GRID = 5;
export const ACTION_DIM = ACTION_GRID * ACTION_GRID;

/** Runtime contract metadata embedded in shipped models for compatibility checks. */
export const RUNTIME_CONTRACT = {
  feature_dim: FEATURE_DIM,
  action_dim: ACTION_DIM,
  schema_name: "pro_cpu_features_v1",
  schema_version: 1,
  physics: {
    gravity: GRAVITY,
    ball_weight: BALL_WEIGHT,
    ball_gravity: BALL_GRAVITY,
    air_drag: AIR_DRAG,
    ball_court_cor_min: BALL_COURT_COR_MIN,
    ball_court_cor_max: BALL_COURT_COR_MAX,
    ball_court_cor_energy: BALL_COURT_COR_ENERGY,
    paddle_ball_cor_usap: PADDLE_BALL_COR_USAP,
    floor_bounce_vx_mul: FLOOR_BOUNCE_VX_MUL,
    physics_court_scale: PHYSICS_COURT_SCALE
  }
};
