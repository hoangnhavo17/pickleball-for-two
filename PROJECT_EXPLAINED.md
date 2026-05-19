# Pickleball Project Explained (Beginner-Friendly)

## What this project is

This is a browser game called **Pickleball For Two**.

You can:
- play against a CPU
- switch between a normal scripted CPU and a learning CPU
- train/export/import CPU models

So this project is both:
- a playable game, and
- an AI/ML portfolio project.

---

## Big idea in simple terms

There are **two CPU modes**:

- **Normal mode**  
  Uses hand-written game logic (if this happens, do that).

- **Pro mode (ML)**  
  Uses a small neural network that tries to imitate an "expert" movement policy.

Think of it like:
- Normal = a fixed recipe
- Pro = a student that learns from examples

---

## How gameplay works

The game runs in a canvas with pickleball-like rules:

- serve flow
- two-bounce rule
- kitchen faults
- scoring and side-outs

You control:
- paddle movement (arrow keys)
- serve angle and power timing
- optional queued shot types (Drive/Lob/Drop/etc.) for returns

---

## How the CPU decides what to do

### 1) Movement decision

- In **Normal** mode, the CPU uses scripted prediction and movement.
- In **Pro** mode, the CPU builds a feature vector and chooses a movement action from a grid.

Action space is now **5x5 = 25 actions**, which gives smoother movement than the old 3x3.

### 2) Shot decision

On contact (returns), the game picks a shot intent:
- Dink, Drop, Drive, Lob, Smash

Each intent maps to power/angle ranges, then physics/constraints apply.

Note: serve logic is separate from return shot-intent logic.

---

## ML part (what is being trained)

The model learns to imitate an expert action label.

- Input: 24 features (`f0..f23`)
- Model: small MLP (24 -> hidden -> action logits)
- Output: action id (0..24)

Training uses behavior cloning + DAgger-style updates.

---

## Data pipeline (offline)

The simulator and training pipeline both run from the repo root:

1. `node ml/sim/run_offline_sim.js` (a.k.a. `npm run ml:simulate`)
   Headless gameplay simulator using the same `src/sim/*` core that the
   browser game uses. Writes `ml/data/raw/simulated_gameplay.csv`.

2. `node ml/sim/run_benchmark_ladders.js` (a.k.a. `npm run ml:benchmark`)
   Rating-tier benchmark across `2.5 / 3.0 / 3.5 / 4.0` style suites; writes
   `ml/artifacts/rating_benchmark.json`.

3. `ml/src/preprocess.py`
   Discovers all `*.csv` files in `ml/data/raw/`, upgrades any legacy
   schemas, and merges them into `ml/data/processed/train_ready.csv`.

4. `ml/src/train_behavior_clone.py`
   Trains the model and writes a checkpoint with metrics + metadata.

5. `ml/src/eval_rollout_benchmark.py`
   Computes rollout-style proxy metrics.

6. `ml/src/check_acceptance.py`
   Applies pass/fail gates from config (now includes `ladder_*` checks).

7. `ml/src/export_model.py`
   Exports browser-ready model JSON.

8. `ml/src/smoke_check_model.py`
   Verifies exported model structure.

9. `ml/src/write_training_report.py`
   Generates a readable report markdown file.

One command runs the pipeline:

```bash
npm run ml:build
```

Quick automated parity checks on the same code the browser uses for physics
and rules:

```bash
npm run ml:regression
```

---

## Why this is robust now

This project includes safeguards that many prototypes skip:

- one-time model load semantics (no accidental re-randomizing)
- schema contract for features
- runtime contract checks (schema + physics constants)
- confidence/OOD checks with safe fallback
- acceptance gates before export
- reproducibility metadata (seed, dataset hash, hyperparams)

---

## File map (quick orientation)

- `src/sim/` -> shared pure-JS simulation core (constants, physics, rules, step, rally)
- `src/game/game.js` -> browser adapter (canvas/UI/input/ML hooks) calling into `src/sim/`
- `src/ai/cpuPolicy.js` -> in-browser policy + online training hooks
- `src/ai/mlAI.js` -> shipped model loading/inference and safety checks
- `src/data/recordGameplay.js` -> feature/action recording export
- `ml/sim/run_offline_sim.js` -> headless dataset generator (uses `src/sim/`)
- `ml/sim/run_benchmark_ladders.js` -> rating-tier benchmark suite
- `ml/sim/config/sim_scenarios.json` -> tier definitions (skill, rallies, shot mix)
- `public/models/pro_cpu_v1/bea-haverkrone.json` -> **shipped** offline-trained weights (commit for itch.io)
- `ml/config/feature_schema.json` -> feature/action schema contract
- `ml/config/acceptance_gates.json` -> pass/fail thresholds (validation + ladder)

---

## In one sentence

This is a playable pickleball game where a scripted CPU and a learned CPU coexist, and the learned CPU now has a full training/evaluation/export pipeline with quality gates and safety checks.
