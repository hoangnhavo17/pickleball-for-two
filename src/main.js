/**
 * Browser entry: loads policy + optional ML helpers, then boots the game.
 * Python is not required to play — see /ml for offline training only.
 */
import "./ai/cpuPolicy.js";
import "./data/recordGameplay.js";
import "./ai/mlAI.js";
import "./game/game.js";
