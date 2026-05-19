# Pickleball for Two

> Browser singles pickleball vs a CPU. Tennis for Two–style canvas game — **Normal mode**, no ML weights.

Singles vs a hand-tuned CPU opponent. Court uses real-ish rules (two-bounce serve, kitchen NVZ, rally scoring to 11). **Pro (ML)** is optional and not required for this release.

## Play online

- **[itch.io](https://oneyayvo.itch.io/pickleball-for-two)** — game page (embed + fullscreen)
- **[GitHub Pages](https://hoangnhavo17.github.io/pickleball-for-two)** — live demo from this repo

## Controls

| Input | Action |
|-------|--------|
| **Arrow keys** | Move paddle (up/down is faster for shot-height changes) |
| **1–4** | Serve angle preset |
| **Space** | Time the power bar, then serve (or ready for CPU serve) |
| **Q** | Toggle soft shots: kitchen → dink, box → drop |
| Paddle height (no Q) | High → smash, mid → drive, low → lob |

Pick **Normal** under CPU mode. **Pro (ML)** is experimental and needs files under `public/models/` (not required for Normal play).

## Rules (in-game)

Open **Pickleball rules (singles)** under the court for full detail. Short version:

- **Scoring:** server’s score first; only the server scores; side-out if the receiver wins; game to **11**, win by **2**
- **Two-bounce:** serve must bounce on the receiver’s side; return must bounce on the server’s side before volleys
- **Kitchen (NVZ):** no volley in the kitchen before the ball bounces on your side
- **Serve fault:** first bounce in the opponent’s kitchen is out
- **Second bounce** on your side after the last hit → point lost

Court zones use a **30px paddle unit** for layout (kitchen, box, net). On-screen paddles are drawn larger for gameplay.

## Project layout

| Path | Purpose |
|------|---------|
| `src/sim/` | Physics, rules, rally logic (shared core) |
| `src/game/` | Canvas, input, HUD |
| `src/index.html` | Game page |
| `index.html` | Redirect to the game |
| `ml/` | Offline training pipeline (optional; not needed to play) |

## Development

Sim regression (rules / scoring smoke test):

```bash
npm run ml:regression
```

More on the ML side and Pro CPU: see `PROJECT_EXPLAINED.md` and scripts under `ml/`.

## License

MIT — see [LICENSE](LICENSE).
