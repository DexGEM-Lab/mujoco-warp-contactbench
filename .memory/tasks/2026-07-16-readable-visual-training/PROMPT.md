# Readable Console and Local Visual Training

## User goal

1. Replace the current training terminal's unreadable giant numeric JSON arrays with a human-readable progress surface while retaining exact episode returns and machine-readable telemetry.
2. Provide an executable local training script equivalent to `headless=false`, configured for eight batched environments and showing the actual PPO rollout worlds while training.

## Mechanism

The current trainer writes the exact batched episode JSON record to the durable JSONL file and then duplicates the full `env_ids` and `returns` arrays to stdout. It also prints the entire final metrics object, including all updates. The existing viewer advances the environment itself, so it cannot be used during PPO training without double-stepping. Training visualization must mirror already-produced MJX states after each rollout step without owning physics or policy actions.

## Required behavior

- Add a human console mode that prints concise one-line update, completed-episode batch, checkpoint, and completion summaries. Exact episode values remain losslessly flushed to `.episodes.jsonl.partial` and atomically published on success.
- Preserve an explicit JSON console mode for machine consumers and existing event schemas. Do not silently remove structured logging.
- Add a standard-library log watcher that can summarize an existing/current `.train.log` and optionally follow it, so the active Server2 run need not restart.
- Add `--headless {true,false}` training CLI behavior, default true. `false` opens a tiled MuJoCo/GLFW renderer over existing training worlds; renderer never calls environment.step or policy methods.
- Add validated viewer controls for tile count/render stride as needed. For user close, finish the current rollout/update and exit through normal checkpoint/evaluation/artifact publication rather than raising an interrupted partial run.
- Add an executable local wrapper for cube1/gesture01 with 8 envs, 8 evaluation envs, rollout-compatible minibatch 384, W&B off, unique output prefix, and headless false. It must use the configured local conda interpreter but allow environment/CLI overrides where practical.
- Update README/protocol with exact local command, graphical-session requirement, viewer controls, performance implications, output paths, and current Server2 watcher command.
- Add focused tests for console formatting/no giant arrays, JSON compatibility, watcher parsing, renderer no-step ownership, close behavior, CLI validation/defaults, and wrapper syntax/config.

## Constraints

- Preserve PPO/reward/checkpoint/W&B/Rerun semantics and active Server2 process; no remote restart or signals.
- Do not make CPU training a silent fallback. Local visual training still requires CUDA-capable Torch/MJX-Warp and an X11/Wayland desktop session.
- Use existing MuJoCo tiled visual conventions and camera controls where possible.
- Work only in `feat/readable-visual-training`; do not touch `PI_HANDOFF.md` or generated outputs.

## Acceptance

- Focused trainer/viewer/checkpoint/W&B tests pass.
- A bounded local GUI smoke test is run only if a graphical session is available; otherwise parser/renderer unit tests and help/shell syntax checks must pass and the missing graphical validation is reported.
- Independent review finds no double-step, incomplete-update, artifact, or structured-log compatibility regression.
