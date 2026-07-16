# Current Epistemic Model

## Phenomenon

Training telemetry is lossless but unusable as a terminal surface because multi-thousand-element episode arrays and the complete final result dominate stdout. The active Server2 process cannot adopt a code change without restart. Local PPO has no renderer hook; the existing viewer owns simulation stepping.

## Supported mechanism

Separate durable/machine telemetry from human presentation. Keep exact JSONL and JSON event formats, but route human console output through summaries and provide a parser for existing logs. Add a render-only observer callback after PPO's existing environment step. The callback mirrors states to native MuJoCo data and renders; PPO remains the sole owner of actions, stepping, transition recording, and updates.

## Current claim

`TrainingViewer.render` reads only `host_data_batch()` and MuJoCo/GLFW rendering APIs. It shares standalone tiled controls: left-drag rotate, right-drag horizontal pan, middle-drag vertical pan, wheel zoom, R reset, and Esc/window close. Its constructor owns GLFW transactionally after `glfw.init`: any subsequent setup error invokes idempotent `close`, which destroys an existing window and terminates GLFW once.

`_train` calls its observer after `runtime.env.step`, then checks `close_requested` only after all configured rollout steps, so a close cannot produce a partial PPO update. `run` owns recorder and viewer in nested cleanup so a viewer construction failure still closes the recorder. The normal post-rollout path still writes final checkpoint, evaluation, metrics, JSONL publication, Rerun finalization, and W&B artifacts. An eight-world rollout has 384 samples, so minibatch 384 is the direct valid one-minibatch local configuration.

Human console output summarizes completed episodes after their exact JSONL line is flushed; explicit JSON mode preserves raw episode/checkpoint/update/complete events and final JSON result. The watcher parses one line at a time and summarizes large legacy return records without echoing their arrays.

## Decisive validation

Fake GLFW/MuJoCo tests prove render behavior without stepping, all controls, idempotent close, and partial-constructor cleanup. Focused tests also prove recorder cleanup after viewer-construction failure, rollout-boundary close behavior, JSON shapes including final main output, giant legacy-record watcher aggregates, and CLI viewer validation. GUI behavior remains conditional on an available graphical session.
