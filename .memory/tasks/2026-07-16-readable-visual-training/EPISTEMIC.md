# Current Epistemic Model

## Phenomenon

Training telemetry is lossless but unusable as a terminal surface because multi-thousand-element episode arrays and the complete final result dominate stdout. The active Server2 process cannot adopt a code change without restart. Local PPO has no renderer hook; the existing viewer owns simulation stepping.

## Supported mechanism

Separate durable/machine telemetry from human presentation. Keep exact JSONL and JSON event formats, but route human console output through summaries and provide a parser for existing logs. Add a render-only observer callback after PPO's existing environment step. The callback mirrors states to native MuJoCo data and renders; PPO remains the sole owner of actions, stepping, transition recording, and updates.

## Current claim

`TrainingViewer.render` reads only `host_data_batch()` and MuJoCo/GLFW rendering APIs. `_train` calls its observer after `runtime.env.step`, then checks `close_requested` only after all configured rollout steps, so a close cannot produce a partial PPO update. The normal post-rollout path still writes final checkpoint, evaluation, metrics, JSONL publication, Rerun finalization, and W&B artifacts. An eight-world rollout has 384 samples, so minibatch 384 is the direct valid one-minibatch local configuration.

Human console output summarizes completed episodes after their exact JSONL line is flushed; explicit JSON mode preserves raw episode records, update/complete event schemas, and final JSON result. The watcher parses a line at a time and summarizes large return arrays without echoing them.

## Decisive validation

Focused tests prove rollout-boundary close behavior, renderer source has no step ownership, compact formatter behavior and JSON compatibility, giant-record watcher behavior, and CLI viewer validation. GUI behavior remains conditional on an available graphical session.
