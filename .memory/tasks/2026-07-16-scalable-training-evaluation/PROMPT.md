# Scalable ManoRL Training Evaluation

## Goal

Allow 4096-environment ManoRL PPO training to complete checkpoint-reload evaluation on one 24 GiB GPU without creating a second 4096-world physical runtime.

## Observed mechanism

Both GPU2 and an otherwise empty GPU3 completed one 4096x48 PPO update at about 10.5k environment transitions/s. Both then failed in the same Warp convex narrowphase allocation after the final checkpoint, when `run()` created a second 4096-world runtime for fresh checkpoint evaluation while the training runtime remained resident. The failed allocation was 275,930,880 bytes. This rules out single-runtime infeasibility and GPU2 contention.

## Required behavior

- Decouple evaluation vector count from training vector count through a validated CLI/budget field, defaulting to `min(training_num_envs, 128)` unless explicitly set.
- Zero, untrained-policy, and trained-policy evaluations must use the same evaluation count and same trajectory assignment prefix.
- The untrained evaluation must represent the exact policy/value/normalizer/optimizer initialization used to start training. Preserve this with a native checkpoint load boundary rather than relying on repeated RNG initialization.
- Final trained evaluation must continue to use a fresh runtime loaded from the final native checkpoint.
- Evaluation-only PPO minibatch configuration must be valid for its vector batch; record training and evaluation counts/configs distinctly and do not change the training minibatch or PPO update semantics.
- Avoid leaving temporary checkpoint artifacts after success; preserve explicit failures rather than silently falling back.
- Update docs and focused tests.

## Constraints

- Work only in `feat/scalable-training-evaluation`.
- Preserve v2 reward/checkpoint contracts, 128-env and smaller behavior, W&B/telemetry artifacts, and exact delayed-reset evaluation semantics.
- Do not launch GPU training from the worker; coordinator owns the Server2 4096x1 acceptance rerun.
- Do not modify unrelated files or generated outputs.

## Success criteria

- Tests prove no second training-sized environment is constructed when training envs exceed evaluation envs.
- Tests prove all three evaluation modes use identical evaluation assignments/count and that training loads the same native initialized state evaluated as untrained.
- Existing training, W&B, checkpoint, and model runtime focused tests pass.
- Commit a coherent change with updated task evidence.
