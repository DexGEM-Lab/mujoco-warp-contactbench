# Training Throughput And Episode Telemetry

## Goal

Make ManoRL long-run training expose performance and episode-return evidence needed to compare vector-environment scales.

## Required behavior

- For every PPO update, compute instantaneous environment transitions per second over that update and cumulative environment transitions per second over training.
- Print the update telemetry to stdout and include it in metrics JSON and W&B.
- Report final training throughput after training completes.
- Preserve every completed episode return. At each vector step with completions, print one structured JSON record containing the vector step/update context and parallel arrays of completed env IDs and exact episode returns; persist the same records under an output-derived `.episodes.jsonl` artifact.
- Keep W&B update-level `completed_episode_count` and `episode_return_mean`; add a useful per-update return distribution without issuing one W&B log call per episode.
- Avoid per-episode synchronous writes that become a bottleneck at 4096 environments.
- Include the episode JSONL artifact in W&B artifact upload when W&B is enabled.
- Expose PPO minibatch size as a validated training CLI/runtime option, preserving the existing 1024 default but allowing the Server2 4096-env run to use the IsaacGym command's minibatch size of 4096. Serialize the selected value in metrics/checkpoint runtime configuration.
- Update the training protocol documentation and add focused tests.

## Constraints

- Work only in `feat/training-throughput-episode-telemetry` and its linked worktree.
- Preserve current reward/checkpoint contracts and training behavior.
- Do not modify unrelated files or generated outputs.
- No training launch or GPU process changes in this worker task.

## Success criteria

- Focused tests prove exact FPS arithmetic under a controlled clock and complete episode-return serialization for multiple envs completing on the same vector step.
- Existing focused training/W&B tests remain green.
- The worker commits a coherent change and reports commit, tests, and residual risks to the coordinator.
