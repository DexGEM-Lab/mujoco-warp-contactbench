# Current Epistemic Model

## Phenomenon

The custom ManoRL loop previously retained cumulative elapsed time and only an update mean for completed episodes. That hid per-update throughput and discarded episode-level return evidence needed to compare vector-environment scales.

## Mechanism

`tools/train_manorl_cube1.py` observes delayed-reset completion masks and exact `snapshot.episode_return` values at every rollout vector step. It now emits one `manorl.completed_episode_returns.v1` record per completed vector step, with parallel `env_ids` and `returns`, to stdout and an output-derived `.episodes.jsonl`. This bounds synchronous writes by vector steps rather than episode count. The raw same-update values feed one W&B histogram; persisted update metrics retain aggregates only.

The same update boundary measures elapsed time before and after each PPO rollout, producing instantaneous FPS from one rollout's transitions and cumulative FPS from all completed transitions. Final throughput uses the full training elapsed time. `ManoPPOConfig.skrl_config` remains the source of truth for minibatch divisibility, invoked both by CLI validation and runtime construction. The selected minibatch is included by existing budget and PPO-config serialization in metrics/checkpoint metadata.

## Evidence And Claim

Focused tests passed under the configured ManoRL interpreter: deterministic clock assertions cover exact update/cumulative FPS; a four-environment fake verifies one batched completion record preserves two returns; W&B tests verify one histogram and JSONL artifact inclusion; CLI tests verify the default 1024, accepted 4096-with-4096-env selection, and rejected non-divisible selection. See OPS.md entry `2026-07-16T12:08:04+08:00`.

The implementation preserves PPO, reward, checkpoint, and evaluation behavior while adding the requested telemetry. No GPU or training process was launched.
