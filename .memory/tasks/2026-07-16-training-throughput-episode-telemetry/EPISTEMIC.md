# Current Epistemic Model

## Phenomenon

The custom ManoRL loop previously retained cumulative elapsed time and only an update mean for completed episodes. That hid per-update throughput and discarded episode-level return evidence needed to compare vector-environment scales.

## Mechanism

`tools/train_manorl_cube1.py` observes delayed-reset completion masks and exact `snapshot.episode_return` values at every rollout vector step. It emits one `manorl.completed_episode_returns.v1` record per completed vector step, with parallel `env_ids` and `returns`. The record is written and flushed to `<output>.episodes.jsonl.partial` before it is printed to stdout with `flush=True`; therefore a BrokenPipe can stop training only after the corresponding durable record exists. The partial is atomically renamed to `<output>.episodes.jsonl` after both training and recorder cleanup finish normally. A failure leaves the partial intact, and collision checks refuse to overwrite either name. This bounds writes by vector steps rather than episode count.

The same update boundary now obtains one `update_finished` sample, deriving both update and cumulative elapsed denominators from it. Raw same-update values feed one W&B histogram and are discarded after the update. The maximum target batch is 48 * 4096 = 196,608 values, so it does not accumulate across updates. Persisted update metrics retain aggregates only. `ManoPPOConfig.skrl_config` remains the source of truth for minibatch divisibility, invoked both by CLI validation and runtime construction. The selected minibatch is included by existing budget and PPO-config serialization in metrics/checkpoint metadata.

## Evidence And Claim

Configured-interpreter focused tests pass (37): deterministic clock assertions prove the shared timing boundary; a four-environment fake verifies batched returns; a BrokenPipe test proves write/flush precede stdout; partial success/failure and final/partial collision tests prove publication behavior; W&B/model/checkpoint tests remain green. See OPS.md entry `2026-07-16T12:16:07+08:00`.

The implementation preserves PPO, reward, checkpoint, and evaluation behavior while adding the requested telemetry. No GPU or training process was launched.
