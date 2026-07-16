# Operations Evidence

- 2026-07-16T16:17+08:00: coordinator classified the request as T2 because a
  credible speedup likely crosses environment, framework-boundary, tests, and remote
  GPU benchmarking. Created `feat/end-to-end-training-throughput` from dev commit
  `8328547` using `scripts/start_pi_task.sh`; no feature edits were made in primary
  `dev`.
- At task creation, Server2 physical GPU 0 was idle (38 MiB used, 24,046 MiB free,
  0% utilization). Physical GPU 3 hosted the protected 20,000-update run as PID
  `1570368`, using about 3,875 MiB at the sample.
- Existing reference pair on Server2 GPU 0 used 2,048 training environments, 128
  evaluation environments, rollout 48, minibatch 2,048, 12 updates, and W&B off.
  Legacy: 115.4 s / 10,221.1 transitions/s. Combined device-control/no-snapshot
  mode: 100.1 s / 11,783.9 transitions/s. Artifacts use prefixes ending
  `20260716_155449` and must be preserved.
- The existing remote benchmark launcher ran in zsh but used Bash
  `${PIPESTATUS[0]}`, producing an empty exit status and `[: unknown condition: -ne`.
  Future runs must use an explicit Bash shell or zsh's `$pipestatus[1]` and validate
  metrics/checkpoint existence.
- Feature benchmarking must use an isolated remote source mirror, not the Server2
  primary path managed by continuous Unison. Record that mirror path before the first
  run.

- 2026-07-16T16:43:19+08:00: Added opt-in `profile_phases` timing to
  `sim/manorl/environment.py`, separating action conversion/processing, controller
  target work, MJX physics, delayed reset, state/contact extraction, termination,
  reward, observation, and transition recording. Added PPO-loop timings for policy
  action, transition recording, and PPO update in `tools/train_manorl_cube1.py`.
  JAX/Torch synchronization occurs only when profiling is enabled. Extended
  `tools/benchmark_manorl_device_controls.py` to independently cross controls and
  transition diagnostics, randomize mode order, repeat runs, and emit phase JSON.

-  2026-07-16T16:43:19+08:00: Validation commands and observations: `python -m
  py_compile sim/manorl/environment.py tools/benchmark_manorl_device_controls.py
  tools/train_manorl_cube1.py` passed; `git diff --check` passed;
  `/home/jay/anaconda3/envs/manorl_mujoco/bin/python -m pytest -q
  tests/manorl/test_device_resident_controls.py tests/manorl/test_train_budget.py`
  passed 29 tests; `tests/manorl/test_environment.py` passed 21 tests; focused
  observation/reward tests passed 9 tests with one pre-existing fixture-dependent
  test deselected because
  `outputs/manorl/cube1_01_009_isaacgym_20260714_recovered.npz` is absent in this
  worktree. CPU four-way smoke command with `--num-envs 1 --warmup-steps 1
  --steps 2 --repeats 1 --profile-phases` completed all four modes and emitted all
  required phase keys. The missing fixture is a validation gap, not a profiling
  failure.

- 2026-07-16T17:34:09+08:00: Repaired Phase 1 attribution locally without a
  commit or remote experiment. `tools/benchmark_manorl_device_controls.py` now
  requires provenance plus a v3 schema, launches every mode/repeat in a fresh
  child process, preserves child stdout/stderr/JSON artifacts, rejects stale
  schema/legacy invocation, and publishes the aggregate only after all children
  succeed. The per-sample child no longer calls `jax.clear_caches`; process exit
  owns JAX/Warp lifecycle. CPU subprocess validation recorded four distinct child
  PIDs and passed fail-closed tests for nonzero child exit, missing JSON, and stale
  provenance/schema.
- 2026-07-16T17:34:09+08:00: Added opt-in attribution for action NumPy
  materialization, full runtime `env.step`, skrl action tensor-to-NumPy and
  NumPy response-to-tensor conversions, split state/contact-buffer/Python decode,
  reset indexed writes versus full-batch forward, trainer finite checks, reset
  `.item()`, host telemetry, all `post_interaction` calls, and only actual PPO
  rollout-boundary calls. Contact profiling records per-step `nacon`, mean/p50/p95/max,
  capacity, raw buffer metadata, and host materialization metadata without naming
  PCIe traffic. `TrainingBudget`/CLI now independently select diagnostics, with
  `None` preserving the prior inverse-of-controls default.
- 2026-07-16T17:34:09+08:00: Local validation passed:
  `JAX_PLATFORMS=cpu /home/jay/anaconda3/envs/manorl_mujoco/bin/python -m pytest
  -q tests/manorl/test_benchmark_device_controls.py
  tests/manorl/test_device_resident_controls.py tests/manorl/test_train_budget.py
  tests/manorl/test_environment.py tests/manorl/test_model_runtime.py
  tests/manorl/test_train_wandb.py` -> 84 passed; `python -m py_compile
  sim/manorl/environment.py sim/manorl/skrl_runtime.py
  tools/benchmark_manorl_device_controls.py tools/train_manorl_cube1.py` and
  `git diff --check` passed.
- 2026-07-16T17:34:09+08:00: Attempted isolated mirror refresh at
  `jay@192.168.9.220:/home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/feat-end-to-end-training-throughput-b7f1171`.
  The explicit rsync transfers completed for `sim/manorl`, owned tools, tests,
  and a provenance artifact, but the final remote hash verification stopped at
  `zsh:1: command not found: python`. Under the current no-remote boundary no
  retry or benchmark was issued; the mirror must be rehashed/refreshed after this
  expanded attribution diff before any experiment.
- 2026-07-16T17:37:37+08:00: Reviewer reported that protected Server2 GPU3
  PID `1570368` was stopped after update 639. Its checkpoint and `last.pt` remain,
  but Ctrl-C ended with `IOT instruction core dumped`; classify this as abnormal,
  non-graceful termination and exclude it from all benchmark statistics. GPU3 is
  now released. Future benchmark execution is restricted to physical GPU0 and
  GPU3; GPU1/GPU2 are forbidden.
- 2026-07-16T17:37:37+08:00: Balanced evidence design is two GPU blocks (GPU0,
  GPU3), with every mode/repeat sampled independently on both blocks. Each cell
  gets its own Python process, output prefix, and utilization/memory sampler;
  mode order is randomized/interleaved independently per GPU and repeat. A fresh
  occupancy check precedes every child, and any GPU0/GPU3 contention or any GPU1/
  GPU2 use aborts the run. No remote process was launched from this worker after
  the local gate; reviewer approval remains the execution boundary.
