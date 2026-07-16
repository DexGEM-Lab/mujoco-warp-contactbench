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
