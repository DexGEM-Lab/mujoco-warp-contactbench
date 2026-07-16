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
- 2026-07-16T17:47:10+08:00: Rebuilt the isolated mirror only at
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/feat-end-to-end-training-throughput-b7f1171`.
  Seed source was the remote Unison-managed primary
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco`, copied with `outputs/` and `.git/`
  excluded from deletion; no sync-back was performed. Overlay files were exactly:
  `sim/manorl/environment.py`, `sim/manorl/skrl_runtime.py`,
  `tools/benchmark_manorl_device_controls.py`, `tools/train_manorl_cube1.py`,
  `tests/manorl/test_benchmark_device_controls.py`,
  `tests/manorl/test_device_resident_controls.py`, and
  `tests/manorl/test_train_budget.py`.
- 2026-07-16T17:47:10+08:00: Uploaded provenance
  `/tmp/bench_phase_ablation_49b2baf.source_provenance.json` with source commit
  `49b2baf2f4de9865203584d32c36c0a5b0f6d9aa`, feature diff SHA256
  `6d01bb34a32836358d2d7fe33986599be9389cab1bd6de85d730997eeb47dce1`, and
  seven source/mirror file hashes. Remote verification used explicit
  `/home/jay/miniconda3/envs/manorl_mujoco/bin/python`; all seven hashes matched.
  The pre/post mirror `outputs/` manifests matched exactly: 13 files, 27,265 bytes.
  No benchmark code ran during mirror rebuild or verification.
- 2026-07-16T17:48:12+08:00: Two initial remote verification command attempts
  failed in the reporting heredoc before completing output: one referenced an
  undefined `source_commit` name and the next had a quote typo in a diagnostic
  branch. Neither changed mirror files or outputs. The simplified rerun completed
  the same manifest and seven-file hash checks successfully.
- 2026-07-16T17:59:25+08:00: Frozen remote evidence state. Formal runner log
  `/tmp/manorl-phase1-formal.log` stopped after eight GPU0 stepping cells and the
  one GPU3 first-block stepping cell; no PPO cell ran. Stop reason was allowed-GPU
  contention from PID `2072971`, user `zjx`, command
  `/mnt/user-home/zjx/miniconda3/envs/seq2seq_retarget/bin/python
  tools/diffusion/sample_diffusion.py ... --device cuda` on GPU3. The runner
  exited failed-closed; no process was killed. GPU3 later became idle, but the
  runner was not resumed. Preserve root
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/feat-end-to-end-training-throughput-b7f1171/outputs/manorl/bench_phase1_gpu_blocks_20260716_1750`,
  its manifest, and all raw JSON/log/smi/provenance artifacts.
- Missing evidence cells are GPU3's remaining 11 stepping cells, GPU0 repeat 2's
  four stepping cells, and all PPO cells. The target PPO block is 24 cells total
  (GPU0/GPU3 x repeats 0/1/2 x four modes); zero PPO cells are valid. This partial
  run is not benchmark completion and no statistics are reported from it.
- 2026-07-16T18:04:39+08:00: Later GPU0-only continuation was explicitly
  stopped before execution. A local `/tmp/run_gpu0_completion_49b2baf.sh` draft
  was written but never transferred, launched, or used; no child process or
  sampler started, and no remote artifact changed. The blocker remains the
  reviewer freeze after GPU3 contention; preserve the partial root and wait for
  explicit direction before any resume.
- 2026-07-16T18:17:22+08:00: Implemented only the local vectorized contact
  decoder change in `sim/manorl/environment.py`, retaining the original loop as
  `_decode_contact_forces_reference` for equivalence tests. On a 12-contact,
  4-world randomized valid fixture, numerical max absolute difference was 0;
  40-call median timing was 93.325 us for the reference and 60.956 us for the
  vectorized decoder (1.531x median speedup). Focused decoder/device tests passed
  6/6; environment plus device-control tests passed 27/27. No remote command,
  source optimization outside the decoder, PPO/profile change, or commit was made.
- 2026-07-16T18:21:52+08:00: Independent realistic decoder microbenchmark on
  batch=2048, count=13756, ngeom=21, nefc=512 reported reference median
  95.462 ms and vectorized median 1.582 ms, a 60.349x median speedup. Maximum
  absolute difference was 3.5527e-15; allclose and output-count assertions passed.
  Independent runtime equivalence used two identical CPU
  `MujocoManoEnvironment` instances for two identical steps, running the
  vectorized decoder on one and temporarily monkeypatching the module decoder
  to `_decode_contact_forces_reference` only while stepping the other. Observation,
  reward, reset, timeout, physical positions, and contact/force fields matched at
  `atol=1e-12`.
- 2026-07-16T18:22:30+08:00: Confirmed decoder-only evidence summary: the realistic
  2,048-world/13,756-contact microbenchmark measured 95.462 ms reference versus
  1.582 ms vectorized, 60.349x, with maximum absolute difference 3.55e-15. The
  two-step CPU runtime equivalence matched at `atol=1e-12`. Focused validation
  passed 7 decoder/device tests and 27 environment/device tests.
- 2026-07-16T18:26:48+08:00: Attempted to build the new isolated mirror
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/feat-end-to-end-training-throughput-00918e5`
  from remote primary `/home/jay/dexrobot/FromSSH/manoRL_mujoco`, but rsync
  failed with `No space left on device` while copying
  `assets/all_assets/Assets/sim/DexYCB/010_potted_meat_can/textured.obj`.
  Remote `/mnt/user-home` reported 2.0T total, 100% used, 18M available; the
  partial mirror reached 1.7G and has no outputs. No feature overlay, benchmark
  child, sampler, or GPU process started. No prior artifact was deleted.
- 2026-07-16T18:37:55+08:00: Recreated the new mirror at
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/feat-end-to-end-training-throughput-00918e5`
  with `cp -al` from old mirror
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/feat-end-to-end-training-throughput-b7f1171`.
  Removed only the new clone's hardlinked `outputs/` directory, overlaid committed
  `sim/manorl/environment.py` and `tests/manorl/test_device_resident_controls.py`
  without `--inplace`, and uploaded commit `00918e59c2045374a6f50e727a5c950c0fbe6966`
  provenance. Old mirror feature hashes and its 74-file outputs manifest matched
  before and after exactly.
- 2026-07-16T18:37:55+08:00: Existing v3 child harness validation completed all
  16 stepping children (GPU0/GPU3 x repeats 0/1 x four modes), each with unique
  JSON, stdout/stderr, sampler, and provenance artifacts; all child exits were 0,
  occupancy checks passed, and the completion marker was written. New output root:
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/feat-end-to-end-training-throughput-00918e5/outputs/manorl/bench_phase1_vectorized_00918e5_20260716_1826`.
  The comparison artifact is `comparison_old_prevectorized.json`; 9 of 16 new cells
  had matching old pre-vectorized artifacts, with median throughput ratio 2.7287x.
  Seven new cells had no matching old artifact and were not used in that comparison;
  prior incomplete/failure artifacts remain excluded.
- 2026-07-16T18:37:55+08:00: New validation summary reports static contact
  capacity 63,552 and example profiled nacon sample count 48, mean 13,756.625,
  p50 12,697, p95 18,692.5, max 18,836. Raw `efc__force` is float32 shape
  `(2048,512)` nbytes 4,194,304; host materialization is float64 shape
  `(2048,512)` nbytes 8,388,608. These are raw-buffer/materialization bytes only,
  not PCIe-transfer measurements. Summary artifact: `validation_summary.json`.
