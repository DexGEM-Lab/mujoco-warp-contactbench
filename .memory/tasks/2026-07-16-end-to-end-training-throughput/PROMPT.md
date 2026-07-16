# End-to-End ManoRL Training Throughput

## Objective
Materially improve measured end-to-end PPO training throughput for the production
`cube1/01`, 2,048-environment workload while preserving its action, physics,
observation, reward, termination/reset, PPO, checkpoint, W&B, and CPU-mode
contracts. The optimization must be attributed by measurement rather than inferred
from aggregate GPU utilization.

## Workbench

- Branch: `feat/end-to-end-training-throughput`
- Local worker worktree:
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-end-to-end-training-throughput`
- Pi name: `manorl-feat-end-to-end-training-throughput`
- Remote benchmark host: `jay@192.168.9.220`
- Remote benchmark devices: physical GPUs 0 and 3 only, after a fresh occupancy check for every child
- Remote benchmark source must be an isolated mirror outside the continuously
  synchronized primary project directory. Seed it from the remote primary tree as
  needed, then overlay only this feature worktree's owned source/test/tool files.
  Never benchmark uncommitted feature code by overwriting the remote primary tree.
- The prior protected GPU3 run (`manorl-s2-cube1-01-2048-20k-20260716_151629`,
  PID `1570368`) ended at update 639 with non-graceful `IOT instruction core
  dumped` termination. Its checkpoint and `last.pt` remain historical artifacts;
  exclude it from benchmark statistics. GPU3 is now released. GPUs 1 and 2 are
  forbidden because they are used by other users.

## Required sequence

1. Read this task memory, `.memory/project/pi-orchestration.md`, and the repository
   skill before substantive work.
2. Trace one rollout control step and one PPO update across Torch/skrl, Gymnasium,
   NumPy, JAX, and MJX-Warp. Add or use low-overhead phase timing that separates at
   least policy action, action conversion/processing, controller target work, MJX
   physics, state/contact extraction, observation/reward/termination, transition
   recording, and PPO update. Synchronize devices at timing boundaries only in an
   explicit profiling mode; do not slow the production path.
3. Establish balanced blocked evidence on remote GPUs 0 and 3. Run every
   four-way mode (legacy/device controls crossed with transition diagnostics on/off)
   independently on both GPUs for each repeat. Each sample must be its own Python
   process, output prefix, and utilization/memory sampler. Randomize/interleave
   order independently per GPU and repeat. Use GPU index as a statistical block;
   never compare a GPU0-only mode against a GPU3-only mode. Perform a fresh
   occupancy check before each child and abort on allowed-GPU contention or any
   GPU1/GPU2 use.
4. Implement the smallest semantics-preserving optimization at the measured
   dominant boundary. Expected candidates include device-side contact decoding and
   physical feature extraction, device-side observation/reward/termination, and a
   Torch/JAX zero-copy or DLPack environment boundary. Do not assume those candidates
   are dominant without the profile.
5. Add focused equivalence tests, including multi-environment partial/delayed reset,
   randomized actions, terminal behavior, and GPU equivalence where the configured
   runtime permits it. Keep an explicit debug/diagnostic path for exact transition
   records.
6. Run matched complete PPO benchmarks at 2,048 training environments, 128
   evaluation environments, rollout 48, minibatch 2,048, W&B disabled, and enough
   post-warmup updates for a credible throughput distribution. Report median and
   spread, not only one sequential pair. Capture per-mode, per-GPU utilization and
   memory. Exclude the historical abnormal GPU3 training termination and all prior
   stale-mirror/OOM failures from statistics while preserving their raw artifacts.
7. Commit a coherent feature result, update OPS/EPISTEMIC with commands and evidence,
   and report the commit plus residual risks.

## Acceptance criteria

- The committed production default either becomes measurably faster or exposes an
  explicit optimized mode suitable for headless production training.
- The speed claim is end-to-end PPO throughput, not merely physics stepping, and is
  supported by repeated matched measurements on the same remote GPU.
- Exact action, observation, reward, termination/delayed-reset, and completed-episode
  return semantics pass focused equivalence checks at tolerances justified by device
  precision.
- No hidden per-step Torch CUDA -> NumPy -> JAX CUDA -> NumPy -> Torch CUDA round trip
  remains in any boundary claimed to be device resident. Any boundary that remains
  host-resident is named and timed.
- CPU mode remains functional under the same MJX-Warp backend. Do not reintroduce a
  MuJoCo CPU backend.
- Existing public CLI behavior remains compatible unless a new opt-in profiling or
  optimized flag is documented and tested.
- The historical GPU3 run and all existing generated outputs remain untouched. New
  benchmark artifacts go under unique `outputs/manorl/bench_*` prefixes and are not
  deleted automatically.
- Record the remote mirror path and exact source commit/diff used by every benchmark,
  so results cannot accidentally be attributed to stale Unison state.

## Ownership

The worker may change only the feature worktree's `sim/manorl/**`,
`tools/train_manorl_cube1.py`, focused benchmark/profiling tools under `tools/`,
focused tests under `tests/manorl/`, and this task-memory directory. Dependency or
lockfile changes require a measured, documented necessity. Do not edit
`PI_HANDOFF.md`, asset submodules, generated Lance datasets, unrelated viewers, or
the primary `dev` worktree.

## Stop conditions

- If the first bounded implementation attempt fails from executor/model budget,
  preserve the partial worktree and report; do not automatically resume.
- If the correct fix requires a broad semantic rewrite whose equivalence cannot be
  established, stop at the profile plus a concrete design/blocker rather than
  landing a speculative fast path.
- If GPU0/GPU3 becomes occupied, GPU1/GPU2 is used, or any allowed-GPU contention
  appears, stop remote benchmarking and report before choosing another device.
