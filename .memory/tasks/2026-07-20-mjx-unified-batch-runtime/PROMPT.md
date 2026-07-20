# MJX Unified Batch Runtime

## Objective

Redesign the ManoRL MJX-Warp runtime so mixed-object training can execute with
bounded GPU memory and a measurable batched physics path, instead of invoking
one Python/JAX/Warp route per object type. Preserve the existing cube1 single-
object contract while establishing an evidence-backed path for all-object,
all-action shared-policy training.

## Context

- Branch: `feat/mjx-unified-batch-runtime`
- Worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-mjx-unified-batch-runtime`
- Base commit: `4bb66cf`
- Current heterogeneous implementation runs 13 object routes serially and
  failed a server2 GPU3 full run at update 5 with a Warp CUDA allocation error.
- The Isaac Gym reference uses one PhysX simulation with heterogeneous actors and
  one device tensor pipeline; use it as a behavioral and architectural reference
  only. Do not modify the sibling Gym repository.

## Required sequence

1. Trace the current route construction, MJX model/static-shape constraints,
   Warp allocation lifetime, and the Gym reference's single-simulation path.
2. Prototype the smallest technically valid unified representation. It must
   explicitly document how object geometry, collision/contact masks, mass,
   gravity, reset placement, and per-environment trajectory state are encoded.
3. Prove semantics on a bounded two-object fixture before attempting all 13
   objects. Compare observations, rewards, termination, reset behavior, and
   contact diagnostics against the existing route implementation.
4. Add focused tests for mixed-object construction, partial/delayed reset,
   randomized actions, finite outputs, and device allocation stability.
5. Run a short server2 GPU3 benchmark only after local tests pass. Record source
   commit, command, environment count, throughput, peak memory, and any OOM.
6. If a correct unified model cannot be implemented without a speculative
   collision approximation, stop at the prototype/design and report the blocker
   rather than weakening physics semantics.

## Acceptance

- Existing cube1 single-object focused tests remain green.
- A bounded mixed-object test demonstrates one stable batched execution path or
  a concrete, justified fixed-shape prototype with no per-step route allocation.
- Full all-object/all-action training is a post-integration action only: do not
  launch it until the unified runtime passes focused semantic and allocation
  validation and the feature is reviewed and merged into `dev`.
- No production claim of Gym-like throughput is made without matched GPU
  measurements.
- The implementation must not silently fall back to CPU or alter the reward,
  observation, termination, checkpoint, or W&B contracts.
- Any object-geometry approximation is opt-in, explicitly versioned, and cannot
  be used by the production all-object command by accident.

## Scope

The worker may change only `sim/manorl/**`, focused `tools/**`, focused
`tests/manorl/**`, relevant runtime documentation, and this task-memory
directory. Do not edit generated outputs, asset submodules, `PI_HANDOFF.md`, or
the primary `dev` worktree. Do not launch a full training job from the worker.
