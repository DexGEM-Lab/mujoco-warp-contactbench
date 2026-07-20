# Operations Plan And Evidence

## Execution contract

- Repository branch: `feat/trajectory-quality-experiment-design`.
- Dataset: `cube1`, gesture `01`, `includeSuffixFiles=false`.
- W&B: enabled for every training arm, project `one_policy`, with a distinct
  run name and tags for experiment ID, quality tier, seed, env count, and
  commit. Offline mode is not an acceptable substitute unless the server is
  unavailable; record that exception explicitly.
- Server2: use one isolated process/session per idle GPU. Set `DISPLAY=:1`
  when the runtime requires it, keep `JAX_PLATFORMS=cuda`, disable JAX memory
  preallocation, and never claim a GPU is idle without checking the process and
  memory tables immediately beforehand.
- Outputs live below an experiment-specific directory under
  `outputs/manorl/`; never reuse a prior run directory. Preserve logs,
  metrics JSON, episode JSONL, checkpoints, sidecars, and W&B run IDs.

## Four-GPU launch batch

Launch at most four arms concurrently, one per confirmed idle GPU. The exact
GPU IDs are discovered at launch time; do not assume a fixed numbering. A
representative command shape is:

```bash
export DISPLAY=:1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CUDA_VISIBLE_DEVICES=<gpu> python tools/train_manorl_cube1.py \
  --object-type cube1 --gesture 01 --num-envs 2048 --updates 300 \
  --wandb true --wandb-project one_policy \
  --wandb-group trajectory-quality-20260719 \
  --wandb-tags experiment=<E-ID> --wandb-tags seed=<SEED> \
  --wandb-tags quality=<TIER> --wandb-tags commit=<SHA> \
  --output outputs/manorl/trajectory-quality-20260719/<RUN-ID>
```

The command is a template: use the repository's current CLI spelling and
explicitly pass the experiment's trajectory selection once that selector is
available. Keep a `tmux` session per run and write a `.train.log` beside the
output. Before launch, record `nvidia-smi`, host RAM, and the resolved command;
after launch, verify the process owns only its assigned GPU.

The first four-card batch should be the highest-information screening arms,
not four copies of a long production run. Recommended initial allocation is
E1 Core anchor `003`, E1 Core-easy anchor, E2-core, and E2-all, each with seed
42 (or use seeds 42/43/44 as soon as the code can schedule the complete paired
matrix). Repeat the same arm with the remaining seeds rather than comparing
different configurations on different seeds.

## Preflight checklist

1. Freeze the quality manifest and confirm no trajectory is classified from RL
   return.
2. Confirm the selected feature commit and clean source mirror on Server2.
3. Check idle GPU IDs, GPU memory, host physical memory, and output filesystem
   capacity; do not scan or delete unrelated user directories.
4. Dry-run argument parsing and instantiate one evaluation environment before
   allocating all 2,048 environments.
5. Verify W&B authentication, project/group/name, and that each run has a
   unique output prefix.
6. Save the exact resolved runtime configuration in the run directory.

## Monitoring

At minimum sample each run at startup, update 1, update 10, every 50 updates,
and completion. Capture:

- process state, GPU memory/utilization, host memory, and filesystem use;
- update duration/FPS, LR, exact KL, entropy, value loss, and reset counts;
- positive-transition and zero-positive-minibatch telemetry;
- per-identity episode returns and completion/deviation metrics;
- W&B run URL/ID and checkpoint sidecar hash.

Do not terminate a run solely because completed-episode mean is low during the
first contact window. Stop for OOM, NaN/Inf, a dead process, a wrong dataset or
GPU, or a clearly invalid source manifest. If a run is intentionally stopped,
mark it `stopped` and preserve its partial artifacts; never call it a failed
convergence result.

## Post-run evidence bundle

For each arm, collect:

```text
<run>.train.log
<run>.json
<run>.episodes.jsonl (and .partial if interrupted)
checkpoint-*.pt plus sidecars
W&B run ID/URL
resolved command and git SHA
```

Run deterministic evaluation for every declared Core and Hard-valid identity
using fixed point-template seeds. Summarize a row per training seed and a row
per evaluation identity. Include first-positive update, positive fraction,
zero-positive-minibatch ratio, completion calls, max deviation, and reset
reason. A high episode return without these fields is diagnostic only.

## Stop/cleanup policy

The coordinator owns remote process cleanup. After all four arms finish or are
stopped, verify no child process remains and GPUs are released. Keep artifacts
for promoted and diagnostic arms; remove only explicitly approved duplicate
copies. Do not merge experiment-specific sampling or curriculum changes into
`dev` from this documentation task.
