# Autonomous PPO telemetry and evaluation

`sim.manorl.autonomy_telemetry` reduces transition measurements at PPO update
boundaries. The environment remains the source of physical truth: object pose,
reference pose, collision witnesses, genuine hand-object force and the
object-relative contact-motion slip proxy are read from `info`. No reference
pose is substituted for an actual pose.

## Training

`tools/train_manorl_autonomy.py formaltrain` logs one row per completed PPO
update. W&B uses `transitions` (environment transition count) as its history
step axis and defines all metrics against that axis. The default W&B path is
entity `sunjay45711-dexerto`, project `mujoco-mano`; `WANDB_ENTITY` and
`WANDB_PROJECT` override it.

Existing `RlGamesPPO` tracking names are mapped to the mature logger aliases:
policy/value/entropy losses, exact and approximate KL, learning-rate schedule,
minibatch count and policy standard deviation. A metric is emitted only when
the canonical agent exposes it. The current agent does not expose PPO clip
fraction, so no fabricated clip-fraction value is logged.

Per-update physics telemetry includes reward terms, path RMSE, orientation
error, peak/target lift, genuine hand-object contact frame counts and sustained
runs, airborne-contact counts, slip proxy, failure-phase counts and transition
throughput. Accumulation accepts batches of `info` mappings so a future batched
adapter can reduce at update boundaries without materializing device tensors on
every step.

## Post-training publication

The bounded publication command consumes an existing trace and checkpoint:

```bash
python tools/publish_manorl_autonomy_evaluation.py \
  --trace outputs/.../learned-full-start.json \
  --checkpoint outputs/.../checkpoint-final.pt \
  --wandb-run-id n5zkg0eg
```

It requires the trace/checkpoint format to match, computes a readable JSON
summary with explicit denominators, and appends evaluation metrics/artifacts to
an existing **FINISHED** run. It does not retrain or write historical training
losses. The MP4 is a real MuJoCo offscreen render: each frame places actual and
reference states into the same one-world model and camera, side-by-side. The
`max_lift_endpose_diagnostic` field is intentionally separate from
`task_success`; sustained airborne genuine hand-object contact, path fidelity,
slip and release are required for a future success claim.
