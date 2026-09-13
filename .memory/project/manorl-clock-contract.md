# ManoRL clock contract

ManoRL records three coupled clocks in current v2.3 checkpoints and Lance rows:

- reference/source: explicit `100|120` Hz for public coupled modes;
- policy/control: the selected `100|120` Hz clock (`0.01` or `0.008333... s`);
- physics: four integer substeps per control period (`400|480` Hz).

Legacy v2.2 checkpoints and rows retain the historical clock: policy/control
`200 Hz` (`0.005 s`) and physics `400 Hz` (`0.0025 s`, two substeps). The
clock resolver treats an omitted/200 Hz value as this legacy mode. Compact
replay rows make the resolved source contract, reference/control frequencies,
physics frequency, timesteps, and substep count explicit so replay never
assumes the legacy clock.

Source trajectories are resampled onto the selected control grid before
environment table construction. Hand angular coordinates are unwrapped over
time before linear interpolation, object position uses linear interpolation,
and object orientation uses quaternion SLERP. Source movement windows and
termination are mapped by physical time. The checkpoint-bound
`reference_fps`/`control_fps` pair must be restored for inference and export;
an explicit conflict is rejected.

Canonical usage and rationale: `README.md` → “ManoRL PPO Training” and
“Compact synthetic Lance synthesis”; `docs/manorl_cube1_training_protocol.md`
→ preconditions/checkpoint sections. Implementation:
`sim/manorl/contracts.py`, `sim/manorl/trajectory.py`,
`sim/manorl/view_environment.py`, `sim/manorl/lance_v2.py`, and environment
ABI v7 in `sim/manorl/abi.py`.

## Autonomy first-learner boundary
The contact-conditioned autonomy M3 learner currently uses an ordinary single
Gymnasium environment for honest N=1 CPU/GPU validation over the canonical
MJX-Warp clock/contact producer. It uses skrl `RlGamesPPO` and canonical GAE with
explicit finite-horizon termination; terminal observations are recorded before
an explicit reset. This first learner is intentionally not a vectorization or
generalization claim. Final autonomy acceptance requires a frozen checkpoint to
consume a new held-out reference without per-reference retraining, manual masks,
or special reward; max lift plus endpoint pose is insufficient.

## v4 optimizer continuation boundary
Autonomy v4 optimizer resume equality-gates physical clock, asset/package/ABI
provenance and fixed training configuration. Warp physical state is not saved:
continuation declares `full_start_new_episodes`, resets episode telemetry, and
restores sampling RNG after reset. Cumulative PPO counters and Adam continue.
See `docs/manorl_autonomy_v4.md` → “Continuing a v4 PPO run”.

## v4 teacher supervision boundary
Optional online teacher-action anchoring is post-PPO training supervision only;
pre-step runtime labels cannot enter the physical adapter's execution path.
Anchor-only Adam steps must clear gradients to `None` so value-only momentum
cannot move critic parameters. Checkpoints record the complete teacher recipe;
old absent metadata means disabled anchoring. See `docs/manorl_autonomy_v4.md`
→ “Optional online teacher-action anchor”.
