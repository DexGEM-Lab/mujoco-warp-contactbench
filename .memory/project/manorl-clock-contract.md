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
