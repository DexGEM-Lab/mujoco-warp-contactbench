# ManoRL clock contract

ManoRL separates three clocks:

- reference/source: explicit `100|120` Hz;
- policy/control: fixed 200 Hz (`0.005 s`);
- physics: fixed 400 Hz (`0.0025 s`, two substeps per control target).

Source trajectories are resampled onto the control grid before environment table construction. Hand angular coordinates are unwrapped over time before linear interpolation, object position uses linear interpolation, and object orientation uses quaternion SLERP. Source movement windows and termination are mapped by physical time. The checkpoint-bound `reference_fps` must be restored for inference/export; an explicit conflict is rejected, and legacy checkpoints without this field retain legacy frame-per-step behavior.

Canonical usage and rationale: `README.md` → “ManoRL PPO Training”; `docs/manorl_cube1_training_protocol.md` → preconditions/checkpoint sections. Implementation: `sim/manorl/trajectory.py`, `sim/manorl/view_environment.py`, and environment ABI v7 in `sim/manorl/abi.py`.
