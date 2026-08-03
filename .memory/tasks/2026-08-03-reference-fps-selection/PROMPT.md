## Objective
Expose an explicit reference/source trajectory FPS selector with exactly two supported values, `100` and `120`, for ManoRL training and inference. Both paths must preserve the existing 400 Hz physics and 200 Hz policy/control clocks while replaying source motion at the selected physical rate.

## Success criteria
- Training and inference/viewer entrypoints accept the same named reference-FPS option, restricted to `100|120`.
- Source trajectories are time-resampled onto the existing 5 ms control grid; hand joints and object position use linear interpolation after angle unwrapping, and object orientation uses quaternion SLERP.
- Source movement/contact windows, source-index provenance, and trajectory termination preserve the selected source duration.
- Physics remains `0.0025 s × 2 substeps = 0.005 s/control`; PPO and residual-action timing do not silently change.
- Checkpoint/runtime metadata records the reference FPS and resampling contract; inference restores it or rejects an explicit conflict.
- Stable shell entrypoints expose the option, focused tests cover both rates, and documentation states the clock separation.

## Evidence motivating the change
The active v295 full-pair run consumes one stored row every 5 ms. Across its 1,919 unique assigned Lance rows, metadata reports 119/120 Hz and row-average timestamp spacing is approximately 8.333 ms, while the user states raw acquisition is 100 fps. The implementation currently ignores timestamps for simulation timing, so reference and control clocks are conflated.

## Constraints
- Work only in `feat/reference-fps-selection`; do not modify primary `dev` directly.
- Do not change or regenerate authoritative Lance datasets.
- Do not stop or modify the currently running production training without a separate explicit user request.
- Preserve fail-closed checkpoint semantics and explicit GPU/CPU behavior.
- No subagents: the user did not authorize delegation.
