# Seeing the hand's residual motion range

Render the current22 finger-DoF command envelopes as paired, fixed-camera views
of the same right-hand asset:

```bash
MUJOCO_GL=egl PYTHONPATH=. python tools/visualize_hand_residual_ranges.py \
  --operator cheyingtong --output outputs/hand_residual_ranges
```

Requires MuJoCo rendering, Pillow, imageio/FFmpeg, materialized hand assets,
and Noto Sans CJK Regular for the labels. The output directory must not exist.
`--preview` renders the cover and numeric manifest only. No source assets,
Lance data, controller defaults or running training jobs are modified.

## What the animation means

The gray hand is a fixed illustrative, slightly curled reference pose. The
cyan hand changes one finger coordinate at a time, holding at both extremes.
Orange screen-space markers locate the current joint even when the skin hides
its pivot. Both views use fixed framing and the same joint configuration.
The angle amplitude is not exaggerated. The camera zoom is illustrative.

For each coordinate, with normalized action magnitude at most1 and decay gamma,
the limiting residual reachable from a zero state is:

```
envelope = min(configured_cap, per_step_scale / (1 - gamma))
negative = max(-envelope, joint_lower - reference_angle)
positive = min(+envelope, joint_upper - reference_angle)
```

The renderer reads current `ResidualActionConfig` finger scales, multipliers
and decay, plus source-compiled joint limits. In the current configuration,
four-finger DIP hard caps are0.2rad, but the zero-state limiting offset is0.1rad;
the video uses0.1rad. Limits are asymptotic under sustained maximal actions;
this slow geometric sweep is not a literal time sequence of policy actions.

These are kinematic **command-range illustrations**, not sampled PPO actions,
measured joint tracking, or guaranteed collision-free motion. No physics step
is performed. The reference pose is disclosed in the manifest and is not a
selected training frame. A different reference can clip differently at the
same anatomical joint limits.

## Outputs

- `hand_residual_ranges.mp4`:22segments,4seconds each by default,24fps.
- `preview.gif`:a short index-PIP animation for inline viewing.
- `preview.png`:cover image showing the positive index-PIP endpoint.
- `dof_06_negative.png` through `dof_27_positive.png`:both endpoints per joint.
- `index.html`:local video player, per-joint chapter links and endpoint images.
- `ranges.json`:full reference posture, actual residual configuration, source
  asset identity, joint order, limits and displayed envelopes.
- `asset_manifest.json`:the explicit hand profile used for this rendering.

Every generated frame is checked to vary no more than its designated coordinate
and to remain within the compiled joint limits. Unit tests compare the computed
envelope to400steps of the residual recurrence and test endpoint holds.
