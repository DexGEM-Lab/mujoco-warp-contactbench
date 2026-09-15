# Raw capture replay interpretation

Procedure: `docs/raw_capture_replay.md`. Pinned September 14 case results:
`docs/sept14_raw_replay_results.md`.

Use explicit scene/object slots and verify the recorded hand betas against the
selected source operator. The replay-only manifest leaves the default training
profile unchanged. Preserve full source approach/release frames and passive
scene bodies; selecting the right hand need not load a passive left model.

Separate grasp acquisition, load-bearing retention, trajectory tracking, and
release. A height threshold on the object reference point can miss a real grip
when a heavy object is held low or tilted. Conversely, a transient height apex
without hand contact is not a retained grasp. Inspect actual contact/support
states and the recorded physics, not the kinematic comparison panel.

Before attributing a heavy-object lift deficit to grasp failure, compare wrist
position error with payload mg/k. The 1.28kg pitcher at 100N/m predicts 12.6cm
static sag; the source-matched raw replays establish grasp but show this scale
of tracking deficit and later placement/withdrawal failure. This case does not
establish that RL is necessary.
