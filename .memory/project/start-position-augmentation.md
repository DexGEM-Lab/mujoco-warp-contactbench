# Physical hand-start augmentation

Canonical method: `docs/start_position_campaign.md`; executable planner/queue
`tools/run_start_augmentation.py`. Only42 strict Cheyingtong120Hz repaired parents
are eligible; B035 is functional-only and excluded. Published parent bundle and
all original states stay immutable.

Augment both actual initial handXYZ and early desired wristXYZ by the same
positional offset. The quintic residual reaches exact zero at C-1; derivative at
C and the remaining repaired targets are unchanged. Never reset physical or PID
state at C. Preserve120Hz/480Hz, prestep frame0/N-1 intervals and post-preload
finger corrections. Initial world/objects/velocities stay unchanged.

The0.1s precontact guard supports meaningful10cm starts in the tested short
B043 case: merge error~0.2mm and~0.007m/s. Old0.4s guard would leave only0.09s
of taper, making the same displacement violent. Choose convergence windows from
actual approach/controller evidence, not from a historical constant alone.

Same-parent32-world batching shares model/kernel work but retains independent
states, PID integrals, warmstarts and validation. Scalar/batch first controls and
initial/target arrays agree; physical traces need not be bitwise identical.

Initial production:42×4radii(3/5/7.5/10cm)×8octants=1344 real simulations,1261
accepted and83 rejected. Geometry preflight resamples within the same fixed cell
at most16times and records all rejections; physical failures are never replaced.
Failure rates are seed-dependent (A0206/32,B2007/32), not monotonic with radius.
These are measured accepted trajectories, not guaranteed repeatable commands.

Acceptance retains original action-specific gates and checks early hand/scene
contact plus>=25ms sampled unsupported no-hand flight above2cm. Diagnostics are
120Hz native geometry, not480Hz force closure. See candidate ledger and videos;
do not promote a seed label or transformed recording into a new success label.
