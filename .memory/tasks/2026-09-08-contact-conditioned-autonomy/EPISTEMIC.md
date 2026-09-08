# Current model

## Objective
Accurate demonstrations specify hand/object geometry and contact intent but not physically valid actuator commands. The autonomous actor must own all 28 commands and learn physical grasp/lift/transport/place/release under the pinned simulator.

## Supported mechanism
M1's standalone path was corrected in M2 rather than treated as competence. `simulation_clock(120)` and `compile_model(... physics_timestep=1/480)` enforce 120 Hz control and four 480 Hz physics substeps. Raw object pose plus initial quaternion and actual collision vertices define one support translation; that translation is applied to every reference hand XYZ/object frame and reset while source arrays remain immutable. Reference FK is evaluated per frame.

Reference and measured contact intent now use `mj_geomDistance` over actual hand-segment and object collision geoms. The nearest signed-distance witness endpoint is transformed to object-local coordinates and receives distance/penetration confidence. Solved hand-object forces come from `MjxWarpPhysicalProducer.hand_object_force_on_object_world_N`; table contacts cannot masquerade as hand-object force. Object-local relative keypoint motion is an explicitly named slip proxy masked by measured hand-object force.

The action map has explicit physical rates (.5 m/s wrist XYZ, 2 rad/s wrist rotation, 4 rad/s fingers) and measured-state envelopes (.02 m/.25 rad/.35 rad). The envelope bounds integrated target windup around measured state without clamping to a reference, preserving load-induced servo error. The reference-pursuit diagnostic actor computes a bounded next-command action; the command map sees only that output and measured state.

Reward compares object path and orientation, reference hand/object relation, demonstrated contact proximity/anchors, measured hand-object contact, finger configuration, velocity and masked slip. Net supporting force is trace-only; impact magnitude is not rewarded. Release is inferred from the demonstrated proximity window. Drop/path-divergence termination is distinct from horizon completion.

## Evidence
Prior negative traces showed zero hold could earn high reward while target cube lift was 19.4 cm, actual hand-on-object force exactly zero, and contact_count 35/per-hand geometry force represented table contacts. Prior pursuit diverged at 233 after large XY displacement; mean reference proximity max 0.0744 made the old release threshold unreachable.

After correction, six semantic tests pass. Same identity `cube2_02_2833` full-start diagnostics (538 frames) complete for zero and bounded reference pursuit. Corrected zero still has zero hand-object force and no lift despite 35 contacts; corrected pursuit has hand-object force up to 2.07 N over 84 frames and slip proxy up to 0.281 m/s but no lift competence. Both traces record source package, contract IDs, clock and config hash. Full package summary loads all 50 identities.

## Boundary and next question
M2 establishes a discriminating runnable diagnostic, not learned competence. The next training interface must reuse canonical GAE/PPO with identity splits and evaluate full-start episodes; no standalone PPO smoke remains. The next causal question is whether a canonical policy can convert demonstrated collision witnesses and measured hand-object wrench into sustained support and target lift without table-contact leakage.
