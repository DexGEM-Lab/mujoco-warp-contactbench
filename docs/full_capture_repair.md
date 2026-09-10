# Full Guangxue contact-trajectory repair

This continuation repairs every source-v5 row of
`dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance`.
The first five accepted examples remain immutable; new examples reuse their
physically validated grasp, hold, and release relationships. No learned policy is
trained. The scene remains free-body physics after reset.

## Meaning of a repaired row

Each row binds a source UUID and contains measured post-step object poses, actual
hand joint poses, and separate applied actuator targets. Some candidates retain
the original uncontacted prefix and source time grid; harder cases substitute a
validated contact primitive conditioned on that row's object placement,
recipient-bowl position and destination. These substitutions are explicitly
retimed task repairs, not framewise recovery of the original human kinematics.

All accepted replays use the explicit first-batch solver profile: elliptic
friction cone with `impratio=100`. This reduces soft-contact creep without
changing mass, Coulomb friction coefficients, gravity, or actuator gains. It is
local to the replay recipe, not a change to project-wide defaults. Successful
replay under the old default solver is not claimed.

## Evaluation

The registry has one entry per source UUID and never equates generated files with
successful task behavior. Actual-motion checks distinguish:

- **Hold:** lift clear of the support, retain opposing finger contact with no
  support contact, remain stable through the selected held tail. Final tilt and
  last-second drift are measured separately.
- **Transport/place:** travel toward the receiver-specific destination, end on
  the intended table or block, become upright, release hand contacts and stop
  moving. A high intermediate lift does not compensate for dropping at the end.
- **Container tilt:** the physical outlet must remain above the bowl opening
  while the container is tipped, not merely have a nearby center or pass an
  arbitrary tilt threshold. There is no liquid or delivered-volume simulation.

Geometric contacts are recomputed from saved qpos with the exact recorded object
order; these are not claimed as direct GPU solver-force telemetry. Continuous
height/pose arrays and sampled contacts are combined with visual inspection.
The same frozen command stream may exhibit small GPU contact variations, so task
outcomes, not bitwise state reproduction, define replay success.

## Tool chain

- `bowl_hold_template_transfer.py`: donor49 bowl-hold contact transfer with
  receiver-relative orientation and receiver-timed acquisition.
- `prepare_full_bowl_batch.py`: bowl task primitive conditioned on each source
  initial scene, lateral route and placement target.
- `refine_bowl_transport.py`: replace a failed carry/release segment using the
  receiver's measured stable grasp, retaining the successful prefix.
- `run_repair_inventory.py`: bounded sequential physical runs with explicit
  failure records and no overwrite of partial or accepted results.
- `evaluate_full_repairs.py`: outcome checks from actual saved motion.
- `build_repair_registry.py`: source-complete status and selected-run registry.
- `transfer_nearest_container_success.py`: preserve a nearby accepted container
  contact primitive, correcting scene alignment and recipient/destination offsets.
- `render_repair_registry.py`: previews/videos of selected measured motions.
- `publish_full_repair.py`: accepts only a registry covering all59UUIDs, checks
  command hashes and output value round trips, writes a new NAS bundle.

The sibling container case supplies its CPU-only v1/v2/v3 candidate generators.
Only the parent runs GPU jobs; competing workers do not share the simulator.

## Output compatibility

The generated Lance contract is `direct_repaired_physical_motion.v1`. It is an
actual-motion output, not interchangeable with source-capture Lance inputs.
For live reproduction use source + per-row patch + any hashed frozen command
track through the packaged `replay_repaired_capture.py`. The source capture and
first-batch bundle are never overwritten. Each row's validation and source-frame
mapping describe its individual fidelity and timing boundaries.
