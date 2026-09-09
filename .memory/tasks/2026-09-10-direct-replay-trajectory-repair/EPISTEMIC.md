# Current model

## Delivered behavior and remaining scope
Five representative source-v5 rows, one per action category, have physically
validated repairs: row0 stove-to-table bowl, row19 bottle tilt/return, row31
pitcher tilt/return, row36 table-to-stove bowl, row49 lift-and-hold bowl. The
remaining54 rows are unchanged. The user's clarified objective is the first
five selected trajectories, not all59. Canonical instructions and scope:
`docs/direct_capture_repair.md`.

Each example has a version/UUID-bound JSON patch, optional hashed frozen
actuator targets, actual post-step motion Lance/NPZ, full video and inspected
storyboard. All objects move only via free-body physics after initialization.
No RL implementation was inspected and no learned policy was trained. Recipe
solver settings are explicit, local, and do not change dev defaults.

## Mechanisms supported by experiments
- Retargeted finger gaps, intermediate-link penetration and premature closure
  break acquisition. Contact-aware wrist/finger fitting, an open approach and
  object-relative transport repair it. Passive supports remain physical bodies.
- Default pyramidal/impratio1 soft friction creeps. Same bottle grip under
  elliptic/impratio1 drifts~8.14mm/s late; impratio100~0.118mm/s. Source bowl hand
  still fails with improved solver, showing geometry and solver are independent
  requirements. Mass, mu, gravity and actuator gains remain unchanged.
- Pitcher1.28kg loads a100N/m wrist servo: roughly14cm sag is expected. Staged
  load compensation in target space after contact enables lifting. Construction
  feedback helped author commands; final independent replay uses frozen commands.
- Corrected Euler commands require nearest representation relative to actual
  joints; a2pi return jump previously flipped the pitcher.
- Handle release requires clearing thumb then sideways finger withdrawal.
  Radial pulling or uncurling inside the loop hooks and topples the pitcher.
- Moving-bowl examples needed a translated proven grasp primitive, explicit
  axial-orientation alignment, slower carry, leveling and release. Their timing
  is deliberately longer (25.05/25.25s), not claimed as original-speed replay.

## Accepted sample evidence
- row49: extra10s held-tail minimum lift20.26cm, final tilt4.94deg, final position
  error6.18cm, last-second movement<0.2mm.
- row19: maximum lift12.46cm; representative pouring frame error1.43cm/2.68deg,
  bottle axis intersects recipient bowl interior; final tilt1.06deg, error2.78cm.
- row31: max lift16.66cm, full open-loop return and release, final tilt0.45deg,
  error5.56cm, final no finger contact.
- row0: max lift24.86cm, carry/level/place/release; final tilt0.39deg,
  error2.88cm to edited target.
- row36: max lift25.39cm, placed back on physical cuboid support; final tilt0.35deg,
  error4.17mm to edited target. Initial bowl lowered7.74mm to meet table.
No fluids are simulated; pouring evidence covers container geometry/motion only.

## Data and replay invariants
Generated-motion Lance is measured output with actual hand qpos and separate
command_target_dof, not an unmodified source-capture input. Live reproduction
uses original source plus patch/frozen commands. Frame mappings document
correspondence, not original timestamps for retimed primitives. Small GPU contact
variation is present; task behavior, not bitwise trajectory identity, is verified.

Trace body/joint order must match exactly. Runtime sorts object types; a manual
scene-order probe once swapped bowl with bottle/pitcher. Those contact probes are
invalid, while viewer-ABI-matched saved poses stand. Native coordinate mirrors do
not imply valid native contact buffers; native recomputed forces are separately
labeled from MJX solver forces.

## Final task state
Published NAS bundle `dexgem_vla_demo_guangxue_astra_repair_5samples_20260910`
contains five-row/8664-frame generated-motion Lance, complete physical state
recordings, immutable patch/command assets, videos, previews, source catalog,
Chinese README and executable replay/play_all launchers. Content round-trips
exactly. Re-executing all five from NAS preserves task outcomes: final placement
tilts<1.2deg, held bowl<5deg and >20cm lift. Small placement differences between
replays are recorded, not hidden. The DISPLAY=:1 playlist completed; the remaining
optional live loop and stale viewer were stopped after delivery to release GPU0.
Do not restart them without coordinating with the other training task. Videos
remain usable without simulation.
Source version5 and its59 rows remain untouched. The complete 62-file NAS
checksum inventory matches current contents. User-authorized independent
read-only review reproduced source identities, measured task outcomes and
export/replay semantics from the decisive artifacts, with no blocking defect.
The reviewer lacked image capability; visual judgment remains grounded in the
parent's earlier direct inspection of all five storyboards and desktop replay.
The first-five objective is complete under the explicit recorded solver profile;
default-profile success and original-timing fidelity are not claimed. Final
source-branch checkpoint and artifact paths are recorded in OPS.md.
