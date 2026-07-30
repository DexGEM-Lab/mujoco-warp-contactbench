## Phenomenon
The previous reward became neutral after the raw contact window even while the hand remained physically against the object. The previous exporter also held every accepted nested row in one Python process, so repeated 49-identity rounds accumulated host memory and native-library lifetime.

## Mechanism
The v2 reward keeps expected-contact reward inside the inclusive raw `object_move.end_frame`, gives ten neutral frames, then applies `-1.2` whenever any of the 16 physical hand keypoint forces is strictly above `0.2 N`. This closes the unlabelled-link loophole. Host NumPy and device JAX paths implement the same predicate and boundaries.

Repeated synthesis is an accepted-sample process: each raw identity needs five complete episodes within ten completed attempts. Episode index, generation attempt, attempt seed, generated UUID, and raw seed UUID are distinct persisted concepts. Each attempt round runs in a fresh child process and appends one Lance fragment. This bounds native MJX-Warp/Lance lifetime and host memory while the parent maintains per-identity counters and atomically promotes only a complete dataset.

## Evidence
The first non-isolated 49-identity run completed three rounds, reached 147 accepted rows, then segfaulted while serializing row metadata in round four. RSS had grown to about 7.4 GiB. Fresh-process isolation removed the repeated native lifetime; the complete run reached 245/245 rows with all 49 identities at `(attempts=5, saved=5)` and no failed candidates.

Full isolated-row validation established: schema v2.2; 49 identities × 5 episodes; 107,790 state frames; 107,545 transitions; unique 245 generated UUIDs; episode seeds 42..46; generation attempts 1..5; exact MANO root position; maximum MANO root rotation error `1.217e-7 rad`; maximum right-shape error 0; and force-frame errors below `1.1e-6 N`. It observed 26,923 late penalty steps with minimum contact reward `-1.2000000477`. The one-identity boundary probe had contact reward `+0.4` at `contact_end`, 0 through `contact_end+10`, and `-1.2` beginning at `contact_end+11` even though raw expected-contact reward was 0, directly confirming the any-contact mechanism.

## Current commitment
The cube2 contact-v2 run was stopped after update 851; `checkpoint-000800.pt` is the latest complete scheduled checkpoint. A five-environment cube2:02 viewer completed every source-length episode without deviation termination, and the user observed that the hand no longer lingered on the object.

The authoritative all-action synthetic delivery now uses that checkpoint. It covers all 213 valid right-hand cube2 identities (`01,02,03,04,10,11`) with five accepted episodes each: 1,065 rows, 465,670 states, and 464,605 transitions. Every identity reached the target in exactly five attempts with no candidate failures. Full local and NAS isolated-row validation passed; 11 of 1,065 NAS row decodes required a bounded second attempt. The dataset contains 16,802 late-contact penalty steps (3.62% of transitions), exact MANO root position, zero right-shape error, and force-coordinate residuals below `2.2e-6 N`. It is published atomically with checksums under `/mnt/nas-222-project/sunjieqiang/mujoco_synthetic/cube2_all_actions_checkpoint800_ratio5_seed42_v22.lance`.

Cube1 contact-v2 training continues separately at N12288. Its source/reward/runtime modules must remain fixed until that run stops.

## Capacity boundary
On the 24 GiB RTX 4090, steady rollout memory is not the limiting measurement. N16384 and N14336 both execute three updates, then fail when the first reset-heavy transition requests additional Warp device storage. N12288 completes the same five-update test, including 8,088 resets in update 4. The justified production ceiling for this exact cube1/contact-v2/CCD contract is therefore N12288. N8192 remains the lower-memory operating point; N14336 is the nearest tested failing point.
