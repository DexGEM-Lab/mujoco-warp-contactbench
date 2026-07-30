## Phenomenon
The previous reward became neutral after the raw contact window even while the hand remained physically against the object. The previous exporter also held every accepted nested row in one Python process, so repeated 49-identity rounds accumulated host memory and native-library lifetime.

## Mechanism
The v2 reward keeps expected-contact reward inside the inclusive raw `object_move.end_frame`, gives ten neutral frames, then applies `-1.2` whenever any of the 16 physical hand keypoint forces is strictly above `0.2 N`. This closes the unlabelled-link loophole. Host NumPy and device JAX paths implement the same predicate and boundaries.

Repeated synthesis is an accepted-sample process: each raw identity needs five complete episodes within ten completed attempts. Episode index, generation attempt, attempt seed, generated UUID, and raw seed UUID are distinct persisted concepts. Each attempt round runs in a fresh child process and appends one Lance fragment. This bounds native MJX-Warp/Lance lifetime and host memory while the parent maintains per-identity counters and atomically promotes only a complete dataset.

## Evidence
The first non-isolated 49-identity run completed three rounds, reached 147 accepted rows, then segfaulted while serializing row metadata in round four. RSS had grown to about 7.4 GiB. Fresh-process isolation removed the repeated native lifetime; the complete run reached 245/245 rows with all 49 identities at `(attempts=5, saved=5)` and no failed candidates.

Full isolated-row validation established: schema v2.2; 49 identities × 5 episodes; 107,790 state frames; 107,545 transitions; unique 245 generated UUIDs; episode seeds 42..46; generation attempts 1..5; exact MANO root position; maximum MANO root rotation error `1.217e-7 rad`; maximum right-shape error 0; and force-frame errors below `1.1e-6 N`. It observed 26,923 late penalty steps with minimum contact reward `-1.2000000477`. The one-identity boundary probe had contact reward `+0.4` at `contact_end`, 0 through `contact_end+10`, and `-1.2` beginning at `contact_end+11` even though raw expected-contact reward was 0, directly confirming the any-contact mechanism.

## Current commitment
The v2.2 dataset is scientifically and structurally valid. It still uses the frozen v5 checkpoint, so the penalty exposes persistent contact but does not teach release behavior. Behavioral change requires retraining under the v2 reward contract. Publish the 245-row artifact atomically, preserve the current v2.1 artifact until NAS readback/checksum validation succeeds, then integrate the code into `dev`.
