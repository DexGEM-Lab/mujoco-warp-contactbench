# Epistemic Model

## Phenomenon
A high-quality filtered RL-episode dataset must become a physically replayable and trainable ManoRL reference catalog. The immediate question is whether its recorded Gym/MuJoCo trajectories remain dynamically executable under the current MJX-Warp runtime, not merely whether the stored arrays look successful.

## Supported mechanism
The first import failure is a Lance reader-generation mismatch: project pylance 0.24.1 recognizes the dataset manifest but cannot decode encodings21 pages, while pylance 7.0.0 reads them. A compatible isolated decoder is therefore necessary before any ManoRL schema judgment.

The row schema is intentionally smaller than raw capture and canonical ManoRL synthetic schemas. It represents complete generated RL episodes: object/action identity is in `index.scene` and `index.action_code`; full timing, 28-DoF hand targets, one object pose track, and source-RL provenance are present; movement annotations and canonical checkpoint provenance are absent. The existing modern-row decoder already supports a missing movement annotation by treating the full row as the movement interval, but candidate discovery currently rejects these rows because `is_generated=true` is reserved for a different strict generated-reference contract.

`trajectory_metadata.data_fps` is not an authoritative clock for this dataset. Across the 600 mayonnaise rows, 437 declarations disagree with exact timestamp spacing: 481 rows are physically sampled at 200 Hz and 119 at 120 Hz, while 299 of the 200 Hz rows declare 100 and another 138 declare 120. Action05 is uniformly declared100/observed200. Interpreting frames from metadata would stretch most episodes and corrupt controller dynamics; elapsed timestamps are the supported timing mechanism.

An explicit refined-RL import mode should separate these semantics instead of weakening the canonical generated-reference validator. It should use the full episode, preserve source provenance, normalize every 100/120/200 Hz row to the user-fixed 120 Hz clock, bind the user-fixed `cheyingtong` physical hand profile, and package decoded arrays so the trainer remains Lance-free.

## Ruled out
The source path is present and the dataset is not empty or corrupt at the logical level: version, schema, row count, selection manifest, and rows under pylance 7.0.0 are coherent. Updating only a dataset path cannot solve the import. Treating all `is_generated=true` rows as canonical ManoRL generated references would erase a real provenance/clock distinction.

## Anomalies
The source catalog has two observed simulator clocks, 120 and 200 Hz, despite three declared metadata values; the user resolved the output contract by requiring 120 Hz for every row. Timestamp-duration resampling must preserve physical time while changing sample count. The dataset also contains objects/actions beyond the mayonnaise replay subset; current physical-runtime coverage must be measured before promising all 12,200 rows in one production run.

## Current claim
The import mechanism is now validated through the complete mayonnaise catalog: pylance 7 decodes encodings21, all 600 rows pass the strict RL schema/provenance checks, timestamp-duration resampling produces 226,190 frames at 120 Hz with no more than one output-step terminal hold, and the MTP binds the Cheyingtong manifest with 0 rejected rows. Package-only loading/assignment is deterministic and Lance-free. Physical replay success remains unknown until Server1 executes the 600 worlds under zero residual actions; structural import quality cannot substitute for that measurement.

## Most informative next observation
Stage the verified 600-row MTP and exact Cheyingtong manifest on Server1, execute one 600-world zero-residual MJX replay, and inspect per-action deviation failures. A high completion rate supports proceeding to 1×-joint training; systematic failures localized by action, source clock, or source RL lineage would identify a coordinate, asset, or controller mismatch.
