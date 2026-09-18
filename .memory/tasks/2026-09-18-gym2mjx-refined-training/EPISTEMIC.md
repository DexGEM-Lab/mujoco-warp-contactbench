# Epistemic Model

## Phenomenon
The refined RL dataset is structurally high quality in its source runtimes, but the operational question is transfer: can its stored 28-DoF targets physically carry mayonnaise under the fixed `cheyingtong` hand and current MJX-Warp object/solver contract, and can 1× joint residual training repair the gap?

## Supported mechanism
The import boundary is solved. Pylance 0.24.1 can parse the dataset manifest but panics on encodings21 data pages; pylance 7.0.0 decodes them. The explicit refined-RL importer keeps this schema separate from raw captures and canonical ManoRL generated references, preserves source RL path/version/row/UUID, and emits a standard Lance-free MTP bound to the Cheyingtong manifest.

Timestamps, not `trajectory_metadata.data_fps`, define physical duration. Of 600 mayonnaise rows, 437 declarations are wrong: 481 rows have exact 5 ms spacing (200 Hz) and 119 have 1/120 s spacing. Timestamp-duration interpolation to 120 Hz retains the final pose with at most one 8.33 ms edge hold. All 600 rows decode, producing 226,190 package frames and zero deterministic rejections (OPS compile and clock entries).

Physical transfer is weak but learnable. Zero-residual MJX replay completes 14/600 horizons (2.33%): action01 0%, action02 0%, action03 2%, action04 5%, action05 0%, action08 7%. Success is similar across observed source clocks (120 Hz 1.68%, 200 Hz 2.49%), so resampling origin does not explain the failure. A direct failed/successful pair comparison localizes the mechanism: the failed row has only 8 contact frames, 1.03 N peak force, and 2.94 mm actual displacement while its target moves 10.17 cm by call58; the successful row has 146 contact frames, 7.24 N peak force, and follows a 3.58 cm target displacement through call781. Most fixed-hand references make contact too weakly or briefly to transport the object before the 10 cm deviation boundary.

The 1× residual policy can improve that contact mechanism. A fresh N96/U64 smoke increased deterministic six-pair evaluation return from 80.53 to 157.60 and contact reward from 0.02165 to 0.05186, converting action01 from failure to a completed horizon. This is direct evidence that training can compensate part of the fixed-hand transfer gap; it is not evidence that the full catalog is solved.

## Ruled out
The dataset is neither empty nor logically corrupt: version, schema, row count, selection manifest and all selected rows are coherent under pylance 7. Initial world-coordinate mismatch is not the dominant replay failure: target error begins near zero and grows as the target moves while the object remains nearly stationary. Declared FPS cannot be trusted, and observed 120 versus 200 Hz does not stratify replay success. Treating all generated rows as canonical ManoRL physical references would falsely assert provenance and asset parity.

The initial 600-world replay OOM does not imply N600 training is infeasible. That attempt used 128 contact slots/world while another GPU1 process restarted. Six 100-world replay batches completed, and an N600 training preflight with 64 slots/world completed five updates at 4.54k transitions/s with about 13.6 GiB still free and no overflow warning.

## Anomalies
The 14 direct replay successes cluster in a few source cohorts, especially one weekly `gg` cohort for actions04/08; source clock alone does not explain this. Action05 is 0/100 and its entire source cohort is declared100/observed200. The broader 12,200-row catalog contains additional objects and contracts; only the six mayonnaise pairs are justified by current physical evidence.

## Current claim
The complete mayonnaise subset is imported, replayed, and under active production training on Server1. The fresh N600/U8000 run uses every one of the 600 trajectories exactly once per vector assignment, Cheyingtong assets, 120/480 Hz, joint increment/cap multipliers1.0, minibatch3600 and 64 contact slots/world. It crossed durable checkpoint100 and reached update110 /3.168M transitions healthy at 4.13k transitions/s; update110 had reward0.2787, contact0.0464 and4 successes/352 episodes. The checkpoint sidecar binds the requested clock, multipliers, package and asset hashes. No Lance/PyArrow mappings or overflow/OOM/non-finite log evidence is present. W&B run `vpcg1w7r` and tmux `manorl-gym2mjx-mayo-production-g1` are the live observability points (OPS production entries).

## Most informative next observation
Checkpoint200 is the next learning boundary. Compare rolling per-action success/contact/return against checkpoint100, zero replay and the N96 smoke while verifying GPU memory and contact/constraint logs remain stable. A rise in success across actions01/02/05 would support the contact-compensation mechanism; success confined to the already transferable action03/04/08 cohorts would show that one shared policy is exploiting easy references rather than repairing the full fixed-hand gap.
