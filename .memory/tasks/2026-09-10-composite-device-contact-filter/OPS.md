# Operations evidence

- 2026-09-10: Verified host `_decode_contact_forces` filters reward force by per-world `active_object_geom_ids`, while device `reduce_warp_contacts` used global `object_geom_ids`. Implemented the matching static-table reduction.
- 2026-09-10: Focused primitive tests and checkpoint-contract tests passed with the worker virtualenv and `PYTHONPATH=.`. GPU/MJX composite transition parity remains a deployment-gated test because this worker has no configured CUDA environment.
- 2026-09-10: Parent ran `outputs/contact-parity/shared.py` on dataset-v5 bowl/cuboid1 (one shared post-physics host/device snapshot) for 260 steps, with an explicit reset at step 240. Observation error peaked at 2.17e-7; reward, done, and transition counters matched, and target-contact reward was nonzero. An earlier independent-simulation `check.py` comparison drifted in qpos by 5e-5, so this shared-snapshot test isolates the transition arithmetic rather than solver divergence.
