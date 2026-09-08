# ManoRL autonomous cube2 milestone

This path is a separate `manorl.autonomy.v1` contract. The existing residual
route and its checkpoints are unchanged.

## Run

The first physics check uses one environment and the content-addressed MTP
package:

```bash
PYTHONPATH=. /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python \
  tools/train_manorl_autonomy.py train \
  --package /home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/contact_conditioned_autonomy/cube2_02_v295_f120_pre180_post180 \
  --device cpu --updates 1 --horizon 32 --epochs 2 \
  --checkpoint outputs/manorl/contact_conditioned_autonomy/cube2_02_autonomy.pt

PYTHONPATH=. /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python \
  tools/train_manorl_autonomy.py evaluate \
  --package /home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/contact_conditioned_autonomy/cube2_02_v295_f120_pre180_post180 \
  --device cpu --checkpoint outputs/manorl/contact_conditioned_autonomy/cube2_02_autonomy.pt \
  --trace outputs/manorl/contact_conditioned_autonomy/cube2_02_eval_trace.json
```

`--near-contact` is an explicitly labeled training/diagnostic reset. Final
evaluation starts from the first package frame and does not use it.

## Contracts

* `autonomy_contracts.py` defines versioned action, observation, reward and
  checkpoint identifiers. All 28 DOFs are actor-owned from the first step.
* The actor action is a normalized 28-vector. `rate_limited_command` integrates
  the measured previous command, clips to physical joint limits, and bounds the
  per-step rate. Reference tensors are absent from this function, preventing
  hidden reference playback, additive targets, finger masks, and wind-up.
* Observations include measured q/qdot, object velocity, current/next/future
  reference q and object motion, hand-object relations, previous command,
  action identity, object dimensions, keypoint geometry, and per-keypoint
  nearest collision-mesh surface anchor/proximity/confidence. Surface features
  are geometric intent; they are not force labels.
* `object_pos_raw` remains available in the MTP. The autonomous route consumes
  the package's declared aligned `object_pos` and applies one shared support
  translation to hand/object reference relations without rewriting source
  arrays.
* Reward terms are additive and active before object movement: object motion,
  reference hand-object relation, surface proximity/contact, stability and
  release, with action smoothness and phase shaping. There is no binary
  `object_move.start` gate.
* The package is cube2:02, dataset version 295, 50 decoded identities, 120 Hz
  reference/control and 480 Hz physics metadata. Checkpoints record the package
  digest and manifest SHA-256 (`e826de23d4586611001230d340eaf0c68752b09988eed7928bf8884f056d8706`).

This milestone demonstrates runnable contracts and one-world physical stepping;
it does not claim learned full-start grasp/lift/transport/place competence.
