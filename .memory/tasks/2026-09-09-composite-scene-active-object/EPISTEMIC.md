# Current model

## Phenomenon

The new Lance capture represents a scene containing two tracked objects, while exactly one object is manipulated per trajectory. ManoRL's physical runtime supports batches whose environments each activate one of several object types, but trajectory discovery previously interpreted the full `index.scene` string as one object identifier.

## Mechanism

Modern-row discovery now parses the ordered scene-object list and uses the unique `trajectory_metadata.trajectory_info.object_move` entry as the active policy object. Its position in the scene list selects the corresponding `objects` state. The resulting standard `object_action_sequence` identity feeds the existing homogeneous or unified-object runtime without changing policy dimensions or environment routing. Ambiguous rows with zero or multiple manipulated objects fail discovery/decoding instead of silently selecting a state.

Policy-free Rerun recording now couples control FPS to the requested reference FPS when no checkpoint is loaded. Checkpoint replay continues to restore its recorded control-clock ABI.

## Supported claim

All 59 version-5 rows are valid candidates: bowl:03=19, bowl:07=10, bowl:09=10, mayonnaisebottle:05=12, pitcherbase:06=8. Runtime assets and grasp mappings exist for all five pairs. A production MJX-Warp GPU replay of bowl:03 completed its full reference horizon with residual actions disabled and produced a valid Rerun recording (OPS.md, adapted dataset validation and policy-free GPU replay).

The selector intentionally balances environment slots across object/action pairs. Consequently, 59 simultaneous environments do not mean one environment per source row; a 95-environment window is the smallest first-cycle window that covers all 19 rows of the longest pair while retaining equal pair allocation.

## Boundary

The passive second scene object is used only to map the active object's state index and is not materialized in ManoRL physics. The replay verifies active-object pose decoding and executable source following. It does not establish equivalence when the passive object is a physical support or obstacle; that requires an explicit multi-body scene model.
