## Phenomenon
The v2 exporter labeled the physical wrist/keypoint body pose as MANO global pose. The body carries an almost constant pi rotation relative to the URDF floating root and a translation offset up to 1.9 mm. Shape metadata also stored left/right shapes while declaring only right active.

## Mechanism
The refined MANO contract is defined by the 28D URDF state: translation is q[0:3], and global rotation is the axis-angle representation of intrinsic XYZ `Rx(q3) @ Ry(q4) @ Rz(q5)`. The physical body transform belongs to physics/contact computation, not the MANO global field. Shape rows follow active `hand_names`, while fixed `hand_slots` may retain an empty left slot.

## Current commitment
Publish a new schema identity, compute MANO globals solely from exported 28D state, select and validate the raw right shape, regenerate/validate before deleting any superseded artifact.
