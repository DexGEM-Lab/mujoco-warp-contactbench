# v2 container contact-primitive transfer

Every receiver replaces its full hand motion with a retimed validated donor `trajectory.npz:ctrl` curve. A yaw-plus-XYZ transform maps the donor’s *actual* initial active-object state to the receiver physical active-object start; finger values receive no residual addition. Receiver object and passive-bowl reset poses remain unchanged.

The hand translation receives two smooth, measured-scene conditions: donor-to-receiver bowl displacement over the actual donor approach/return, then captured receiver-final-active displacement over placement. The small donor/receiver resting roll-pitch mismatch is recorded and applied to wrist orientation rather than tilting an object off its floor support.

These assets replace receiver noncontact approach as well as contact motion. They are CPU artifacts, not replay outcomes.
