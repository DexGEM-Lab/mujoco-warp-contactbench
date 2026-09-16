# Evidence

## 2026-09-16T18:20:42+08:00 — implementation and focused validation
Shared ambient/diffuse constants now reach both scene XML builders and the tiled
viewer; the hand-range fixture inherits them. All 44 focused tests passed in
`test_render_defaults.py`, `test_view_environment.py` and
`test_hand_residual_visualization.py`.

Compiled MuJoCo models, with and without visual meshes, expose RGB 0.4/0.65.
Removing the headlight XML leaves mass, inertia, damping, friction, collision
masks, solref/solimp, actuator settings, gravity and timestep identical.
Validation used the primary virtualenv and shared read-only asset store;
no submodule or live training process was changed.
