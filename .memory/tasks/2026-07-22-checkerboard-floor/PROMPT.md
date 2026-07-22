# Checkerboard Floor

Make the generated ManoRL floor use a MuJoCo checkerboard texture by default in
both homogeneous and unified scenes. Preserve all floor physics properties and
checkpoint/runtime contracts; this change is visual only.

Owned files:

- `sim/manorl/assets.py`
- `tests/manorl/test_assets.py`
- `tests/manorl/test_environment.py`

Acceptance:

- Both generated scene variants define and bind the same checkerboard material.
- Floor position, size, collision masks, and friction remain unchanged.
- Focused asset and environment tests pass.
