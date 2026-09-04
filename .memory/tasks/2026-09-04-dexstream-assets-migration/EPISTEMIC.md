# Epistemic Model

## Phenomenon
ManoRL had mixed the old object submodule, old hand submodule, and copied
runtime meshes. The integrated target is one reproducible DexStream source for
both physical objects and MANO hand, with a fresh-checkout LFS closure and no
fallback that can silently mix geometry.

## Supported mechanism
- `assets/dexstream_digital_assets` is pinned to
  `f98da997f316c8a6b4bc2931cabed19e831ef163` from
  `git@github.com:DexGEM-Lab/dexstream_digital-assets.git`.
- ManoRL uses `hand/mano/sunke/{right,left}` and preserves the 28-DoF z-up
  contract. MANO axes are read from the pinned URDF.
- The adapter discovers 28 same-name rigid `objects/DexGEM` bundles, reads every
  URDF-declared collision/visual path and scale, and normalizes only the two
  uppercase camera directory names at the API boundary.
- The generated manifest pins all runtime-required Git/LFS files: hand URDF,
  metadata, 16 visual meshes, 16 collision meshes, skin fragment, `.skn`, and
  `.npz`; object URDF/visual/collision files are pinned likewise. Optional USD
  assets are outside the ManoRL runtime closure.
- Expected-contact aliases are ManoRL task metadata, not physical geometry.
  Strict checkpoints and generated-data manifests record the DexStream identity.

## Evidence
- Fresh skip-smudge setup on the integrated `dev` line materializes and hashes
  the complete runtime closure; no selected hand runtime file is an LFS pointer.
- All 28 homogeneous object models and right/left/bimanual unified models
  compile with visual skin enabled.
- Post-closure asset/scene/submodule/grasp/checkpoint/Lance tests pass 58/58.
- A real v530 `banana:09` trajectory loads and steps through CPU MJX-Warp with
  `num_envs=1`, 480D observations, and nonzero contacts.
- With `DISPLAY=:1`, a single-world banana+MANO scene renders visibly and the
  saved image shows a complete, continuous MANO skin and correct banana visual.
- Old physical asset directories are absent from the primary `dev` worktree;
  only unrelated pre-existing user files remain dirty.

## Known physical boundary
The current DexStream snapshot is a new physical version: MANO palm collision
scale is 1.0 rather than the removed runtime's 0.7, and several object CoACD
and inertial assets changed. Legacy checkpoint tensor compatibility therefore
does not imply rollout/contact equivalence. Strict resume/inference rejects
missing or mismatched asset provenance; policy-transfer/warm-start is explicit.
The source no longer contains `bottlewithcap` or `scissor`, so those names fail
rather than being mapped to different geometry.

## Known operational boundary
The historical default trajectory viewer fixture is unavailable locally, but
modern v530 env=1 and asset-only visual paths are validated. External X11
`ImageGrab` during GLFW teardown can return 139 after a rendered frame; use the
in-process renderer for automated screenshots and clean exit status. This is a
viewer shutdown interaction, not an asset/model failure.

## Current claim
The DexStream asset migration, fresh-checkout closure, env=1 runtime check, visual
acceptance, and an isolated Server1 deployment are complete and integrated on
`dev`. The deployment uses the same root/source pins and a fully materialized
298-file runtime closure without depending on Server1 GitHub credentials. The
migration branches used for implementation and evidence are merged and can be
removed without losing reachable commits; unrelated worktrees, active jobs, and
legacy snapshots must remain.
