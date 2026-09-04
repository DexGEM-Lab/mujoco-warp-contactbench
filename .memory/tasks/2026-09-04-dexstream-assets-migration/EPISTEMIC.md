# Epistemic Model

## Phenomenon
ManoRL previously mixed the `assets/all_assets` object submodule,
`assets/mano_hand_s02` hand submodule, and copied hand/cube files under
`sim/manorl/runtime_assets`. The target is now implemented on the protected
`dev` line: one reproducible physical source for both hand and objects, with a
fresh-checkout installation contract and no silent fallback.

## Supported mechanism
- `assets/dexstream_digital_assets` is the sole physical source, pinned to
  `f98da997f316c8a6b4bc2931cabed19e831ef163` from
  `git@github.com:DexGEM-Lab/dexstream_digital-assets.git`.
- ManoRL consumes `hand/mano/sunke/{right,left}`. Both bundles satisfy the
  28-DoF `cmc3_mcp2_28dof`, z-up, 16-link collision/visual contract; pose axes
  are read from the pinned URDF.
- ManoRL discovers 28 same-name rigid `objects/DexGEM` URDF bundles and reads
  all collision pieces/scales from their declarations. `CAMERA1`/`CAMERA2`
  source directory case is normalized only at the adapter boundary.
- `dexstream_manifest.json` pins required Git/LFS files by source commit,
  SHA-256, and size. Its MANO file set includes the XML skin fragment, binary
  `.skn`, and bind `.npz`, so skip-smudge setup does not rely on a machine's
  unrelated LFS cache.
- Expected-contact aliases remain ManoRL-owned task metadata because the
  physical source does not provide them. Strict checkpoint signatures include
  repository, source commit, and manifest SHA; explicit weight transfer is
  the cross-version boundary.

## Evidence
- All 28 homogeneous object models and right/left/bimanual unified models
  compile with the materialized source. The fresh-checkout setup script and
  post-merge 32-test core suite pass.
- A real v530 Lance→CPU MJX-Warp path on `dev` loaded `banana_09_1384` with
  `num_envs=1`, returned `(1,480)` observations, and stepped through nonzero
  contacts.
- Local `DISPLAY=:1` rendering produced a manually inspected banana + MANO
  skin image; the surface is complete and continuous at finger joints. A
  collision companion image showed the expected hand links and banana CoACD
  pieces.
- The source changes MANO palm collision scale from the removed runtime's 0.7
  to 1.0 and changes several object CoACD/inertial assets. It is a new
  physical version, and old checkpoint rollout equivalence is not claimed.
- The specified source no longer contains `bottlewithcap` or `scissor`; the
  runtime fails explicitly for those historical names rather than substituting
  different geometry.
- The primary worktree now has no old physical asset directories. Its only
  remaining dirty files are the user's pre-existing `test.sh` and unrelated
  untracked work.

## Ruled out
- A URL-only replacement cannot work because DexStream reorganizes paths and
  removes some historical object names.
- Mapping removed names to another object would silently change geometry and
  is prohibited.
- Copied runtime meshes or unmaterialized LFS pointers cannot be fallbacks;
  both create mixed provenance or hidden setup failures.
- The live-window SIGSEGV observed after an external X11 capture or during
  Python/GLFW teardown is not evidence of a bad asset: the rendered frame is
  correct, no-capture control scenes compile and exit, and the in-process EGL
  render is stable.

## Current claim
The asset migration is complete and integrated on `dev`. The repository now
uses one pinned DexStream hand/object source, validates the complete runtime
asset closure (including binary skin dependencies), removes the old physical
sources, records asset identity in checkpoints and dataset manifests, and has
passed structural, dynamic env=1, and visual checks.

## Remaining boundary
- The historical default trajectory viewer fixture is absent on this machine;
  modern v530 env=1 and asset-only visible paths are validated.
- Automated screenshot capture should use the in-process renderer rather than
  racing an active GLFW window during interpreter shutdown.
- Remove the delivered migration/repair/evidence worktrees and branches only
  after their commits are reachable from `dev`; preserve unrelated worktrees.
