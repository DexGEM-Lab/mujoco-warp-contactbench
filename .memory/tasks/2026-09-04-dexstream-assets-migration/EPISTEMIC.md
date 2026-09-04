# Epistemic Model

## Phenomenon
ManoRL previously mixed the `assets/all_assets` object submodule,
`assets/mano_hand_s02` hand submodule, and copied hand/cube files under
`sim/manorl/runtime_assets`. The target is one reproducible physical source for
both hand and objects, with setup that works from a fresh skip-smudge checkout.

## Supported mechanism
- `assets/dexstream_digital_assets` is the sole physical source, pinned to
  `f98da997f316c8a6b4bc2931cabed19e831ef163` from
  `git@github.com:DexGEM-Lab/dexstream_digital-assets.git`.
- ManoRL consumes `hand/mano/sunke/{right,left}`. Both bundles satisfy the
  28-DoF `cmc3_mcp2_28dof`, z-up, 16-link collision/visual contract; pose axes
  are read from the pinned URDF.
- ManoRL discovers 28 same-name rigid `objects/DexGEM` URDF bundles and reads
  all collision pieces/scales from their declarations. Camera source directory
  case is normalized only at the adapter boundary.
- `dexstream_manifest.json` pins required Git/LFS files by source commit,
  SHA-256, and size. The generator now includes every file named by the MANO
  metadata skin bundle, including `.skn` and `.npz`; setup validates both sides
  and all objects without modifying GitGuard hooks.
- Expected-contact aliases remain ManoRL-owned task metadata because the
  physical source does not provide them. Strict checkpoint signatures include
  repository, source commit, and manifest SHA; explicit weight transfer is the
  cross-version boundary.

## Evidence
- 28 DexGEM URDF bundles have one rigid link, one visual mesh, finite inertial
  data, and resolvable mesh references.
- Homogeneous compilation passed for all objects; right/left/bimanual unified
  compilation passed with visual skin enabled.
- Focused migration tests passed 159 tests (7 skipped); final repair-branch
  fresh-checkout asset/scene/submodule tests passed 27/27.
- A real local `DISPLAY=:1` screenshot with one world showed continuous MANO
  skin and a valid banana visual mesh. A v530 Lance→CPU MJX-Warp env=1 smoke
  returned 480D observations and nonzero contacts.
- Fresh integration setup initially failed with `insufficient data in SKN`
  because `.skn` was omitted from the manifest. After the generator fix, the
  same skip-smudge setup materialized right/left `.skn` (45184 bytes) and
  `.npz` (100378 bytes), and the generic scene compiled successfully. This
  closes the previously hidden machine-cache dependency.
- The current source changes MANO palm collision scale from the removed
  runtime's 0.7 to 1.0 and changes several object CoACD/inertial assets; it is
  a new physical version.

## Ruled out
- A URL-only replacement cannot work because DexStream reorganizes paths and
  removes some historical object names.
- Mapping removed `bottlewithcap` or `scissor` to another object would silently
  change geometry and is prohibited.
- Copied runtime meshes or an unmaterialized LFS pointer cannot be a fallback;
  both create mixed provenance or hidden setup failures.
- The live-window SIGSEGV observed when external X11 `ImageGrab` races GLFW
  teardown is a display-capture interaction; the same scene renders and exits
  cleanly without that race.

## Current claim
The repair branch completes the asset migration and the fresh-checkout
installation contract: one physical source, complete MANO skin dependencies,
explicit task metadata, no old-source fallback, generic scene adaptation, and
checkpoint provenance binding. The physical source change means legacy
checkpoint rollout equivalence remains unclaimed.

## Remaining boundary
- The historical default trajectory viewer fixture is absent on this machine;
  modern v530 env=1 and asset-only visible paths are validated.
- The external X11 capture/GLFW teardown interaction should be avoided in
  automated checks; use the in-process renderer output for a clean exit code.
- Merge this repair branch into `dev`, then remove the temporary repair branch
  and worktree while preserving unrelated primary-worktree user files.
