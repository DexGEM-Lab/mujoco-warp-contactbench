# Epistemic model

## Phenomenon
A synthetic Lance row stores `hands[0].urdf_dof_target` as the post-`command_target` controller target used by the MJX-Warp environment. Reapplying that sequence with two physics substeps per 5 ms frame reproduces the recorded object-level behavior closely enough for diagnostic replay and visualization.

## Implemented mechanism
The repository now has a versioned NPZ+JSON package contract and a `TargetDofReplay` adapter. The loader validates metadata, shapes, finite values, timing, normalized quaternions, and payload SHA256 before any simulation allocation. The adapter constructs the existing homogeneous `MujocoManoEnvironment`, initializes the recorded frame-0 physical state, writes each target vector directly to `data.ctrl`, and advances only the existing MJX-Warp kernels. GUI rendering mirrors that state into the existing native visual-model ABI path; it never runs a second native simulation.

## Evidence
Seven pure contract tests pass. The formal CLI completed an 8-transition CPU diagnostic with an explicit CCD override and a 519-transition GPU replay on a local RTX 4060 Ti. The full GPU result reproduced the prior smoke metrics: max object position error about 3.84 mm, max rotation error about 0.0214 rad, and maximum lift within 0.33 micrometers of the recorded value.

## Boundaries
Exact qpos/contact parity is not expected because contact solving and floating-point execution can diverge; object behavior is the primary smoke criterion. CPU mode rejects packages with explicit GPU CCD scratch unless the user passes an override, and reports that physics change. The package exporter belongs on a healthy machine with Lance access. Server2's native Lance/PyArrow path remains disqualified; a successful package replay only establishes the isolated physics/rendering path.
