# OPS

- 2026-07-29T23:32: Task created on feat/lance-v2-checkpoint-export from dev db4424c. User explicitly requested v2 corrected GPU cube2:02 rollout synthesis and Lance delivery.
- 2026-07-29T23:32: Standard exemplar analysis established normal-only force without 0.5 scaling, corrected actual link-frame force, and data_fps=200 for dt=0.005 as v2 requirements.
- 2026-07-30T00:08: Implemented explicit Arrow/Lance v2 schema, corrected contact decoder, deterministic checkpoint rollout exporter, root synthesize.sh, right 28D URDF-to-MANO conversion, manifest provenance, and focused tests.
- 2026-07-30T00:11: First real cube2:02 CPU transition exposed a wrong field assumption (`ResidualActionResult.processed`); direct ABI showed the field is `actions`. Corrected it, then the real MJX-Warp transition and raw contact__pos extraction succeeded.
- 2026-07-30T00:14: Focused contract/action/episode/shell tests passed (32); an additional viewer batch passed except one asset-dependent test before worktree submodules were materialized. Local Lance 0.24 output was read successfully by Lance 7.0 in an isolated compatibility probe.
