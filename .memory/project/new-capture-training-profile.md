# New capture training profile

Canonical instructions: `docs/sept15_capture_training.md`; preset
`configs/manorl/sept15_right.env` uses right-only,120Hz,pre120/post250.

New trainer defaults: XYZ contribution2mm/cap1cm, five distal finger expected
contacts. Generic EnvironmentConfig/ResidualActionConfig remain historical;
checkpoint signature distinguishes source_mapping/five_fingertips and scales.
Five sites enable all22 finger residual axes; they refer to collision segments,
not point sensors. All contact observations remain available.

`--hand-side right` alone retains passive left models. The additional
`--drop-uncontrolled-hands` removes unselected sides during decode and binds
this choice into the package selection. Explicit MANORL_ASSET_MANIFEST matches
operator/selected-hand betas and binds packages/checkpoints to that profile.

New object IDs egg_cup/egg_ellipsoid require splitting identity from the right
into object/action/sequence. Legacy source_path suffix eligibility remains
strict. September15 v4 has132 rows after owner deletion of the low-lift bowl09.
Action04 has one primary target egg_cup;16rows encode combined cup/bowl names.
Use explicit `--target-object-overrides egg_cup:04` at compile/train; do not
rewrite source or infer target from comma ordering. This selects all132 rows.
