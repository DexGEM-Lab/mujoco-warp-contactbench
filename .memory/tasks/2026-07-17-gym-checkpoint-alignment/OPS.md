# Operations Record

- Do not modify the source IsaacGym repository or source checkpoint.
- Do not overwrite existing target outputs.
- Record commands, test results, source SHA256 before/after, output paths, and any
  live-runtime blocker here as work proceeds.

## Review and validation

- Current implementation was transferred from the dirty primary `dev` worktree
  into this feature worktree before review fixes; unrelated primary files were
  not transferred.
- Converter smoke on the checkpoint currently present at the source path
  completed without changing that file; its current SHA256 is
  `2efe3a4df916fe53763345addbd1758f72315f51cb1f7d242b5c242da20e23ae`.
  This is not the earlier validated Gym-evaluation source: the retained v7
  converted checkpoint records SHA256
  `791e72fc2aebf1e2a74dc5ea60c2893888ab059fc94af899b7c8626779597fb9`.
- Review conversion artifact:
  `outputs/gym_checkpoint_alignment_20260717/MANOHand_film_dynamic_strict_review.pt`.
  Generated outputs are retained as required by repository policy.
- Gym conversion tests cover source-key actor/critic parity, dtype conversion,
  schema/provenance, and fail-closed structural cases.
- Final Gym conversion suite: 13 passed.
- Final ManoRL suite: 240 passed, 2 deselected. One deselection is a generated NPZ
  fixture absent from this worktree. The import-boundary deselection was run in
  isolation and passed (1 passed).
- `py_compile` and `git diff --check` passed. `uv lock --check` could not run
  because `uv` is not installed or available on this host PATH.
