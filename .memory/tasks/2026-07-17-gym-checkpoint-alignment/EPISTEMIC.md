# Epistemic Record

- Initial state: task assigned from `dev` commit `81de3b7`.
- Network-forward parity was established read-only before task creation: 41
  mapped tensors, actor mu max absolute difference 0, critic value max absolute
  difference 0 on a fixed synthetic batch.
- The aligned target observation and value normalizers use rl-games epsilon
  `1e-5`, float64 running state, and the same update/forward formulas.
- Dynamic surface sampling now uses the same surface-area CDF, square-root
  barycentric transform, and global CUDA Torch RNG as the source.
- Later user direction superseded the initial compatibility-only design: the
  aligned contract is the production training default and the legacy policy
  selector is removed.
