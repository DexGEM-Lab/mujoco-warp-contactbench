# Epistemic Notes

## Confirmed

- The stopped aligned MuJoCo run used initial LR `3e-4` but reached the `1e-6`
  floor by update 11 and then trained roughly 1,090 more updates at that floor.
- Contact reward collapsed to approximately zero while episode return remained
  near the zero-action baseline.
- skrl currently uses sampled action log-ratio approximate KL, stops a
  mini-epoch above `kl_threshold`, and schedules once per mini-epoch.
- The source rl-games continuous PPO stores old mean/std, computes exact Gaussian
  KL after each optimizer step, and with default legacy scheduling adjusts LR
  after every minibatch without KL early stopping.
- The source critic objective uses the elementwise maximum of unclipped and
  clipped squared value errors, then applies the effective coefficient
  `0.5 * critic_coef = 2.0` for the resolved `critic_coef=4` configuration.
- The source continuous-policy bounds loss starts outside the soft bound `1.1`.
- Source PPO updates from the first completed rollout; delaying skrl updates to
  the environment contact-start frame is not part of the source contract.
- The validated Gym run's resolved minibatch size is 4096, not the generic YAML
  default 1024 and not the stopped MuJoCo run's 2048.

## Open until measured

- Exact legacy scheduling no longer leaves LR permanently pinned at the floor.
  The 50-update Server2 run reached `1e-6` during its first optimizer update but
  recovered to `3.8443359375e-5` by update 50 while completing all 72 configured
  minibatches per update.
- Fifty updates are not convergence evidence. Deterministic evaluation improved
  from the zero baseline return `67.1179` to `76.9706`, but both reset at call
  283 and the trained contact mean was only `0.00955`.
- The first optimizer update still has very large exact KL (mean `4.88195`, max
  `15.46646`). The critic clipping/weight, source bounds loss, and first-rollout
  start differences have now been aligned. The raw sampled-action boundary and
  contiguous Gym versus shuffled skrl minibatch order remain plausible
  convergence-significant differences.
