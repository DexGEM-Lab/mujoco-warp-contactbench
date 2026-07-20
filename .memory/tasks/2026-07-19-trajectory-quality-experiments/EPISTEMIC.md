# Epistemic Notes

## Confirmed from existing MuJoCo runs

- MuJoCo can converge on the current `cube1/01` task: several FiLM=true and
  FiLM=false seeds completed the 789-call deterministic evaluation.
- Success is not uniform across seeds. A 0.2N reward-threshold / 2N
  observation-threshold matrix produced only a subset of complete runs; the
  aligned 0.2N/0.2N subset improved some successful trajectories but did not
  remove variance.
- A scheduler warmup of ten updates produced 0/3 successes in the tested
  subset; it must not be promoted as the default based on the present evidence.
- Long runs have emitted occasional high-return episodes while the update mean
  and deterministic evaluation remained low. Episode telemetry is therefore not
  proof that the corresponding transitions were repeatedly used by PPO.
- High-return episodes in the diagnostic long run were concentrated in one
  shorter trajectory identity, which is evidence for identity/difficulty
  imbalance, not evidence that every source trajectory is valid.
- The current on-policy memory samples rollout transitions uniformly; completed
  episode return is recorded for telemetry and is not an automatic priority.

## Working hypotheses to test

- Rare contact transitions are diluted by the large 2,048-environment rollout;
  this is causal only if positive-transition and minibatch-hit telemetry
  confirms it.
- Some source trajectories are genuinely invalid or disproportionately hard;
  mixing them without a quality tier can suppress learning on Core data.
- Dynamic point sampling and the global CUDA Torch RNG amplify seed variance,
  but they may not be the primary cause of failed discovery.
- A curriculum can improve discovery if Core data is learnable and
  Hard-valid data is valid but poorly covered. It must be tested against a
  uniform Core+Hard-valid arm.
- If smaller matched batches outperform 2,048 environments at equal total
  transitions, update-level dilution is supported. If not, exploration or
  trajectory coverage is more likely.

## Unresolved questions

- Which identities are Core, Hard-valid, or Invalid under an RL-independent
  audit? This must be answered before a pass-rate denominator is published.
- Does the training loop pair reset observations and actions exactly as
  intended at terminal boundaries? A fixed transition trace is required.
- Does the target evaluation identity (`003`) receive positive transitions in
  the all-trajectory assignment, or are positives confined to easier identities?
- What fraction of PPO minibatches contain a positive transition, and how much
  actor-loss contribution do those transitions produce?
- Does identity-balanced assignment improve Core completion without making the
  policy overfit one anchor?
- Are point-template seeds stable enough for deterministic acceptance, or must
  evaluation report a distribution over fixed templates?

## Interpretation constraints

- Do not infer data quality from a policy's final return; that would turn the
  experiment into a circular label.
- Do not infer causal dilution from an episode maximum alone. Measure the
  transition-level sampling path.
- Do not call a single successful seed, a single easy identity, or a transient
  high-return episode a general convergence result.
- Do not call Hard-valid failures regressions in the Core acceptance metric;
  report them separately.
- Do not mix old elite trajectories into standard PPO memory without explicitly
  labeling the resulting algorithm off-policy or biased and recording the new
  contract.

## Evidence needed before production training

The next long run should be selected only after one arm demonstrates, on fresh
confirmation seeds, stable Core completion across consecutive checkpoints and
an improved positive-sample coverage diagnostic. A larger update budget alone
does not establish the cause and should not be used as the first correction.
