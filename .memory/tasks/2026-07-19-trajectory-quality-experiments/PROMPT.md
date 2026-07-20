# Trajectory Quality And Contact-Discovery Experiments

## Objective

Determine why the current MuJoCo policy is seed- and trajectory-sensitive, then
raise contact-discovery reliability on the valid data distribution. MuJoCo has
already demonstrated that it can converge; this task is not an attempt to make
every raw trajectory converge and is not a requirement to reproduce every Gym
detail.

The experiment must distinguish three cases before judging a policy:

- `Core`: source data is continuous, physically plausible, and has a usable
  contact window. These trajectories form the primary training and acceptance
  set.
- `Hard-valid`: data is valid but unusually fast, short-windowed, or otherwise
  difficult. These trajectories are a challenge set and may be handled by a
  curriculum; they do not define the primary pass rate.
- `Invalid`: data has a data/asset defect (for example a discontinuity,
  teleport, malformed contact window, or severe geometry inconsistency). It is
  excluded from the convergence denominator with an explicit reason.

No quality tier may be assigned from RL return. The tier is frozen before the
training comparisons and recorded in a manifest.

## Frozen baseline

Select one previously successful MuJoCo checkpoint/run as `M0` and record its
commit, checkpoint sidecar, dataset version, trajectory manifest, and runtime
configuration. Unless an experiment explicitly changes one factor below, keep
these settings fixed:

- `cube1`, gesture `01`, `includeSuffixFiles=false`;
- dynamic point cloud and the current global CUDA Torch RNG backend;
- FiLM and the current production scheduler setting from `M0`;
- 2,048 training environments, rollout 48, minibatch 4,096, three epochs;
- the current MuJoCo reward, action, termination, and observation contract.

The contact threshold, FiLM flag, scheduler, point sampler, PPO batch, and
action scaling are separate factors. Do not change more than one in an
experiment. The known 2N physical contact setting is not reopened by this
task.

## Experiment matrix

Use screening seeds `42`, `43`, and `44` first. Run each training arm for 300
updates with identical checkpoint intervals, evaluation seeds, and W&B
settings. Promote only the most informative arms to 600 updates and then to a
five-seed confirmation run.

### E0: Quality audit and manifest (no RL)

Audit every selected `cube1/01` trajectory independently of policy return.
Check timestamps and NaN values, source slices, length and contact-window
coverage, joint/pose/velocity/acceleration discontinuities, object teleport or
table penetration, hand/object geometry, and zero-residual/reference replay
error. Review outliers in Rerun. Freeze a manifest containing trajectory UUID,
identity, quality tier, exclusion reason, length, contact-window length, and
difficulty indicators.

### E1: Anchor isolation

Train and evaluate on one `Core` anchor at a time. Start with the existing
evaluation identity (`cube1_01_003`) and one easier `Core` anchor (the prior
high-return `093` is a candidate only after the audit). This measures whether a
valid trajectory can be learned when other identities cannot dilute its signal.

### E2: Core versus all

Compare the current all-available assignment (`E2-all`) with a `Core`-only,
identity-balanced assignment (`E2-core`). Keep total environment slots and
PPO settings equal. The comparison tests whether invalid/hard data contaminates
the update rather than whether the network can fit a single example.

### E3: Difficulty curriculum

Compare uniform `Core + Hard-valid` training with a staged curriculum: begin
with Core, introduce Hard-valid identities only after Core completion/contact
coverage passes a pre-registered threshold, and eventually restore the target
mixture. Evaluation always starts from the full trajectory, not a curriculum
state.

### E4: Matched-batch dilution test

Hold total environment transitions and optimizer minibatch updates constant
while changing the number of simultaneous environments:

| environments | updates | transitions | purpose |
| ---: | ---: | ---: | --- |
| 2,048 | 600 | 58,982,400 | current baseline |
| 1,024 | 1,200 | 58,982,400 | half-size update |
| 512 | 2,400 | 58,982,400 | discovery-sensitive update |

This isolates whether aggregating 98,304 transitions per update dilutes rare
contact transitions. If a smaller batch wins, retain throughput separately
from discovery reliability.

### E5: Rollout-local on-policy stratification

Only if telemetry confirms dilution, compare current-rollout uniform sampling
with actor-batch stratification at 10% and 25% positive samples. Positive means
contact/progress or within-identity high-advantage rank, not global raw episode
return. Use only transitions collected by the current policy in the current
rollout; never replay old actions across updates. Critic and normalizer remain
on the full uniform rollout. If sampling probabilities change, record and use
importance weights for the strict PPO arm; otherwise label the arm as an
explicit biased discovery experiment.

## Required telemetry

For every update and identity, record:

- positive/contact transition fraction and counts;
- advantage quantiles and positive-advantage fraction;
- number and fraction of minibatches containing positive transitions;
- actor-loss contribution from positive versus non-positive samples;
- completed episodes, deviation resets, and first-positive update;
- episode completion and max-deviation metrics by identity;
- point-template/RNG seed used for evaluation.

The existing episode-return JSONL remains useful, but an episode maximum alone
is not evidence that PPO used the episode. The dilution diagnostic is:

```text
p = positive_transitions / (num_envs * rollout_length)
expected_positive_per_minibatch = p * minibatch_size
zero_positive_minibatch_ratio = zero_positive_minibatches / minibatch_count
```

## Acceptance and decision rules

The primary metric is completion on the pre-declared `Core` set, not the best
single return. A screening arm is informative when it has three seeds and
identity-level telemetry; a promoted arm must use five confirmation seeds.
Require a successful policy to complete the Core trajectory for three
consecutive checkpoints and report median/max deviation. Report Hard-valid
completion separately; it has no mandatory pass rate. Invalid trajectories do
not enter the denominator but their exclusion reasons remain visible.

Interpretation rules:

- E1 succeeds while E2-all fails: identity mixture or data contamination is
  dominant.
- E2-core succeeds while E2-all fails: invalid/Hard-valid data is diluting the
  useful distribution.
- E3 beats uniform Core+Hard-valid: curriculum is justified.
- E4 smaller batches beat 2,048 at matched transitions: update-level dilution
  is causal.
- E5 improves Core positive coverage and completion without destabilizing KL:
  rollout-local stratification is a candidate discovery mode.
- None of these helps: inspect transition/reset semantics, reward signal, and
  observation/action alignment before increasing the training budget.

Do not select a checkpoint by one high-return episode or by success on only the
easy anchor. A 2,000/8,000-update production run starts only after an arm wins
the five-seed confirmation criteria.

## Scope boundary

This task documents and runs experiments only. It does not make a permanent
change to PPO sampling, reward definitions, data filtering, or the production
default until an experiment has passed the above comparison and review.
