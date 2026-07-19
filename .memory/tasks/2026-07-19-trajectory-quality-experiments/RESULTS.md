# M0 All-Trajectory Screening Results

## Scope

This document records the four-GPU M0 screening batch run on Server2 on
2026-07-19. It is an all-available-trajectory seed screen, not E1 anchor
isolation or the E2 Core-versus-all comparison. The runtime did not yet expose
a quality-manifest trajectory selector, so all ten `cube1/01` identities were
assigned round-robin across the training environments. Deterministic evaluation
used `cube1_01_003` in environment zero.

The frozen run contract was:

- 2,048 training environments, rollout 48, 300 updates, and 29,491,200
  transitions per seed;
- FiLM enabled, dynamic point clouds, global CUDA Torch RNG, residual actions,
  terminal resets, and scheduler warmup zero;
- 2N observation and reward contact thresholds;
- W&B online in project `one_policy`, group
  `trajectory-quality-20260719`;
- Server2 source mirror
  `/mnt/user-home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/contact-discovery-experiments-20260719`.

At launch, `tools/train_manorl_cube1.py` and
`sim/manorl/rl_games_ppo.py` in the mirror matched the local
`feat/contact-discovery-experiments` snapshot at `6af8846`. All four runs
reached 300 updates without OOM, NaN, traceback, or an early process exit.

## Deterministic Evaluation

The operational completion rule for this evaluator is `calls == 789` with
`max_object_target_distance < 0.1 m`. The historical
`trained.completed_horizon` field is false even for the two 789-call runs, so
it is not used as the completion decision here. Likewise,
`acceptance.accepted` only compares the trained result with the zero-reference
result; it is true for all four runs and is not a convergence criterion.

| seed | W&B ID | elapsed (s) | return | calls | contact mean | max object error (m) | result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | `0m61d18r` | 2,113.53 | 815.63 | 789 | 0.14753 | 0.06766 | completed |
| 43 | `en8tgmg7` | 1,778.98 | 71.22 | 289 | 0.00000 | 0.10056 | deviation reset |
| 44 | `g69v8skx` | 2,072.35 | 633.64 | 789 | 0.14195 | 0.08161 | completed |
| 45 | `2ebz8639` | 2,008.56 | 79.64 | 289 | 0.01107 | 0.10058 | deviation reset |

Thus the identical 300-update configuration completed the fixed `003`
evaluation for 2/4 seeds. This is evidence that the current MuJoCo task is
learnable and that contact discovery is not yet robust across seeds.

## Training Episode Evidence

The following statistics are reconstructed from each
`*.episodes.jsonl`. Return thresholds are diagnostic only; they are not a
quality label or an acceptance rule.

| seed | episodes | mean return | max return | return >=175 | return >=300 | first >=175 update | identity coverage >=175 | tail-50 reward | tail-50 contact | tail-50 resets |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 61,673 | 304.50 | 872.03 | 22,991 (37.28%) | 21,361 (34.64%) | 52 | 10/10 | 0.9599 | 0.13933 | 128.76 |
| 43 | 100,552 | 71.53 | 75.77 | 0 | 0 | - | 0/10 | 0.2465 | 0.00000 | 335.86 |
| 44 | 87,361 | 107.66 | 675.58 | 6,176 (7.07%) | 5,805 (6.64%) | 77 | 7/10 | 0.4966 | 0.10536 | 194.44 |
| 45 | 100,407 | 77.33 | 89.73 | 0 | 0 | - | 0/10 | 0.2773 | 0.01222 | 336.08 |

The successful seed 42 developed broad high-return coverage. Seed 44 developed
contact later and covered seven identities. Its missing identities were the
three shortest-window examples:

| identity | source length | movement/contact window | seed 42 >=175 (first update) | seed 44 >=175 (first update) |
| --- | ---: | ---: | ---: | ---: |
| `003` | 790 | 290 | 2,624 (58) | 1,239 (77) |
| `009` | 792 | 292 | 2,805 (63) | 815 (224) |
| `011` | 774 | 274 | 2,366 (108) | 1,014 (196) |
| `012` | 815 | 315 | 2,538 (52) | 591 (234) |
| `014` | 797 | 297 | 2,713 (52) | 916 (153) |
| `025` | 771 | 271 | 2,740 (74) | 824 (221) |
| `042` | 780 | 280 | 1,976 (68) | 777 (219) |
| `092` | 739 | 239 | 2,545 (98) | 0 (-) |
| `093` | 726 | 226 | 2,015 (133) | 0 (-) |
| `094` | 743 | 243 | 669 (230) | 0 (-) |

The shorter windows make `092`, `093`, and `094` candidates for a
Hard-valid audit, but this table cannot assign that tier. Seed 42 learned all
three, including 2,545 high-return `092` episodes. Their failure under seed 44
therefore does not establish a source or asset defect.

The spread across environments is also informative. Seed 44's first >=175
episode was one `003` environment (`env_id=1920`) at update 77. By update 300,
all 205 assigned environments for each of its seven learned identities had
produced at least one >=175 episode. For seed 42, every assigned environment
across all ten identities had done so: 205 environments for each of `003`
through `092`, and 204 for each of `093` and `094`. Seeds 43 and 45 had no
high-return environment. Once a seed found a useful contact path, current
on-policy PPO could therefore spread it broadly; the leading bottleneck is the
initial discovery path, while dilution remains an unresolved contributor.

## Interpretation

The main observation is a contact-discovery split, not a uniform slow-learning
curve. Seeds 42 and 44 developed sustained contact and completed deterministic
evaluation. Seed 43 produced effectively no contact, while seed 45 produced
weak intermittent contact without any episode reaching 175. Their tail exact
KL means remained close (`0.0200` to `0.0208`) and their policy standard
deviations remained close (`0.373` to `0.377`). This does not look like a
simple stopped optimizer or learning-rate-floor failure.

The preliminary RL-independent audit found finite, continuous inputs for all
ten identities and conservative zero-residual/reference replay checks passed
for all ten after respecting each trajectory's actual length. No current
evidence justifies an `Invalid` label. Formal `Core` versus `Hard-valid`
classification still requires the frozen E0 manifest and visual review.

This batch also does not prove PPO batch dilution. Although the sidecars say
`resolved_capture_transition_diagnostics=true`, the saved update records do
not contain per-identity positive-transition fractions, advantage quantiles,
positive-minibatch hit rates, or actor-loss contributions. Aggregate contact
and episode-return data are insufficient to make dilution causal or to exclude
it completely.

## Decision

1. Preserve seeds 42 and 44 as successful reference artifacts, but do not
   promote a production default from a two-seed result.
2. Do not extend seeds 43 and 45 merely to rescue them. Their complete
   300-update traces are useful failed-discovery controls.
3. Finish E0 and add a manifest-backed identity selector before running E1 and
   E2. Compare anchor-only, Core-only identity-balanced, and all-trajectory arms
   on the same seeds.
4. Ensure transition-level positive-sample telemetry is actually persisted
   before E4 or E5. Only that evidence can justify matched-batch or stratified
   PPO changes.
5. Do not classify any trajectory from these returns. `Invalid` remains an
   independent data/asset diagnosis with an explicit reason.

No PPO sampling, reward, data filtering, or production configuration was
changed by this screening result.

## Artifacts

Local evidence bundle:

```text
/home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/trajectory-quality-20260719/
  m0_all_seed{42,43,44,45}_2048env_300u_20260719.json
  m0_all_seed{42,43,44,45}_2048env_300u_20260719.episodes.jsonl
  m0_all_seed{42,43,44,45}_2048env_300u_20260719.eval.npz
  m0_all_seed{42,43,44,45}_2048env_300u_20260719.train.log
```

Remote authoritative output root:

```text
/mnt/user-home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/contact-discovery-experiments-20260719/outputs/manorl/trajectory-quality-20260719/
```

W&B runs:

- seed 42: `https://wandb.ai/sunjay45711-dexerto/one_policy/runs/0m61d18r`
- seed 43: `https://wandb.ai/sunjay45711-dexerto/one_policy/runs/en8tgmg7`
- seed 44: `https://wandb.ai/sunjay45711-dexerto/one_policy/runs/g69v8skx`
- seed 45: `https://wandb.ai/sunjay45711-dexerto/one_policy/runs/2ebz8639`

Each training log confirms online W&B synchronization and artifact upload at
normal completion.
