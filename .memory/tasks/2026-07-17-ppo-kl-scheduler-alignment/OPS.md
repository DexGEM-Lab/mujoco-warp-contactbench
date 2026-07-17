# Operations Evidence

- 2026-07-17: Created `feat/ppo-kl-scheduler-alignment` from `dev` commit
  `8341dba` using `scripts/start_pi_task.sh ... --no-launch`. The separate
  multi-object worktree remains preserved and unchanged by this task.
- Source inspection used the validated run's resolved config and installed
  rl-games implementation. No training process was restarted during diagnosis.
- Implemented `RlGamesPPO` and `RlGamesAdaptiveLR`. Rollout memory stores policy
  mean/std; update uses the source `policy_kl` formula, updates the reference
  distribution for every processed minibatch, schedules LR in legacy
  per-minibatch order, and never applies skrl's approximate-KL early stop.
- The default source minibatch is 4096. CLI omission resolves to the largest
  divisor shared by 4096 and the active 48-step rollout batch, so 2048 envs use
  4096 while 64 envs use 1024. W&B is enabled by default and may be disabled
  explicitly with `--wandb false`.
- Local validation: focused PPO/runtime/trainer/W&B tests passed 58/58. ManoRL
  regression passed 244 tests with two known exclusions: one import-isolation
  test passed separately in a fresh process, and one historical external NPZ
  fixture is absent from the worktree. `py_compile`, `git diff --check`, and the
  isolated import test passed.
- Remote mirror:
  `/home/jay/dexrobot/FromSSH/manoRL_mujoco-benchmarks/feat-ppo-kl-scheduler-alignment-8341dba`.
  The three runtime file hashes matched the local feature; remote focused tests
  passed 11/11.
- Server2 acceptance used physical GPU 2 after an idle check, `DISPLAY=:1`,
  2048 training envs, 128 evaluation envs, rollout 48, minibatch 4096, 3 epochs,
  50 updates, and W&B enabled. Output prefix:
  `outputs/manorl/server2_cube1_01_2048env_50u_ppo_kl_aligned_20260717_215053`.
  W&B run: `sunjay45711-dexerto/one_policy/1zh851zk`.
- The run completed 4,915,200 transitions in 315.8 training seconds at about
  15,562 transitions/s. Update 6 exact KL mean/max was `4.88195/15.46646`; LR
  moved from `3e-4` to `1e-6` while all 72 minibatches completed. Update 50
  exact KL mean/max was `0.0189645/0.0392810`, LR recovered to
  `3.8443359375e-5`, contact was `0.00630`, and completed episode return mean
  was `78.9275`.
- Fresh checkpoint evaluation: zero baseline `283 calls / 67.1179 return /
  0.000425 contact`; trained `283 calls / 76.9706 return / 0.009552 contact`.
  The bounded acceptance predicate passed, but the policy did not complete the
  trajectory and this is not a convergence claim. The process and W&B upload
  exited naturally; GPU 2 returned to 18 MiB.
