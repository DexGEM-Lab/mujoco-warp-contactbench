## Objective
Expose existing EnvironmentConfig constraint capacity to trainer CLI for dense source assets. Server1 preparation probe found njmax512 overflow at553; njmax2048 passed64steps and reset for all8 target categories.
## Scope
TrainingBudget,train/eval construction,CLI,train.sh,Sept15 profile,docs and focused tests. Default512 for backward compatibility; Sept15 override2048. No physics changes and no formal training.
## Constraints
Worker feat/capture-constraint-capacity created by primary launcher from integrated dev. Do not alter other jobs or branches. Validate focused CLI/budget plus runtime probe evidence.
