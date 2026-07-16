# Current Epistemic Model

## Phenomenon

The user is changing target training dynamics, not the normalized PPO action-space bounds. Position residual authority is reduced by 40% and the object/reference deviation reset radius is increased by 50%.

## Mechanism

With gamma 0.9, a constant unit position action and scale 0.003 has steady state 0.003/(1-0.9)=0.03 m, exactly the requested cumulative cap. The reset predicate is evaluated on object position versus the trajectory's reference object position after early-phase suppression and currently uses strict `>` comparison. Changing to 0.15 keeps exactly-threshold states alive.

## Compatibility consequence

The same policy/checkpoint executed under this mapping produces different physical targets and episode boundaries. Reward equations are unchanged, so compatibility should be represented by a separate explicit environment/control contract rather than falsely renaming only the reward equation contract. Strict resume must bind that contract; inference can remain permissive and expose the sidecar metadata.

## Current claim

Target defaults now bind normalized XYZ residuals to `(0.003, 0.003, 0.003)`, gamma `0.9`, and `+/-0.03 m`, while rotation, joint mapping, the 26D `[-1, 1]` Box, and the 100-step early phase remain unchanged. Terminal-enabled environment, train/evaluation, viewer, and recorder constructors resolve `0.15 m`; diagnostic terminal-disabled paths remain unbounded. The strict reset predicate remains `distance > threshold`.

`ENVIRONMENT_CONTRACT_ID` identifies this control/termination behavior independently of reward contracts. Native v2 sidecars save it; strict resume rejects missing or mismatched values before `agent.load`, and inference remains compatible with a native old sidecar missing only this field. Runtime, W&B, metrics, and Rerun metadata serialize the identifier and resolved mapping.

## Decisive validation

Focused configured-runtime tests cover recurrence/caps in both signs, exact and above threshold behavior with early suppression, unchanged rotation/joint/action Box behavior, terminal constructor defaults, strict checkpoint compatibility, inference legacy compatibility, W&B serialization, and Rerun metadata.
