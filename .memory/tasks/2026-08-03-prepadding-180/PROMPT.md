## Objective
Run ManoRL in selectable 100 Hz and 120 Hz modes where source/reference and model/policy/control are uniformly coupled at the selected rate, change default source pre-padding from 100 to 180 frames, and warm-start the new timing contract from the latest durable legacy checkpoint rather than training model weights from scratch.

## Success criteria
- In 100 Hz mode, reference and policy/control run at 100 Hz while physics runs at 400 Hz with four equal substeps.
- In 120 Hz mode, reference and policy/control run at 120 Hz while physics runs at 480 Hz with four equal substeps.
- Reference FPS and control FPS remain distinct explicit metadata even though selected modes require them to match; no 3/3/4 timing jitter is permitted.
- New trajectory selections default to 180 source frames pre-padding; post-padding remains 250.
- Every step-count parameter affected by the control-rate change is either converted by physical time or explicitly retained with a justified discrete-time meaning.
- Warm start transfers compatible learned models/normalizers from the latest durable checkpoint while resetting optimizer, scheduler, memory, and progress; metadata records prior conceptual updates.
- Training/inference/checkpoint ABI and documentation state the new clocks and reject incompatible strict resume.
- Before server1 training starts, Unison is confirmed active and server1's synchronized `dev` commit exactly matches the locally integrated commit.

## Evidence and source checkpoint
- Latest durable source is `checkpoint-005900.pt`, byte-identical to `last.pt`, from the killed 8192-env resume run.
- It records 5900 updates after an upstream update-1200 checkpoint: conceptual durable total 7100. Log progress to 5985 was not checkpointed.
- Source contract is legacy v6, pre-padding 100, and has no reference FPS field; it consumed one source frame per 5 ms control step.

## Constraints
- Work only in `feat/prepadding-180`; preserve primary `dev` untracked files.
- Post-padding remains 250.
- No training process is currently running; do not launch until both selected clock modes and checkpoint migration are validated and integrated.
- Unison synchronization is a hard launch precondition; a running process alone is insufficient without commit equality on server1.
- No subagents; the user did not authorize delegation.
