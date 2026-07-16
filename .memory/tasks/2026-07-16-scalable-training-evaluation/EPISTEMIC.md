# Current Epistemic Model

## Phenomenon

A single 4096-world ManoRL runtime trains successfully on a 24 GiB RTX 4090, but post-training evaluation fails when the process constructs a second 4096-world physical runtime before releasing the first.

## Live mechanism

The failure is deterministic across a contended GPU2 and empty GPU3, at the same fresh-evaluation Warp convex narrowphase allocation after training/checkpoint completion. Training/evaluation vector count is currently coupled through `budget.num_envs`. The executable checkpoint boundary does not require evaluation to use the training vector count; policy and normalizer shapes are environment-count independent. A fixed bounded evaluation set can preserve comparability if zero, untrained, and trained evaluations all use the same assignments and if training starts from the exact native state loaded for the untrained evaluation.

## Current claim

4096-world training is feasible at about 10.5k environment transitions/s. Evaluation now resolves independently to `min(training_num_envs, 128)` unless explicitly overridden, so training constructs one 4096-world runtime and each fresh evaluation constructs 128 worlds. A temporary, output-owned native checkpoint establishes the initial policy/value/normalizer/optimizer boundary: the bounded untrained runtime and the training runtime load the same checkpoint. The final bounded runtime loads the final native checkpoint. Focused mocked construction/load tests support this claim; a coordinator-owned 4096x1 GPU rerun remains the decisive physical validation.

## Next decisive question

Does the coordinator's GPU3 4096x1 acceptance rerun complete the final fresh 128-world evaluation under the actual Warp allocator?
