# v8 public name and dimension guide

Owner: `feat/v8-observation-guide`, ManoRL; base `16d0985`.

Expose the existing VoxMani 453 architecture as `v8`, retaining the old
`v5.voxmani453` spelling and serialized model/observation/checkpoint contracts.
Explain every observation dimension and the separate structured goal snapshot.
Do not alter network weights/layout, input meanings, reward or physical control.

Owned files: policy version registration, CLI/launcher, resume alias handling,
model display name, focused compatibility tests, README and input documentation.
Work locally without subagents. Validate both version spellings, existing
checkpoint compatibility, 453/547 dimensions and the generated dimension guide.
