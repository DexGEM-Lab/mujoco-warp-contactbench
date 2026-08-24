# Current model

## Phenomenon
The no-prefix/no-retreat exporter previously saved every numerically valid rollout that reached the reference horizon. Final object orientation and sustained grasp contact were measured only after production, so they could not prevent low-quality rows from entering Lance.

## Implemented mechanism
No-prefix/no-retreat acceptance is one post-terminal atomic AND over three artifact-aligned observations:
1. terminal success with reason code 1;
2. final simulated/reference object rotations round-tripped through the persisted float32 rotvec representation, converted independently to intrinsic XYZ Euler degrees, then compared by per-axis shortest wrapped absolute difference and arithmetic mean <=35 degrees;
3. at least 101 distinct persisted frames containing `hand_name=right`, the target object name, and any float32-persisted solved normal-force vector with Euclidean norm strictly >0.2 N.

A failed candidate never reaches row construction or a Lance writer. The manifest records every attempt, all failed predicates, XYZ component errors, mean rotation error, and counted contact frames. The validator independently recomputes saved-row measurements, binds every saved row to one unique accepted-attempt key, and rejects diagnostic contradictions.

## Scope boundary
The gate is enabled only when both approach prefix and retreat suffix are disabled. Existing accepted-parent augmentation and historical Lance rows remain unchanged. The seed-plan runner writes only prefix+retreat augmentation and therefore stays under its separately versioned contract.

## Evidence and claim
Local focused tests establish exact threshold behavior (35 passes, >35 fails; 100 frames fail, 101 pass), target contact scoping, persistence precision, AND semantics, no-write rejection, bounded zero-yield diagnostics, and validator/manifest binding.

A real Server1 GPU experiment discriminated save/reject behavior under the canonical checkpoint and pre180 package:
- `banana_01_052` reached reason 1, measured 11.1625° and 306 contact frames, and produced one row. Independent validator recomputed the same values.
- `banana_01_077` reached reason 1 and had 307 contact frames but measured 100.8115°, so the exporter produced zero rows and recorded the rotation failure.

The implementation therefore enforces the requested atomic gate at the actual writer boundary and preserves enough evidence to audit every decision. The smoke is diagnostic-only and historical Lance data remain immutable.
