# Current model

The active correction is restricted to the v4 reference cache, physical state
and raw-observation semantics. The prior v4 cache was physically invalid: it
sampled 16 compiled mesh hand geoms as `geom_size` boxes, treated mesh AABB
size as cube shape, and left reference angular velocity at zero. It also fed
normalized q into command/reward code. These failures could make an internally
finite observation describe geometry and actuator state that the simulator does
not have.

The corrected source compiles static model metadata through
`compile_model_metadata_only`, which does not call legacy native static FK.
Warp FK creates the cache. Every hand region uses its compiled
`mesh_vert/mesh_face` triangles transformed once by static geom pose into body
coordinates. Cube2 source collision triangles are already body-local; applying
the object geom pose would be a double transform. JAX closest-triangle plus
convex halfspaces produces signed reference gaps and returns a surface point
for an interior sample. Geometry is immutable T-only cache content, with
128 deterministic area-spread samples per hand region and 64 object surface
samples. Cache raw/feasible q and object origin receive the same support shift.

Raw SI q is distinct from normalized q. Commands and finger reward consume
raw q. qdot divides by source `dofRate`; servo error/envelope use source
`antiwindupError`. All values resolve from MANOHandAutonomousV1 constants.
The raw ABI has corrected .1/.3/.01/.1m/s scaling and current-reference frame
rules. Actual/reference bottom fields use collision vertices relative to cached
table height, and falling means bottom < table−.05. Reference velocities use
7-frame/poly2 filtered derivatives; angular velocity is quaternion-derived.

Evidence [OPS 2026-09-10T01:00:00Z]: 11 focused checks pass, including a
monkeypatch proving the v4 cache does not invoke legacy native static FK,
compiled-mesh transform/surface spread, skewed convex signed geometry,
reference anchor/joint identity, scaling/bottom fields and prior contact/reward
checks. The real pinned cube2_02_2833 N=1 CPU Warp reset and step gives finite
raw `(1,957)`, reward `2.2570746`, valid=true, and nonzero cache angular
velocity. Training remains stopped; this is source correctness evidence, not
learning evidence.

Next boundary: trainable PointNet/raw→829 model wiring, checkpoint encoder hash,
batching, general cone/helper parity, actual contact slip and force integration
are explicitly deferred. They must not be inferred as complete from this
cache/state slice.
