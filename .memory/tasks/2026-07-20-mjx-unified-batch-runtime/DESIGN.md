# Unified MJX Batch Design

## Representation

`build_unified_scene_xml` compiles one fixed MuJoCo topology containing the
MANO hand and one real collision body per selected object type. Each object
keeps its authoritative URDF inertial parameters, mesh asset, collision mask,
and free joint. An environment selects its active object by placing that
object at the trajectory pose; every other object is initialized at
`(1000 + 10 * object_index, 0, 1000)` with native gravity and zero velocity.
The bounded ManoRL episode cannot bring an inactive body into the floor/hand
workspace, so no geometry or force approximation is introduced.

The model is static and the MJX `Data` is batched with the same Warp-specific
`vmap(lambda _: single_data)` pattern used by the existing homogeneous path.
`UnifiedMjxWarpPhysicalProducer` maps each world to its active body, free-joint
velocity address, and collision geom before extracting physical/contact fields.

## Exactness Evidence

For cube1 and cube2 at the same pose and zero controls, one unified Warp step
matches each corresponding homogeneous Warp model within:

| object | qpos max error | qvel max error |
| --- | ---: | ---: |
| cube1 | `1.7e-14` | `1.4e-11` |
| cube2 | `3.7e-13` | `3.0e-10` |

The first environment-level smoke also returned source-shaped `(2, 476)`
observations, finite rewards/termination flags, and preserved indexed reset.

The bounded CUDA probes used the local RTX 4060 Ti (`CUDA_VISIBLE_DEVICES=0`)
and did not launch PPO:

| objects | worlds | contact capacity | transitions/s | result |
| ---: | ---: | ---: | ---: | --- |
| 2 | 256 | `64 * worlds + 64` | `19.9k` | finite, no narrowphase overflow |
| 13 | 128 | `64 * worlds + 64` | `7.3k` | finite, no OOM |

These are kernel smoke measurements, not a matched training benchmark; the
full training decision remains gated on review and a server2 GPU3 run.
At 512 worlds with all 13 objects, the same kernel used about `2.3 GiB` of
the local 8 GiB GPU after one step. The 2048-world PPO configuration therefore
still needs a separate memory/rollout validation; this feature does not claim
that the historical `2048` budget is safe without that measurement.

## Opt-In Contract

`EnvironmentConfig.unified_object_batch` and the training CLI flag
`--unified-object-batch` are opt-in. The default single-object and legacy
heterogeneous route paths remain unchanged until a matched GPU benchmark and
review are complete. Full all-object/all-action training is explicitly
deferred until after that integration gate.

Dynamic-reset point templates use one RNG stream per global environment in the
unified path. Static checkpoint-compatible templates are directly comparable
with the legacy route; tests use that mode for exact observation alignment.
