# Epistemic Record

## Final claim

The ManoRL-style DexHandRL structural refactor is accepted as behavior-neutral at candidate HEAD `6718da5`. The generated scene, compiled model signature, and every frozen reset-plus-50-step array remained bit-exact to the clean pre-refactor baseline. The refactor therefore changed ownership, validation, compatibility resolution, and viewer UX without changing the frozen `legacy_port_v1` physical behavior (OPS: Stage 1 baseline and final evidence entries).

Migration status remains **candidate; native behavioral parity pending**. The native 461D rl-games checkpoint limitation is unchanged: its network mapping is numerically faithful, but MuJoCo and Gym physical state produce different observation feedback and state-dependent 22D-to-26D routed targets. The structural refactor does not claim that the native checkpoint now grasps, lifts, and places successfully.

## Parameter provenance and ownership

`DexHandPhysicsConfig` is the sole production physical source. Its frozen `DexHandServoConfig` and `DexHandMaterialConfig` own the existing DexHand values: control/physics periods `0.01/0.001 s`, Euler/Newton with 80 iterations, base-position `5000/500/500`, base-rotation `10000/125/100`, finger `15/0/50`, hand/object friction `0.5/0.01/0.001`, and ground friction `0.5/0.5/0`. These values have `legacy_port_v1` provenance because they define the accepted port and its baselines. Asset geometry/inertials, trajectory identity, and runtime allocation retain their separate asset-, trajectory-, and task-authoritative provenance in the serialized `physics_contract` record.

The refactor copied current `dev` ManoRL's architectural pattern—immutable records, one resolver, contract-driven, byte-preserving, fail-closed textual scene transformation, compiled validation, explicit runtime allocation, and a passive viewer—not ManoRL's numeric physics parameters. Direct historical gain transfers across PhysX and MuJoCo already failed, so shared organization is evidence-backed while shared numeric values are not.

Compatibility CLI/API fields are boundary aliases only. `resolve_physics_config` normalizes them once into the immutable profile and rejects a canonical profile combined with legacy physical overrides. Production environments and tools consume the resolved object, and runtime/trace metadata persists its complete `physics_contract` rather than reconstructing provenance from scattered arguments.

## Compiled invariants

Scene construction validates source asset, name, and transformation assumptions before compilation; provenance is serialized metadata. Compiled validation enforces timestep, Euler integrator, Newton solver and 80 iterations, joint and actuator dimensions/order, gains and force limits, friction/contact settings, reset-critical names, and model capacities. These checks prevent a structurally valid XML from silently violating the runtime ABI.

Model and runtime capacities remain intentionally distinct. Compiled MuJoCo scene capacity is `njmax=2048`, `nconmax=8192`; MJX-Warp runtime allocation is `njmax=8192`, `naconmax=8192`. Runtime capacity is not serialized into the scene, and model `nconmax` is not treated as runtime `naconmax`.

## Production viewer

The supported visible path is `sim.dexhandrl.view_environment` with one environment, no checkpoint, `--no-use-residual`, and `--no-headless`. It supplies zero 22D actions to the production `DexHandRLMJXEnv`; the environment remains the only owner of physics stepping. The native passive viewer mirrors completed `qpos`, `qvel`, `ctrl`, and simulation time, runs forward kinematics for display, and synchronizes the window without stepping the model.

The initial exit-139 defect was a teardown race: MuJoCo's GLX viewer thread could still own simulator state when GLFW termination ran. Waiting for the passive handle's `_sim` weak reference to clear resolves that mechanism; the corrected natural-close run exits zero, and the visible screenshot confirms the actual production window (OPS: final viewer and teardown entries).

## Residual uncertainty and scope boundary

Replay, skrl training/evaluation, and the standalone production viewer still have distinct task loops. Training/evaluation additionally use `SyncVectorEnv`; this scope did not replace those loops with one batched rollout producer. The delivered centralization is narrower and explicit: all production paths resolve the same immutable physical contract, record the same metadata contract, validate the same compiled invariants, and share the passive-viewer semantics where rendering is supported. A future batched unified producer would be a separate runtime architecture change and is not implied by Stage 1 acceptance.
