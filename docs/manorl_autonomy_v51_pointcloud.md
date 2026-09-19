# ManoRL autonomy v5.1 point-cloud policy

Status: implemented on 2026-09-19.

The v5.1 policy keeps the current ManoRL environment, reward, PPO, 28-DoF
rate-limited action command, teacher supervision and raw-Gaussian sampling
contract. It changes only the observation ABI and actor-critic network.

## Observation ABI

The raw observation is 2205 floats:

| Block | Width | Meaning |
|---|---:|---|
| Actual state | 119 | Same physical state as v4 |
| Reference state | 109 | Same current reference and live errors as v4 |
| Future numeric | 12 | Object deltas and validity at 6/12/24 frames |
| Region contact | 160 | 16 × 10 region-bound contact features |
| Object wrench | 15 | Gravity, hand and non-hand force/torque |
| Object cloud | 192 | 64 × 3 object-local points |
| Actual hand cloud | 768 | 16 regions × 16 points × 3 |
| Reference hand cloud | 768 | Same-frame goal hand cloud |
| Action identity | 50 | Existing action one-hot |
| Object geometry | 12 | Existing object geometry encoding |

Each 10D region-contact row contains:

1. reference confidence;
2. reference valid flag;
3. measured contact-active flag;
4. `log1p` measured contact count;
5. three paired hand-to-object force components;
6. three tangential-slip components.

Forces are expressed in the live object frame and normalized by object weight.
Torques use object weight times object radius. `asinh` preserves large impulses
without the saturation introduced by the old fixed `tanh(force / 5 N)` path.

The object wrench follows the useful VoxMani v0.5 decomposition:

- gravity;
- summed hand force;
- summed hand torque about object COM;
- non-hand force;
- non-hand torque about object COM.

## Network

- Model width: 128.
- Object tokens: 16 patches, each max-pooled from 4 points.
- Current hand tokens: 16 regions, each max-pooled from 16 points.
- Goal memory: 16 reference-hand region tokens.
- Current set: 16 object + 16 hand + 1 state + 1 readout = 34 tokens.
- Contact features are projected into the matching current-hand token.
- Region and object-patch identities use learned embeddings.
- Actor and critic share only low-level point stems.
- Actor and critic have independent two-layer fusion towers.
- Each fusion layer is self-attention followed by cross-attention to the
  reference-hand memory.
- The first release keeps the flat 28D action mean head and current raw-Normal
  distribution so observation/attention effects can be isolated.

## Version selection

The public CLI remains backward compatible:

```bash
python tools/train_manorl_autonomy.py inspect \
  --policy-version v5.1-pointcloud \
  --device cpu \
  --num-envs 1 \
  --package /path/to/package \
  --identity cube2_02_2833
```

Training and evaluation use checkpoint metadata to reject v4/v5.1 ABI
mismatches. Existing v4 checkpoints are not loadable into v5.1.

## Deliberately unchanged

- reward v4 / Reward v10 hybrid implementation;
- environment and termination semantics;
- PPO/GAE implementation and hyperparameters;
- teacher anchor;
- action command, clipping and anti-windup;
- raw Gaussian likelihood contract;
- deterministic frozen evaluation at the natural first termination.
