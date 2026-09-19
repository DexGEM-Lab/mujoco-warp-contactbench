# Cube2 autonomous inference checkpoint

`policy.pt` contains the released learned weights for `cube2_02_2833` after
67,108,864 environment transitions (512 PPO updates). It preserves all 29 model
tensors exactly, including PointNet, the actor, log standard deviation and the
separate critic. Raw observations are 957-D; actions are 28-D.

## Run

Follow the repository [environment setup](../../README.md#local-uv-environment)
and [asset setup](../../README.md#clone--assets). The asset submodule must be at
`f98da997f316c8a6b4bc2931cabed19e831ef163`.

The data package is **not bundled**. Set `PACKAGE` to an existing compatible
pinned `cube2_02_v295_f120_pre180_post180` package directory. Its required digests
are:

- Package: `6ac29825d444069071f4803718bff45345fab56131b4d9bf4b6e7ef3d8178e90`
- Manifest: `e826de23d4586611001230d340eaf0c68752b09988eed7928bf8884f056d8706`
- Catalog: `2f1df18fbb692bbc67a02bd47c684468291fae3d13edbb9412de0b4e70c346fb`

From the repository root, with the Python environment activated:

```bash
: "${PACKAGE:?Set PACKAGE to the compatible package directory}"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tools/train_manorl_autonomy.py evaluate \
  --device cpu --num-envs 1 --no-persistentworkspace \
  --package "$PACKAGE" --identity cube2_02_2833 \
  --checkpoint checkpoints/manorl_autonomy_v4_cube2/policy.pt \
  --trace outputs/manorl/contact_conditioned_autonomy/bundled-cube2-eval.json
```

This runs a full-start normal frozen episode until natural termination and writes
JSON diagnostics plus an NPZ trajectory. It does not generate a video. Evaluation
uses clipped deterministic policy means. References condition observations; all
executed controls come from the policy, with no hidden anchor teacher.

## Provenance and limits

Training source: `2eaae8ce904a1f6b78c3814e35faeca25dee8912`, run
`v4-gpu2-b4096-anchor-20260913T045115Z`, original `ppo.final.pt` (released as
`v4-autonomous-cube2-02-2833-final.pt`). Training used frame-200-gated +0.2-rad
flexion teacher supervision, beta 1 and two passes, **not** the newer
contact-intent gate on this branch. The weights are unchanged by export.

This is a policy-only inference artifact (`inference_only=true`, `optimizer=None`).
Optimizer state, RNG state and private training configuration were stripped;
full optimizer resume is unsupported and rejected by the current resume loader.
Public asset/package/split/ABI/clock provenance remains embedded. The physical
clock is 120-Hz control and 480-Hz physics (four substeps).

Recorded normal GPU evaluation on the trained identity completed 538 steps with
reason 1, 359 loaded-contact frames, 126 loaded airborne frames, maximum bottom
clearance 0.1678966 m and object-path RMSE 0.04495 m. These results demonstrate
loaded lift/transport on this identity, not perfect placement or release.
Training used one identity; five held-out references produced 0/5 airborne
successes. This checkpoint does not establish unseen-reference generalization.

Export checks passed v4 metadata validation, strict full-model loading, exact
`torch.equal` for every source/export tensor, and identical deterministic means
and values on five synthetic 957-D observations. Non-model metadata is plain
JSON-compatible and checked against an explicit public string allowlist.

Normal CPU validation through the command above on this branch exited 0 and
completed 538 steps, reason 1, with all trace states finite and valid. It recorded
359 loaded-contact and 126 loaded airborne frames, peak bottom clearance
0.1681866 m and path RMSE 0.04470 m. CPU and recorded GPU
trajectories differ numerically; the weights were not adjusted.

## SHA-256

- Original training checkpoint: `fd31f46e242be97110efd15d1abfd1024c728d3a74cd1e3bd0ba7bee7e210a95`
- Bundled `policy.pt`: `8e4e7e0723d66be43c73d044b1823a00d252446fc5e76755c753fe25fb6c1502`

The original checkpoint remained unchanged. Verify the bundled files from this
directory with `sha256sum -c SHA256SUMS`.
