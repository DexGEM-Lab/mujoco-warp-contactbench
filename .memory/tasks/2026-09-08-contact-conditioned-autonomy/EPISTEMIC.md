# Current model

The active bounded objective is no longer empirical autonomy replication. It is
a runnable, contract-correct v4 single-reference PPO/frozen-evaluation path for
pinned `cube2_02_2833`, validated by N=1 CPU two-update physics. Long training,
server jobs, remote activity, and W&B/network runs remain stopped pending the
parent launch decision.

v4 raw state is 957-D. The registered PointNet converts only the raw cloud to
the 829-D actor/value feature inside `AutonomyActorCritic`; PPO memory retains
raw 957 observations. All 28 commands remain policy-owned. PPO records raw
Normal samples and their raw Gaussian likelihoods; clipping occurs only at the
physical adapter boundary. The direct CPU two-update run produced finite losses,
gradients/Adam state and a v4 checkpoint. This validates wiring, not learning.

The PPO loop preserves the mature RlGamesPPO GAE/clip/Normal/Adam path. It
records post-step terminal observations/rewards before `prepare_action`; reset
is driven solely by `runtime.last_done`. Physical terminal transitions do not
bootstrap; rollout cuts do. `pending_reset` is retained only as a runtime
structural field and cannot control PPO resets.

Frozen load rejects old v3/538-D contracts and validates model architecture plus
package/catalog/manifest/split provenance. v4 checkpoints contain raw-action
sampling metadata, PointNet state, optimizer state, config, source/asset and
package provenance, and available RNG. Cache float hashes may differ across
supported builds; they are recorded but not equality-gated. Compatibility gates
source/asset/package/ABI/clock contracts instead.

The current v4 cache/state model remains: compiled hand mesh surfaces are
Warp-FK-derived, object collision geometry is body-local, raw SI q drives
control/reward, and cache geometry/velocities are physically named. The B4096
allocation contract remains `--ccd-contacts-per-world 121`, global CCD scratch
495616 and `naconmax >= naccdmax`, with `njmax=512` per world.

N1 result: two updates × eight control transitions on CPU for frame-0
`cube2_02_2833` were finite/valid with rewards 2.213626 and 2.188532;
hand-object force was zero, as expected at 16 untrained transitions. A frozen
four-control evaluation loaded the produced v4 model+PointNet checkpoint and
returned 8.824434 with no natural termination in those four controls. This is
not a reward-tuning signal and does not change physics/reward/clock/geometry.

The next meaningful decision is whether the parent authorizes a real GPU/W&B
training launch. That decision must specify the experiment budget; the default
CLI exposes B4096 workspace/CCD settings but has not been launched here.
