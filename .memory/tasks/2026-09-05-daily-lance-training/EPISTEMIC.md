# Epistemic Model

## Phenomenon
The daily-v12 capture must train against the new DexStream physical assets without carrying Lance/PyArrow into a long-lived trainer, while every source trajectory—not merely every pair—is represented.

## Supported mechanism
The source catalog is valid as a complete 100 Hz training set under the user-selected pre180/post180 contract: all 696 rows decoded, all 14 pairs retained, and source-equivalence is exact. Although 251 metadata rows report 99 Hz, the explicit training/reference/control contract is 100 Hz; this is a deliberate normalization accepted after visual inspection, not an inferred per-row clock.

Balanced pair assignment, rather than raw trajectory count, determines the coverage floor. With 14 pairs and a maximum of 99 trajectories in cube1:03, every pair needs at least 99 slots. N696 covers only 646 unique trajectories; N1386 covers all 696 but makes the 66,528-transition rollout batch resolve to a pathological minibatch size of 32. N1536, N4096, and N8192 cover every trajectory and preserve the configured 4096 minibatch.

The MTP is the runtime boundary. Its digest is `666e1f13f879be39c757c8fc2ac2a4f4d6776e98cf36095d200c0828f3f341a9`; both server-local copies pass repeated full-hash assignment validation. Cross-server smokes and the active N4096/N8192 trainers use identical DexStream, package, catalog, clock, and padding provenance without Lance/PyArrow mappings.

Pre-DexStream Server1 assets are archived, not deleted. Same-filesystem rename preserved bytes/inodes; compatibility symlinks preserve legacy reset behavior while external jobs remain. The archive manifest is the authoritative inventory.

## Evidence
- 696 decoded, zero rejected, 14 pairs; package-only digest stable 5/5 on both hosts.
- N1386 covers 696 unique assignments but runs at about 1.5–1.6k transitions/s because its minibatch resolves to 32.
- N1536 directly isolated the mechanism: preserving minibatch 4096 increased aggregate throughput to 16.7k/s with 5.4 GiB peak memory.
- Fresh N4096 crossed reset-heavy updates at about 9.1–14.8k/s and 8.0 GiB memory; fresh N8192 crossed the same boundary at about 15.2–29.0k/s and 12.5 GiB memory.
- Legacy archive contains 50 recorded trees totaling 58,460,012,372 bytes with zero broken compatibility links.

## Ruled out
A 696-world run does not satisfy all-trajectory coverage. N1386 is logically sufficient but operationally unsuitable because its gcd-derived minibatch is 32; the slowdown is not a GPU-memory limit. Direct Lance training is unnecessary and violates the production isolation contract. Strict resume from old-asset checkpoints would bind the wrong physical version.

## Current claim
The daily-v12 data, DexStream assets, MTP, and all-trajectory assignment are ready. The user-authorized N1386 run was stopped with its checkpoint history preserved. Fresh N4096 production is active on Server1 GPU2 and fresh N8192 production is active on Server2 GPU0; both passed the first reset-heavy boundary with healthy resources and no runtime error. Completion and learning quality remain empirical outcomes.

## Most informative next observation
The first update-25 checkpoints and per-pair reward/contact/success trajectories will establish comparable durable baselines. Later sample-matched comparison—not equal update numbers—must account for N8192 processing twice as many transitions per update as N4096.
