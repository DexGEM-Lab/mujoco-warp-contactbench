# Epistemic Model

## Phenomenon
The daily-v12 capture must train against the new DexStream physical assets without carrying Lance/PyArrow into a long-lived trainer, while every source trajectory—not merely every pair—is represented.

## Supported mechanism
The source catalog is valid as a complete 100 Hz training set under the user-selected pre180/post180 contract: all 696 rows decoded, all 14 pairs retained, and source-equivalence is exact. Although 251 metadata rows report 99 Hz, the explicit training/reference/control contract is 100 Hz; this is a deliberate normalization accepted after visual inspection, not an inferred per-row clock.

Balanced pair assignment, rather than raw trajectory count, determines the coverage floor. With 14 pairs and a maximum of 99 trajectories in cube1:03, every pair needs at least 99 slots. N696 covers only 646 unique trajectories; N1386 covers all 696 but makes the 66,528-transition rollout batch resolve to a pathological minibatch size of 32. N1536, N4096, and N8192 cover every trajectory and preserve the configured 4096 minibatch.

The MTP is the data-decoder boundary. Its digest is `666e1f13f879be39c757c8fc2ac2a4f4d6776e98cf36095d200c0828f3f341a9`; both server-local copies pass repeated full-hash assignment validation. It removes Lance/PyArrow from trainers, but it cannot repair an unstable host. Server2's N8192 process proved this by completing and checkpointing update 25, then jumping to RIP 0 with SIGSEGV while GPU memory and the MTP boundary remained healthy.

Pre-DexStream Server1 assets are archived, not deleted. Same-filesystem rename preserved bytes/inodes; compatibility symlinks preserve legacy reset behavior while external jobs remain. The archive manifest is the authoritative inventory.

## Evidence
- 696 decoded, zero rejected, 14 pairs; package-only digest stable 5/5 on both hosts.
- N1386 covers 696 unique assignments but runs at about 1.5–1.6k transitions/s because its minibatch resolves to 32.
- N1536 directly isolated the mechanism: preserving minibatch 4096 increased aggregate throughput to 16.7k/s with 5.4 GiB peak memory.
- Fresh Server1 N4096 crossed checkpoint 25 at about 9.0–14.8k/s and 8.0 GiB memory.
- Fresh Server2 N8192 reached checkpoint 25 at about 15.2–29.0k/s and 12.5 GiB memory, then exited 139 with a kernel-recorded null instruction-pointer segfault.
- Legacy archive contains 50 recorded trees totaling 58,460,012,372 bytes with zero broken compatibility links.

## Ruled out
A 696-world run does not satisfy all-trajectory coverage. N1386 is logically sufficient but operationally unsuitable because its gcd-derived minibatch is 32; the slowdown is not a GPU-memory limit. Direct Lance training is unnecessary and violates the production isolation contract. The Server2 N8192 failure was not GPU OOM or Lance/PyArrow decode: it occurred with 11.5 GiB GPU memory free and no such mappings. Strict resume from old-asset checkpoints would bind the wrong physical version.

## Current claim
The daily-v12 data, DexStream assets, MTP, and all-trajectory assignment are ready. `dev` is now pushed to GitHub at `d2562c2`. Idle Server2 is updated to that `dev` commit and passes the asset checks. Server1's active N8192 run remains safely on the frozen `deploy/dexstream-d3ed51a` snapshot; its `dev` ref is fetched but the worktree is unchanged. A fail-closed post-run watcher will switch Server1 after `EXIT:0`, process quiescence, and GPU release. Server2 remains unsuitable for production because its prior N8192 run reproduced a host/VM SIGSEGV after checkpoint 25.

## Most informative next observation
N4096 learning and resource behavior through checkpoints 250–500 is the immediate signal. After N4096 completes, Server1 N8192 will provide the valid sample-efficiency comparison; the Server2 checkpoint-25 run is endurance-failure evidence, not a production baseline.
