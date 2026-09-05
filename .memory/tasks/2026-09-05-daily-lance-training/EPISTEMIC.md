# Epistemic Model

## Phenomenon
The daily-v12 capture must train against the new DexStream physical assets without carrying Lance/PyArrow into a long-lived trainer, while every source trajectory—not merely every pair—is represented.

## Supported mechanism
The source catalog is valid as a complete 100 Hz training set under the user-selected pre180/post180 contract: all 696 rows decoded, all 14 pairs retained, and source-equivalence is exact. Although 251 metadata rows report 99 Hz, the explicit training/reference/control contract is 100 Hz; this is a deliberate normalization accepted after visual inspection, not an inferred per-row clock.

Balanced pair assignment, rather than raw trajectory count, determines the required world count. With 14 pairs and a maximum of 99 trajectories in cube1:03, N1386 gives every pair 99 slots and covers all 696 unique trajectories. N696 covers only 646 because it repeats shorter pairs before exhausting cube1:03.

The MTP is the runtime boundary. Its digest is `666e1f13f879be39c757c8fc2ac2a4f4d6776e98cf36095d200c0828f3f341a9`; both server-local copies pass repeated full-hash assignment validation. Both five-update trainers produced checkpoints with identical DexStream, package, catalog, clock, and padding provenance. The production trainer has no Lance/PyArrow mappings.

Pre-DexStream Server1 assets are archived, not deleted. Same-filesystem rename preserved bytes/inodes; compatibility symlinks preserve legacy reset behavior while external jobs remain. The archive manifest is the authoritative inventory.

## Evidence
- 696 decoded, zero rejected, 14 pairs.
- N1386 source equivalence: 696 unique assignments, package-only digest stable 5/5.
- Server1 and Server2 both completed 5 updates and 332,640 transitions with exit 0 and checkpoints 1–5.
- Server1 production update 1 completed with about 3.1 GiB GPU memory and 55 GiB host RAM available.
- Legacy archive contains 50 recorded trees totaling 58,460,012,372 bytes with zero broken compatibility links.

## Ruled out
A 696-world run does not satisfy “all trajectories”; balanced pair scheduling demonstrably covers only 646 unique trajectories. Direct Lance training is unnecessary and violates the production isolation contract. Strict resume from old-asset checkpoints would bind the wrong physical version.

## Current claim
The daily-v12 data, DexStream assets, MTP, cross-server runtime, and all-trajectory assignment are ready. A fresh 5000-update production run is active on Server1 GPU2 and has crossed checkpoints 25, 50, and 75 without a runtime/resource alert. Completion and learning quality remain empirical outcomes; stable execution proves launch validity, not policy competence. Success remains zero at update 78.

## Most informative next observation
Per-pair reward/contact/success trends through updates 250–500 will distinguish ordinary fresh-policy exploration from a systematic task or reward mismatch. A nonzero success rate, rather than continued process health alone, is the next evidence that learning is working.
